"""Dependency-free reference semantics for Chapter 5 hybrid attention.

The module models the chapter's contracts, not production KDA or MLA.  KDA
layers own fixed-extent opaque vectors; global layers own ordered two-scalar
records.  Both operators are deterministic and causal so schedule, state
identity, continuation, and accounting claims can be checked exactly with
fractions.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Iterable, Sequence

from .contracts import (
    RUNNING_GLOBAL_RECORD_WIDTH,
    RUNNING_HEADS,
    RUNNING_KEY_WIDTH,
    RUNNING_MODEL_WIDTH,
    RUNNING_SCHEDULE,
    RUNNING_VALUE_WIDTH,
    GlobalLayerCache,
    HybridPrefixState,
    HybridTrace,
    KdaLayerState,
    LayerStep,
    PositionTrace,
    TokenInput,
    Vector,
    f,
    fraction_text as fraction_text,
    hybrid_prefill_attention_flops as hybrid_prefill_attention_flops,
    k3_schedule as k3_schedule,
    logical_footprint as logical_footprint,
    ratio_inventory as ratio_inventory,
    relative_global_cache_slope as relative_global_cache_slope,
    repeating_schedule as repeating_schedule,
    validate_repeating_inventory as validate_repeating_inventory,
    validate_schedule,
)


RUNNING_LABELS = ("v1", "v2", "v3", "v4", "x1", "x2", "x3", "x4", "x5", "x6")
DECODE_LABEL = "square"


def _zero_vector(width: int) -> Vector:
    return tuple(f(0) for _ in range(width))


def _validate_vector(vector: Sequence[Fraction], width: int, name: str) -> None:
    if len(vector) != width:
        raise ValueError(f"{name} must have width {width}")


def token_vector(position: int) -> Vector:
    """Create one deterministic width-eight vector without semantic channels."""

    if position < 1:
        raise ValueError("positions are one-indexed")
    return tuple(
        f(((position * (coordinate + 2) + 3 * coordinate) % 17) - 8, 8)
        for coordinate in range(RUNNING_MODEL_WIDTH)
    )


def running_inputs() -> tuple[TokenInput, ...]:
    """Return the ten accepted inputs of the shared request."""

    return tuple(
        TokenInput(label, token_vector(position))
        for position, label in enumerate(RUNNING_LABELS, start=1)
    )


def decode_input() -> TokenInput:
    """Return the first generated token input at position eleven."""

    return TokenInput(DECODE_LABEL, token_vector(11))


def empty_state(schedule: Sequence[str] = RUNNING_SCHEDULE) -> HybridPrefixState:
    """Create the complete prefix-zero state."""

    typed = validate_schedule(schedule)
    kda_states = tuple(
        KdaLayerState(layer, 0, _zero_vector(RUNNING_MODEL_WIDTH))
        for layer, layer_type in enumerate(typed, start=1)
        if layer_type == "K"
    )
    global_caches = tuple(
        GlobalLayerCache(layer, 0, ())
        for layer, layer_type in enumerate(typed, start=1)
        if layer_type == "G"
    )
    return HybridPrefixState(typed, 0, kda_states, global_caches)


def validate_complete_state(state: HybridPrefixState) -> None:
    """Validate layer ownership and one shared accepted-prefix endpoint."""

    typed = validate_schedule(state.schedule)
    expected_kda = tuple(
        layer for layer, layer_type in enumerate(typed, start=1) if layer_type == "K"
    )
    expected_global = tuple(
        layer for layer, layer_type in enumerate(typed, start=1) if layer_type == "G"
    )
    observed_kda = tuple(item.layer for item in state.kda_states)
    observed_global = tuple(item.layer for item in state.global_caches)
    if observed_kda != expected_kda or observed_global != expected_global:
        raise ValueError("state components do not match schedule ownership")
    for kda_state in state.kda_states:
        if kda_state.prefix_length != state.prefix_length:
            raise ValueError("KDA state prefix endpoint is inconsistent")
        _validate_vector(kda_state.values, RUNNING_MODEL_WIDTH, "KDA state")
    for global_cache in state.global_caches:
        if global_cache.prefix_length != state.prefix_length:
            raise ValueError("global cache prefix endpoint is inconsistent")
        if len(global_cache.records) != state.prefix_length:
            raise ValueError("global cache record count is inconsistent")
        for record in global_cache.records:
            _validate_vector(record, RUNNING_GLOBAL_RECORD_WIDTH, "global record")


def _kda_step(
    hidden: Vector,
    state: KdaLayerState,
    *,
    position: int,
) -> tuple[Vector, KdaLayerState]:
    """Advance one fixed-extent abstract KDA memory."""

    _validate_vector(hidden, RUNNING_MODEL_WIDTH, "hidden vector")
    if state.prefix_length != position - 1:
        raise ValueError("KDA state does not precede the current position")
    retained_scale = f(state.layer + 1, state.layer + 2)
    updated = tuple(
        retained_scale * previous + value + f(position, 20 * (coordinate + 1))
        for coordinate, (previous, value) in enumerate(
            zip(state.values, hidden, strict=True)
        )
    )
    output = tuple(
        f(1, 2) * (value + memory)
        for value, memory in zip(hidden, updated, strict=True)
    )
    return output, KdaLayerState(state.layer, position, updated)


def _global_record(hidden: Vector) -> Vector:
    """Compress the toy hidden vector into one opaque two-scalar cache record."""

    even = sum(hidden[0::2], f(0)) / 4
    odd = sum(hidden[1::2], f(0)) / 4
    return (even, odd)


def _global_step(
    hidden: Vector,
    cache: GlobalLayerCache,
    *,
    position: int,
) -> tuple[Vector, GlobalLayerCache]:
    """Append the current record, then read every causally legal record."""

    _validate_vector(hidden, RUNNING_MODEL_WIDTH, "hidden vector")
    if cache.prefix_length != position - 1:
        raise ValueError("global cache does not precede the current position")
    records = cache.records + (_global_record(hidden),)
    context = tuple(
        sum((record[index] for record in records), f(0)) / len(records)
        for index in range(RUNNING_GLOBAL_RECORD_WIDTH)
    )
    output = tuple(
        value + context[coordinate % RUNNING_GLOBAL_RECORD_WIDTH] / cache.layer
        for coordinate, value in enumerate(hidden)
    )
    return output, GlobalLayerCache(cache.layer, position, records)


def process_token(
    token: TokenInput,
    state: HybridPrefixState,
    *,
    overwrite_global_cache: bool = False,
) -> tuple[PositionTrace, HybridPrefixState]:
    """Advance every layer once in depth order for one accepted token."""

    validate_complete_state(state)
    _validate_vector(token.values, RUNNING_MODEL_WIDTH, "token input")
    position = state.prefix_length + 1
    kda_by_layer = {item.layer: item for item in state.kda_states}
    global_by_layer = {item.layer: item for item in state.global_caches}
    hidden = token.values
    steps: list[LayerStep] = []
    for layer, layer_type in enumerate(state.schedule, start=1):
        if layer_type == "K":
            hidden, updated = _kda_step(hidden, kda_by_layer[layer], position=position)
            kda_by_layer[layer] = updated
            persistent = RUNNING_HEADS * RUNNING_KEY_WIDTH * RUNNING_VALUE_WIDTH
        else:
            previous = global_by_layer[layer]
            hidden, updated_cache = _global_step(hidden, previous, position=position)
            if overwrite_global_cache and previous.records:
                updated_cache = replace(
                    updated_cache,
                    records=previous.records[:-1] + (updated_cache.records[-1],),
                )
            global_by_layer[layer] = updated_cache
            persistent = len(updated_cache.records) * RUNNING_GLOBAL_RECORD_WIDTH
        steps.append(LayerStep(layer, layer_type, hidden, persistent, position))
    next_state = HybridPrefixState(
        state.schedule,
        position,
        tuple(kda_by_layer[layer] for layer in sorted(kda_by_layer)),
        tuple(global_by_layer[layer] for layer in sorted(global_by_layer)),
    )
    validate_complete_state(next_state)
    return PositionTrace(position, token.label, tuple(steps), hidden), next_state


def run_sequence(
    sequence: Iterable[TokenInput],
    *,
    state: HybridPrefixState | None = None,
    schedule: Sequence[str] = RUNNING_SCHEDULE,
) -> HybridTrace:
    """Run a causal prefill or continuation from one complete state."""

    current = empty_state(schedule) if state is None else state
    validate_complete_state(current)
    traces: list[PositionTrace] = []
    for token in sequence:
        traced, current = process_token(token, current)
        traces.append(traced)
    return HybridTrace(tuple(traces), current)


def full_running_trace(*, include_decode: bool = False) -> HybridTrace:
    """Evaluate the frozen request from prefix zero."""

    sequence = running_inputs() + ((decode_input(),) if include_decode else ())
    return run_sequence(sequence)


def cached_decode_trace() -> tuple[HybridTrace, HybridTrace]:
    """Return prefill and one-token continuation traces."""

    prefill = full_running_trace()
    continuation = run_sequence((decode_input(),), state=prefill.final_state)
    return prefill, continuation


def replace_suffix(
    sequence: Sequence[TokenInput],
    start: int,
    delta: Fraction,
) -> tuple[TokenInput, ...]:
    """Perturb hidden inputs from a zero-indexed suffix boundary."""

    if not 0 <= start <= len(sequence):
        raise ValueError("suffix boundary is out of range")
    changed: list[TokenInput] = []
    for index, token in enumerate(sequence):
        if index < start:
            changed.append(token)
        else:
            changed.append(
                TokenInput(
                    token.label,
                    tuple(value + delta for value in token.values),
                )
            )
    return tuple(changed)


def flatten_outputs(trace: HybridTrace) -> tuple[Fraction, ...]:
    """Flatten final position outputs for exact comparisons."""

    return tuple(value for position in trace.positions for value in position.output)


def causal_access_matrix(
    length: int,
    *,
    include_diagonal: bool = True,
    allow_future: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Construct a binary global-layer legality matrix."""

    if length < 1:
        raise ValueError("length must be positive")
    rows = []
    for query in range(length):
        row = []
        for key in range(length):
            legal = key < query or (include_diagonal and key == query)
            if allow_future and key == query + 1:
                legal = True
            row.append(int(legal))
        rows.append(tuple(row))
    return tuple(rows)


def validate_causal_access_matrix(matrix: Sequence[Sequence[int]]) -> None:
    """Require the inclusive causal lower triangle exactly."""

    length = len(matrix)
    if length < 1 or any(len(row) != length for row in matrix):
        raise ValueError("access matrix must be nonempty and square")
    expected = causal_access_matrix(length)
    if tuple(tuple(row) for row in matrix) != expected:
        raise ValueError("access matrix must equal the inclusive causal triangle")


def mixed_family_state(
    kda_source: HybridPrefixState,
    global_source: HybridPrefixState,
    *,
    declared_prefix: int,
) -> HybridPrefixState:
    """Construct a deliberately mixed state for restoration-fault tests."""

    if kda_source.schedule != global_source.schedule:
        raise ValueError("source schedules must match")
    return HybridPrefixState(
        kda_source.schedule,
        declared_prefix,
        kda_source.kda_states,
        global_source.global_caches,
    )
