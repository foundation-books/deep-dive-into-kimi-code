"""Execution, cache, and proof helpers for the Chapter 6 reference oracle."""

from __future__ import annotations

from dataclasses import replace
from math import isclose, sqrt
from typing import Iterable, Sequence

from .reference_core import (
    _finish_position,
    _require_width,
    _score,
    dot,
    matvec,
    project_token,
    running_inputs,
    running_weights,
    sigmoid,
    softmax,
    transpose,
    validate_record,
)
from .reference_types import (
    CacheIdentity,
    CompactRecord,
    Dimensions,
    LatentCache,
    PositionTrace,
    Projection,
    RUNNING_DIMS,
    SequenceTrace,
    Vector,
    Weights,
)


def reference_forward(
    inputs: Iterable[tuple[str, Sequence[float]]],
    weights: Weights | None = None,
    dims: Dimensions = RUNNING_DIMS,
) -> SequenceTrace:
    """Evaluate by reconstructing every causally available key and value."""

    active = running_weights(dims) if weights is None else weights
    projections: list[Projection] = []
    traces: list[PositionTrace] = []
    for position, (label, hidden) in enumerate(inputs, start=1):
        current = project_token(hidden, active, dims)
        projections.append(current)
        head_probabilities: list[Vector] = []
        contexts: list[Vector] = []
        for head in range(dims.heads):
            scores = tuple(
                _score(current, head, past.keys[head], past.record.shared_key, dims)
                for past in projections
            )
            probabilities = softmax(scores)
            context = tuple(
                sum(
                    probability * past.values[head][channel]
                    for probability, past in zip(
                        probabilities, projections, strict=True
                    )
                )
                for channel in range(dims.value)
            )
            head_probabilities.append(probabilities)
            contexts.append(context)
        traces.append(
            _finish_position(
                position=position,
                label=label,
                projection=current,
                probabilities=tuple(head_probabilities),
                contexts=tuple(contexts),
                weights=active,
            )
        )
    return SequenceTrace(tuple(traces))


def absorbed_position(
    label: str,
    hidden: Sequence[float],
    records: Sequence[CompactRecord],
    weights: Weights | None = None,
    dims: Dimensions = RUNNING_DIMS,
    *,
    position: int | None = None,
) -> PositionTrace:
    """Evaluate one query from compact records using both legal absorptions."""

    if not records:
        raise ValueError("the current token record must be present")
    active = running_weights(dims) if weights is None else weights
    current = project_token(hidden, active, dims)
    for record in records:
        validate_record(record, dims)
    head_probabilities: list[Vector] = []
    contexts: list[Vector] = []
    for head in range(dims.heads):
        absorbed_query = matvec(
            transpose(active.k_up[head]), current.query_content[head]
        )
        scores = tuple(
            (
                dot(absorbed_query, record.latent)
                + dot(current.query_shared[head], record.shared_key)
            )
            / sqrt(dims.query_head)
            for record in records
        )
        probabilities = softmax(scores)
        aggregate = tuple(
            sum(
                probability * record.latent[channel]
                for probability, record in zip(probabilities, records, strict=True)
            )
            for channel in range(dims.latent)
        )
        contexts.append(matvec(active.v_up[head], aggregate))
        head_probabilities.append(probabilities)
    return _finish_position(
        position=len(records) if position is None else position,
        label=label,
        projection=current,
        probabilities=tuple(head_probabilities),
        contexts=tuple(contexts),
        weights=active,
    )


def absorbed_forward(
    inputs: Iterable[tuple[str, Sequence[float]]],
    weights: Weights | None = None,
    dims: Dimensions = RUNNING_DIMS,
) -> SequenceTrace:
    """Evaluate the causal sequence while persisting compact records only."""

    active = running_weights(dims) if weights is None else weights
    records: list[CompactRecord] = []
    traces: list[PositionTrace] = []
    for position, (label, hidden) in enumerate(inputs, start=1):
        records.append(project_token(hidden, active, dims).record)
        traces.append(
            absorbed_position(label, hidden, records, active, dims, position=position)
        )
    return SequenceTrace(tuple(traces))


def new_cache(
    request_id: str = "running-request",
    model_revision: str = "toy-v1",
    layer: int = 1,
) -> LatentCache:
    """Create a prefix-zero cache with a fully specified identity."""

    return LatentCache(CacheIdentity(request_id, model_revision, layer, 0))


def validate_cache(
    cache: LatentCache,
    dims: Dimensions = RUNNING_DIMS,
    *,
    expected: CacheIdentity | None = None,
) -> None:
    """Validate record count, widths, and optionally every identity field."""

    if cache.identity.endpoint != len(cache.records):
        raise ValueError("cache endpoint and record count disagree")
    if cache.identity.layer < 1:
        raise ValueError("cache layer must be positive")
    if not cache.identity.convention or not cache.identity.dtype:
        raise ValueError("cache convention and dtype must be nonempty")
    for record in cache.records:
        validate_record(record, dims)
    if expected is not None and cache.identity != expected:
        raise ValueError("cache identity does not match the requested continuation")


def append_record(
    cache: LatentCache,
    record: CompactRecord,
    dims: Dimensions = RUNNING_DIMS,
    *,
    accepted_position: int,
) -> LatentCache:
    """Append exactly one current-token record at the next prefix endpoint."""

    validate_cache(cache, dims)
    validate_record(record, dims)
    if accepted_position != cache.identity.endpoint + 1:
        raise ValueError("cache append is not the next accepted position")
    return LatentCache(
        replace(cache.identity, endpoint=accepted_position),
        cache.records + (record,),
    )


def prefill(
    inputs: Iterable[tuple[str, Sequence[float]]],
    weights: Weights | None = None,
    dims: Dimensions = RUNNING_DIMS,
    *,
    cache: LatentCache | None = None,
) -> tuple[SequenceTrace, LatentCache]:
    """Evaluate a fresh prefix and retain one compact record per token."""

    active = running_weights(dims) if weights is None else weights
    current = new_cache() if cache is None else cache
    validate_cache(current, dims)
    traces: list[PositionTrace] = []
    for label, hidden in inputs:
        projection = project_token(hidden, active, dims)
        current = append_record(
            current,
            projection.record,
            dims,
            accepted_position=current.identity.endpoint + 1,
        )
        traces.append(
            absorbed_position(
                label,
                hidden,
                current.records,
                active,
                dims,
                position=current.identity.endpoint,
            )
        )
    return SequenceTrace(tuple(traces)), current


def decode_one(
    token: tuple[str, Sequence[float]],
    cache: LatentCache,
    weights: Weights | None = None,
    dims: Dimensions = RUNNING_DIMS,
    *,
    expected: CacheIdentity | None = None,
) -> tuple[PositionTrace, LatentCache]:
    """Append and evaluate one answer token from an accepted-prefix cache."""

    active = running_weights(dims) if weights is None else weights
    validate_cache(cache, dims, expected=expected)
    label, hidden = token
    record = project_token(hidden, active, dims).record
    updated = append_record(
        cache,
        record,
        dims,
        accepted_position=cache.identity.endpoint + 1,
    )
    trace = absorbed_position(
        label,
        hidden,
        updated.records,
        active,
        dims,
        position=updated.identity.endpoint,
    )
    return trace, updated


def causal_perturbation(
    start: int,
    delta: float = 3.0,
) -> tuple[SequenceTrace, SequenceTrace]:
    """Perturb a zero-indexed suffix while leaving earlier inputs unchanged."""

    inputs = running_inputs()
    changed = tuple(
        (label, tuple(value + delta for value in hidden) if index >= start else hidden)
        for index, (label, hidden) in enumerate(inputs)
    )
    return absorbed_forward(inputs), absorbed_forward(changed)


def permute_records(
    records: Sequence[CompactRecord], permutation: Sequence[int]
) -> tuple[CompactRecord, ...]:
    """Apply a validated simultaneous permutation to position-free records."""

    if sorted(permutation) != list(range(len(records))):
        raise ValueError("permutation must contain every record index once")
    return tuple(records[index] for index in permutation)


def gate_fusion_counterexample(
    weights: Weights | None = None,
    dims: Dimensions = RUNNING_DIMS,
) -> tuple[Vector, Vector]:
    """Produce two outputs for the same head vector but two input-dependent gates.

    A fixed linear map of the head vector alone must return one value for that
    vector, so unequal outputs disprove a general fusion of the dynamic gate
    into the output projection.
    """

    active = running_weights(dims) if weights is None else weights
    head_vector = (1.0,) + (0.0,) * (dims.attention_output - 1)
    hidden_a = (0.0,) * dims.model
    hidden_b = (2.0,) + (0.0,) * (dims.model - 1)

    def output(hidden: Vector) -> Vector:
        gate = tuple(sigmoid(value) for value in matvec(active.gate, hidden))
        return matvec(
            active.output,
            tuple(value * item for value, item in zip(head_vector, gate, strict=True)),
        )

    return output(hidden_a), output(hidden_b)


def logical_cache_elements(prefix_length: int, dims: Dimensions = RUNNING_DIMS) -> int:
    """Count compact logical cache scalars, excluding allocator overhead."""

    if prefix_length < 0:
        raise ValueError("prefix length cannot be negative")
    return prefix_length * dims.compact_record


def materialized_cache_elements(
    prefix_length: int, dims: Dimensions = RUNNING_DIMS
) -> int:
    """Count fully reconstructed per-head K/V scalars for comparison."""

    if prefix_length < 0:
        raise ValueError("prefix length cannot be negative")
    return prefix_length * dims.materialized_record


def max_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the largest elementwise absolute difference."""
    _require_width(left, len(right), "comparison left operand")
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def traces_close(
    left: SequenceTrace, right: SequenceTrace, tolerance: float = 1e-11
) -> bool:
    """Return whether two traces agree in outputs and weights within tolerance."""
    if len(left.positions) != len(right.positions):
        return False
    return all(
        all(
            isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
            for a, b in zip(one.output, two.output, strict=True)
        )
        and all(
            all(
                isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
                for a, b in zip(p, q, strict=True)
            )
            for p, q in zip(one.weights, two.weights, strict=True)
        )
        for one, two in zip(left.positions, right.positions, strict=True)
    )


__all__ = [
    "absorbed_forward",
    "absorbed_position",
    "append_record",
    "causal_perturbation",
    "decode_one",
    "gate_fusion_counterexample",
    "logical_cache_elements",
    "materialized_cache_elements",
    "max_abs_difference",
    "new_cache",
    "permute_records",
    "prefill",
    "reference_forward",
    "traces_close",
    "validate_cache",
]
