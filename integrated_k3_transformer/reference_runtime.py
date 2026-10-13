"""Integrated forward and accepted-prefix execution for the K3 oracle."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .reference_operators import (
    _dense_channel,
    _depth_mix_oracle,
    _depth_mix_primary,
    _global_segment,
    _kda_segment,
    _moe_channel_oracle,
    _moe_channel_primary,
    _readout,
    append_depth_point,
    commit_depth_group,
    depth_sources,
    new_depth_state,
)
from .reference_runtime_state import ForwardAccumulator as _ForwardAccumulator
from .reference_types import (
    SELECTED_EXPERTS,
    TOY_BLOCK_SIZE,
    TOY_SCHEDULE,
    ForwardTrace,
    Grid,
    LayerCheckpoint,
    PrefixState,
    RouteRecord,
    TokenRow,
    channel_schedule,
    decode_token,
    empty_prefix_state,
    prefix_identity,
    running_tokens,
    validate_input_segment,
    validate_prefix_state,
    validate_schedule,
)


def _mix_depth(sources: Sequence[Grid], oracle: bool) -> Grid:
    return _depth_mix_oracle(sources) if oracle else _depth_mix_primary(sources)


def _advance_sequence(
    accumulator: _ForwardAccumulator,
    hidden: Grid,
    *,
    attention_type: str,
    layer: int,
    start_endpoint: int,
) -> Grid:
    if attention_type == "K":
        output, updated_kda = _kda_segment(
            hidden,
            accumulator.kda_by_layer[layer],
            start_endpoint=start_endpoint,
        )
        accumulator.kda_by_layer[layer] = updated_kda
        return output
    output, updated_global = _global_segment(
        hidden,
        accumulator.global_by_layer[layer],
        start_endpoint=start_endpoint,
    )
    accumulator.global_by_layer[layer] = updated_global
    return output


def _advance_channel(
    hidden: Grid,
    *,
    channel_type: str,
    layer: int,
    positions: Sequence[int],
    origins: Sequence[str],
    oracle: bool,
) -> tuple[Grid, tuple[RouteRecord, ...]]:
    if channel_type == "D":
        return _dense_channel(hidden, layer), ()
    runner = _moe_channel_oracle if oracle else _moe_channel_primary
    output, routes = runner(
        hidden,
        layer=layer,
        positions=positions,
        origins=origins,
    )
    if len(routes) != len(positions) * SELECTED_EXPERTS:
        raise ValueError("routed channel path violated route conservation")
    return output, routes


def _append_point(
    accumulator: _ForwardAccumulator,
    output: Grid,
    *,
    forward_id: str,
    block_size: int,
    maximum_points: int,
) -> None:
    accumulator.depth = append_depth_point(
        accumulator.depth,
        output,
        accumulator.point,
        expected_forward_id=forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    accumulator.point += 1


def _record_checkpoint(
    accumulator: _ForwardAccumulator,
    *,
    layer: int,
    attention_type: str,
    channel_type: str,
    pre_attention_sources: tuple[str, ...],
    pre_channel_sources: tuple[str, ...],
    attention_output: Grid,
    channel_output: Grid,
    routes: tuple[RouteRecord, ...],
) -> None:
    accumulator.routes.extend(routes)
    accumulator.checkpoints.append(
        LayerCheckpoint(
            layer=layer,
            attention_type=attention_type,
            channel_type=channel_type,
            pre_attention_sources=pre_attention_sources,
            pre_channel_sources=pre_channel_sources,
            attention_output=attention_output,
            channel_output=channel_output,
            routes=routes,
            depth_points_after_layer=accumulator.depth.partial_points,
            completed_depth_groups=len(accumulator.depth.completed),
        )
    )


def _commit_complete_group(
    accumulator: _ForwardAccumulator,
    *,
    layer: int,
    forward_id: str,
    block_size: int,
    maximum_points: int,
) -> None:
    if layer % block_size:
        return
    accumulator.depth = commit_depth_group(
        accumulator.depth,
        expected_forward_id=forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )


def _run_layer(
    accumulator: _ForwardAccumulator,
    *,
    layer: int,
    attention_type: str,
    channel_type: str,
    positions: Sequence[int],
    origins: Sequence[str],
    start_endpoint: int,
    forward_id: str,
    block_size: int,
    maximum_points: int,
    oracle: bool,
) -> None:
    sources, pre_attention_labels = depth_sources(accumulator.depth)
    attention_output = _advance_sequence(
        accumulator,
        _mix_depth(sources, oracle),
        attention_type=attention_type,
        layer=layer,
        start_endpoint=start_endpoint,
    )
    _append_point(
        accumulator,
        attention_output,
        forward_id=forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    channel_sources, pre_channel_labels = depth_sources(accumulator.depth)
    channel_output, routes = _advance_channel(
        _mix_depth(channel_sources, oracle),
        channel_type=channel_type,
        layer=layer,
        positions=positions,
        origins=origins,
        oracle=oracle,
    )
    _append_point(
        accumulator,
        channel_output,
        forward_id=forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    _record_checkpoint(
        accumulator,
        layer=layer,
        attention_type=attention_type,
        channel_type=channel_type,
        pre_attention_sources=pre_attention_labels,
        pre_channel_sources=pre_channel_labels,
        attention_output=attention_output,
        channel_output=channel_output,
        routes=routes,
    )
    _commit_complete_group(
        accumulator,
        layer=layer,
        forward_id=forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )


def _finish_forward(
    accumulator: _ForwardAccumulator,
    tokens: Sequence[TokenRow],
    active_state: PrefixState,
    typed_schedule: tuple[str, ...],
    origins: tuple[str, ...],
    *,
    forward_id: str,
    oracle: bool,
) -> ForwardTrace:
    terminal_values, terminal_labels = depth_sources(accumulator.depth)
    terminal = _mix_depth(terminal_values, oracle)
    accepted_labels = active_state.accepted_labels + tuple(
        token.label for token in tokens
    )
    accepted_identity = prefix_identity(accepted_labels)
    next_state = PrefixState(
        schedule=typed_schedule,
        endpoint=active_state.endpoint + len(tokens),
        accepted_labels=accepted_labels,
        kda=tuple(
            replace(
                accumulator.kda_by_layer[layer],
                prefix_identity=accepted_identity,
            )
            for layer in sorted(accumulator.kda_by_layer)
        ),
        global_layers=tuple(
            replace(
                accumulator.global_by_layer[layer],
                prefix_identity=accepted_identity,
            )
            for layer in sorted(accumulator.global_by_layer)
        ),
    )
    validate_prefix_state(next_state)
    return ForwardTrace(
        forward_id=forward_id,
        input_labels=origins,
        start_endpoint=active_state.endpoint,
        checkpoints=tuple(accumulator.checkpoints),
        terminal_sources=terminal_labels,
        terminal_grid=terminal,
        logits=_readout(terminal),
        prefix_state=next_state,
        routes=tuple(accumulator.routes),
    )


def run_forward(
    tokens: Sequence[TokenRow],
    *,
    state: PrefixState | None = None,
    schedule: Sequence[str] = TOY_SCHEDULE,
    block_size: int = TOY_BLOCK_SIZE,
    forward_id: str = "integrated-forward",
    oracle: bool = False,
) -> ForwardTrace:
    """Execute one integrated prefill or continuation segment."""

    active_state = empty_prefix_state(schedule) if state is None else state
    typed_schedule = validate_schedule(schedule)
    if active_state.schedule != typed_schedule:
        raise ValueError("prefix-state schedule and forward schedule disagree")
    validate_input_segment(tokens, active_state)
    if block_size <= 0:
        raise ValueError("depth block size must be positive")
    hidden: Grid = tuple(token.values for token in tokens)
    accumulator = _ForwardAccumulator(
        depth=new_depth_state(forward_id, hidden),
        kda_by_layer={item.layer: item for item in active_state.kda},
        global_by_layer={item.layer: item for item in active_state.global_layers},
        checkpoints=[],
        routes=[],
    )
    positions = tuple(token.position for token in tokens)
    origins = tuple(token.label for token in tokens)
    maximum_points = 2 * len(typed_schedule)
    channel_types = channel_schedule(len(typed_schedule))
    for layer, (attention_type, channel_type) in enumerate(
        zip(typed_schedule, channel_types, strict=True), start=1
    ):
        _run_layer(
            accumulator,
            layer=layer,
            attention_type=attention_type,
            channel_type=channel_type,
            positions=positions,
            origins=origins,
            start_endpoint=active_state.endpoint,
            forward_id=forward_id,
            block_size=block_size,
            maximum_points=maximum_points,
            oracle=oracle,
        )
    return _finish_forward(
        accumulator,
        tokens,
        active_state,
        typed_schedule,
        origins,
        forward_id=forward_id,
        oracle=oracle,
    )


def running_prefill(*, oracle: bool = False) -> ForwardTrace:
    """Execute the ten-position running prefix."""

    return run_forward(
        running_tokens(),
        forward_id="prefill-oracle" if oracle else "prefill-primary",
        oracle=oracle,
    )


def full_prefix_decode() -> ForwardTrace:
    """Recompute all eleven accepted rows from prefix zero."""

    return run_forward(
        running_tokens() + (decode_token(),),
        forward_id="full-prefix-eleven",
    )


def cached_decode() -> tuple[ForwardTrace, ForwardTrace]:
    """Prefill ten rows, then continue with one accepted token."""

    prefill = running_prefill()
    continuation = run_forward(
        (decode_token(),),
        state=prefill.prefix_state,
        forward_id="cached-position-eleven",
    )
    return prefill, continuation


def incremental_equivalence() -> dict[str, object]:
    """Compare the last full-prefix row with cached continuation exactly."""

    full = full_prefix_decode()
    prefill, cached = cached_decode()
    return {
        "status": "pass" if full.logits[-1] == cached.logits[-1] else "fail",
        "full_last_logits": full.logits[-1],
        "cached_logits": cached.logits[-1],
        "prefill_endpoint": prefill.prefix_state.endpoint,
        "cached_endpoint": cached.prefix_state.endpoint,
        "exact": full.logits[-1] == cached.logits[-1],
    }


def integrated_forward_equivalence() -> dict[str, object]:
    """Compare token-major and independent source/expert-major paths."""

    primary = running_prefill()
    oracle = running_prefill(oracle=True)
    passed = (
        primary.terminal_grid == oracle.terminal_grid
        and primary.logits == oracle.logits
        and primary.routes == oracle.routes
    )
    return {
        "status": "pass" if passed else "fail",
        "terminal_equal": primary.terminal_grid == oracle.terminal_grid,
        "logits_equal": primary.logits == oracle.logits,
        "routes_equal": primary.routes == oracle.routes,
        "checkpoints": len(primary.checkpoints),
    }


__all__ = [
    "cached_decode",
    "full_prefix_decode",
    "incremental_equivalence",
    "integrated_forward_equivalence",
    "run_forward",
    "running_prefill",
]
