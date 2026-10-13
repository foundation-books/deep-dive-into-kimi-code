"""Depth, sequence, channel, and readout operators for the K3 oracle."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .reference_types import (
    GLOBAL_RECORD_WIDTH,
    MODEL_WIDTH,
    ROUTED_EXPERTS,
    VOCABULARY,
    DepthState,
    GlobalLayerState,
    Grid,
    KdaLayerState,
    RouteRecord,
    Vector,
    _validate_grid,
    _zero_vector,
    f,
)
from .reference_grid import add as _add
from .reference_grid import new_depth_state as new_depth_state
from .reference_grid import scale as _scale


def validate_depth_state(
    state: DepthState,
    *,
    expected_forward_id: str,
    block_size: int,
    maximum_points: int,
) -> None:
    """Validate identity, ordering, and partial-block extent."""

    if state.forward_id != expected_forward_id:
        raise ValueError("depth history belongs to another forward")
    _validate_grid(state.embedding, "depth embedding")
    if len(state.completed) != len(state.completed_points):
        raise ValueError("completed depth values and point sets disagree")
    cursor = 1
    block_points = 2 * block_size
    for value, points in zip(state.completed, state.completed_points, strict=True):
        _validate_grid(value, "completed depth summary")
        expected = tuple(range(cursor, cursor + block_points))
        if points != expected:
            raise ValueError("completed depth group is missing or reordered")
        cursor += block_points
    if state.partial is None:
        if state.partial_points:
            raise ValueError("empty depth partial carries point identities")
    else:
        _validate_grid(state.partial, "depth partial")
        expected = tuple(range(cursor, cursor + len(state.partial_points)))
        if state.partial_points != expected:
            raise ValueError("depth partial is stale, duplicated, or noncontiguous")
        if not state.partial_points or len(state.partial_points) > block_points:
            raise ValueError("depth partial has illegal extent")
        cursor += len(state.partial_points)
    if cursor - 1 > maximum_points:
        raise ValueError("depth history contains a future transformation point")


def depth_sources(state: DepthState) -> tuple[tuple[Grid, ...], tuple[str, ...]]:
    """Return embedding, completed summaries, and an optional current partial."""

    values: list[Grid] = [state.embedding]
    labels = ["embedding"]
    for index, completed in enumerate(state.completed, start=1):
        values.append(completed)
        labels.append(f"block-{index}")
    if state.partial is not None:
        values.append(state.partial)
        labels.append(f"partial-{len(state.completed) + 1}")
    return tuple(values), tuple(labels)


def _depth_mix_primary(sources: Sequence[Grid]) -> Grid:
    """Mix sources with exact positive unit-sum depth weights."""

    if not sources:
        raise ValueError("depth mixing needs at least one source")
    token_count = len(sources[0])
    if any(len(source) != token_count for source in sources):
        raise ValueError("depth sources must share a token grid")
    denominator = f(sum(range(1, len(sources) + 1)))
    output = tuple(_zero_vector() for _ in range(token_count))
    for ordinal, source in enumerate(sources, start=1):
        output = _add(output, _scale(source, f(ordinal) / denominator))
    return output


def _depth_mix_oracle(sources: Sequence[Grid]) -> Grid:
    """Independent scalar-loop depth mixer for equivalence checks."""

    if not sources:
        raise ValueError("depth mixing needs at least one source")
    token_count = len(sources[0])
    if any(len(source) != token_count for source in sources):
        raise ValueError("depth sources must share a token grid")
    denominator = f(sum(range(1, len(sources) + 1)))
    rows = []
    for token in range(token_count):
        row = []
        for coordinate in range(MODEL_WIDTH):
            value = f(0)
            for ordinal, source in enumerate(sources, start=1):
                value += f(ordinal) * source[token][coordinate] / denominator
            row.append(value)
        rows.append(tuple(row))
    return tuple(rows)


def append_depth_point(
    state: DepthState,
    output: Grid,
    point: int,
    *,
    expected_forward_id: str,
    block_size: int,
    maximum_points: int,
) -> DepthState:
    """Append exactly the next transformation output to the live partial."""

    validate_depth_state(
        state,
        expected_forward_id=expected_forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    represented = sum(len(points) for points in state.completed_points)
    represented += len(state.partial_points)
    if point != represented + 1:
        raise ValueError("depth point is missing, duplicated, or out of order")
    partial = output if state.partial is None else _add(state.partial, output)
    updated = replace(
        state,
        partial=partial,
        partial_points=state.partial_points + (point,),
    )
    validate_depth_state(
        updated,
        expected_forward_id=expected_forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    return updated


def commit_depth_group(
    state: DepthState,
    *,
    expected_forward_id: str,
    block_size: int,
    maximum_points: int,
) -> DepthState:
    """Commit one complete decoder-layer group and clear the partial."""

    validate_depth_state(
        state,
        expected_forward_id=expected_forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    if state.partial is None or len(state.partial_points) != 2 * block_size:
        raise ValueError("only a complete nonempty depth group may be committed")
    updated = replace(
        state,
        completed=state.completed + (state.partial,),
        completed_points=state.completed_points + (state.partial_points,),
        partial=None,
        partial_points=(),
    )
    validate_depth_state(
        updated,
        expected_forward_id=expected_forward_id,
        block_size=block_size,
        maximum_points=maximum_points,
    )
    return updated


def _kda_segment(
    hidden: Grid,
    state: KdaLayerState,
    *,
    start_endpoint: int,
) -> tuple[Grid, KdaLayerState]:
    """Advance one fixed-extent exact-rational KDA stand-in."""

    if state.endpoint != start_endpoint:
        raise ValueError("KDA state does not begin at the committed endpoint")
    memory = state.memory
    output = []
    for offset, row in enumerate(hidden, start=1):
        position = start_endpoint + offset
        retention = f(state.layer + 2, state.layer + 3)
        memory = tuple(
            retention * previous + value + f(position, 50 * (coordinate + 1))
            for coordinate, (previous, value) in enumerate(
                zip(memory, row, strict=True)
            )
        )
        output.append(
            tuple(
                (value + remembered) / 2
                for value, remembered in zip(row, memory, strict=True)
            )
        )
    return tuple(output), KdaLayerState(
        state.layer,
        start_endpoint + len(hidden),
        state.prefix_identity,
        memory,
    )


def _compact_record(row: Vector) -> Vector:
    """Compress one width-eight row to an opaque two-scalar logical record."""

    return (
        sum(row[0::2], f(0)) / 4,
        sum(row[1::2], f(0)) / 4,
    )


def _global_segment(
    hidden: Grid,
    state: GlobalLayerState,
    *,
    start_endpoint: int,
) -> tuple[Grid, GlobalLayerState]:
    """Append and causally read compact global-attention records."""

    if state.endpoint != start_endpoint:
        raise ValueError("global state does not begin at the committed endpoint")
    records = state.records
    output = []
    for row in hidden:
        records = records + (_compact_record(row),)
        context = tuple(
            sum((record[index] for record in records), f(0)) / len(records)
            for index in range(GLOBAL_RECORD_WIDTH)
        )
        output.append(
            tuple(
                value + context[coordinate % GLOBAL_RECORD_WIDTH] / state.layer
                for coordinate, value in enumerate(row)
            )
        )
    return tuple(output), GlobalLayerState(
        state.layer,
        start_endpoint + len(hidden),
        state.prefix_identity,
        records,
    )


def _dense_channel(hidden: Grid, layer: int) -> Grid:
    """Apply the toy initial dense channel map."""

    return tuple(
        tuple(
            f(layer + 1, layer + 2) * value + row[(coordinate - 1) % MODEL_WIDTH] / 7
            for coordinate, value in enumerate(row)
        )
        for row in hidden
    )


def _selected_experts(position: int, layer: int) -> tuple[int, int]:
    first = (position + 2 * layer) % ROUTED_EXPERTS
    second = (first + 1 + layer % (ROUTED_EXPERTS - 1)) % ROUTED_EXPERTS
    if second == first:
        second = (second + 1) % ROUTED_EXPERTS
    return first, second


def _expert_vector(row: Vector, expert: int, layer: int) -> Vector:
    scale = f(expert + 2, expert + 3)
    return tuple(
        scale * value + f((expert + 1) * (coordinate + 1), 100 * layer)
        for coordinate, value in enumerate(row)
    )


def _shared_vector(row: Vector, layer: int) -> Vector:
    return tuple(
        value / 3 + f(coordinate + 1, 80 * layer)
        for coordinate, value in enumerate(row)
    )


def _moe_channel_primary(
    hidden: Grid,
    *,
    layer: int,
    positions: Sequence[int],
    origins: Sequence[str],
) -> tuple[Grid, tuple[RouteRecord, ...]]:
    """Token-major routed channel path with immutable route tags."""

    output = []
    routes: list[RouteRecord] = []
    for row, position, origin in zip(hidden, positions, origins, strict=True):
        experts = _selected_experts(position, layer)
        weights = (f(2, 3), f(1, 3))
        routed = _zero_vector()
        for ordinal, (expert, weight) in enumerate(
            zip(experts, weights, strict=True), start=1
        ):
            routes.append(RouteRecord(origin, position, layer, expert, ordinal, weight))
            routed = tuple(
                current + weight * value
                for current, value in zip(
                    routed, _expert_vector(row, expert, layer), strict=True
                )
            )
        output.append(
            tuple(
                shared + selected
                for shared, selected in zip(
                    _shared_vector(row, layer), routed, strict=True
                )
            )
        )
    return tuple(output), tuple(routes)


def _moe_channel_oracle(
    hidden: Grid,
    *,
    layer: int,
    positions: Sequence[int],
    origins: Sequence[str],
) -> tuple[Grid, tuple[RouteRecord, ...]]:
    """Expert-major dispatch/combine oracle with the same route multiset."""

    records = []
    for token, (position, origin) in enumerate(zip(positions, origins, strict=True)):
        experts = _selected_experts(position, layer)
        for ordinal, (expert, weight) in enumerate(
            zip(experts, (f(2, 3), f(1, 3)), strict=True), start=1
        ):
            records.append(
                (
                    expert,
                    token,
                    RouteRecord(origin, position, layer, expert, ordinal, weight),
                )
            )
    accumulated = [_zero_vector() for _ in hidden]
    for expert in range(ROUTED_EXPERTS):
        for routed_expert, token, record in records:
            if routed_expert != expert:
                continue
            transformed = _expert_vector(hidden[token], expert, layer)
            accumulated[token] = tuple(
                current + record.weight * value
                for current, value in zip(accumulated[token], transformed, strict=True)
            )
    output = tuple(
        tuple(
            shared + selected
            for shared, selected in zip(_shared_vector(row, layer), routed, strict=True)
        )
        for row, routed in zip(hidden, accumulated, strict=True)
    )
    token_major = tuple(
        record
        for _, _, record in sorted(records, key=lambda item: (item[1], item[2].ordinal))
    )
    return output, token_major


def _readout(grid: Grid) -> Grid:
    """Map width-eight terminal rows to twelve inspectable toy logits."""

    return tuple(
        tuple(
            sum(
                (
                    value * f((coordinate + 1) * (vocab + 2) % 11 - 5, 17)
                    for coordinate, value in enumerate(row)
                ),
                f(0),
            )
            for vocab in range(VOCABULARY)
        )
        for row in grid
    )


__all__ = [
    "_dense_channel",
    "_depth_mix_oracle",
    "_depth_mix_primary",
    "_global_segment",
    "_kda_segment",
    "_moe_channel_oracle",
    "_moe_channel_primary",
    "_readout",
    "append_depth_point",
    "commit_depth_group",
    "depth_sources",
    "new_depth_state",
    "validate_depth_state",
]
