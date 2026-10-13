"""Full, Block, two-phase, and state-ledger execution for Chapter 7."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence, cast

from .reference_core import (
    _mix_close,
    add_grids,
    max_abs_difference,
    mix_depth,
    mix_depth_oracle,
    running_embedding,
    running_queries,
    transform_grid,
    validate_grid,
)
from .reference_types import (
    BlockState,
    BlockStep,
    BlockTrace,
    DepthIdentity,
    Dimensions,
    FullStep,
    FullTrace,
    Grid,
    MixResult,
    RUNNING_DIMS,
)


Mixer = Callable[[Sequence[float], Sequence[Grid], Dimensions], MixResult]


def _point_name(target: int, dims: Dimensions) -> str:
    point = target % dims.points_per_layer
    if dims.points_per_layer == 2:
        return "pre-attention" if point == 0 else "pre-mlp"
    return f"point-{point + 1}"


def _full_forward_with_mixer(
    mixer: Mixer,
    dims: Dimensions,
) -> FullTrace:
    embedding = running_embedding(dims)
    queries = running_queries(dims)
    sources: list[Grid] = [embedding]
    source_ids: list[str] = ["embedding"]
    steps: list[FullStep] = []
    for target in range(dims.transformations):
        mixture = mixer(queries[target], tuple(sources), dims)
        output = transform_grid(mixture.output, target, dims)
        steps.append(
            FullStep(
                target=target,
                decoder_layer=target // dims.points_per_layer,
                point=_point_name(target, dims),
                source_ids=tuple(source_ids),
                mixture=mixture,
                transformation_output=output,
            )
        )
        sources.append(output)
        source_ids.append(f"z{target + 1}")
    terminal = mixer(queries[-1], tuple(sources), dims)
    return FullTrace(
        embedding=embedding,
        steps=tuple(steps),
        terminal_source_ids=tuple(source_ids),
        terminal=terminal,
    )


def full_forward(dims: Dimensions = RUNNING_DIMS) -> FullTrace:
    """Evaluate Full AttnRes with the primary vectorized-by-token mixer."""

    return _full_forward_with_mixer(mix_depth, dims)


def full_forward_oracle(dims: Dimensions = RUNNING_DIMS) -> FullTrace:
    """Evaluate Full AttnRes with the independent source-major mixer."""

    return _full_forward_with_mixer(mix_depth_oracle, dims)


def new_block_state(
    embedding: Grid | None = None,
    dims: Dimensions = RUNNING_DIMS,
    *,
    forward_id: str = "running-forward",
    model_revision: str = "toy-v1",
) -> BlockState:
    """Create an empty ephemeral depth history for one forward."""

    active_embedding = running_embedding(dims) if embedding is None else embedding
    validate_grid(active_embedding, dims, name="block embedding")
    return BlockState(
        identity=DepthIdentity(
            forward_id=forward_id,
            model_revision=model_revision,
            tokens=dims.tokens,
            width=dims.model,
        ),
        embedding=active_embedding,
    )


def _validated_completed_cursor(
    state: BlockState, dims: Dimensions, block_targets: int
) -> int:
    if len(state.completed) != len(state.completed_indices):
        raise ValueError("completed values and index groups disagree")
    cursor = 0
    for block, (value, indices) in enumerate(
        zip(state.completed, state.completed_indices, strict=True),
        start=1,
    ):
        validate_grid(value, dims, name=f"completed block {block}")
        expected_indices = tuple(range(cursor, cursor + block_targets))
        if indices != expected_indices:
            raise ValueError("completed block is early, stale, or noncontiguous")
        cursor += block_targets
    return cursor


def _validated_partial_cursor(
    state: BlockState, dims: Dimensions, block_targets: int, cursor: int
) -> int:
    if state.partial is None:
        if state.partial_indices:
            raise ValueError("empty partial carries output indices")
        return cursor
    validate_grid(state.partial, dims, name="current partial")
    if not state.partial_indices:
        raise ValueError("nonempty partial lacks output indices")
    expected_partial = tuple(range(cursor, cursor + len(state.partial_indices)))
    if state.partial_indices != expected_partial:
        raise ValueError("partial output indices are missing, stale, or duplicated")
    if len(state.partial_indices) > block_targets:
        raise ValueError("partial exceeds one declared block")
    return cursor + len(state.partial_indices)


def validate_block_state(
    state: BlockState,
    dims: Dimensions = RUNNING_DIMS,
    *,
    expected: DepthIdentity | None = None,
) -> None:
    """Validate identity, shapes, block boundaries, and source conservation."""

    identity = state.identity
    if (
        identity.tokens != dims.tokens
        or identity.width != dims.model
        or identity.lifetime != "one-model-forward"
    ):
        raise ValueError("depth-state identity disagrees with the schedule")
    if not identity.forward_id or not identity.model_revision:
        raise ValueError("depth-state identity fields must be nonempty")
    if expected is not None and identity != expected:
        raise ValueError("depth state belongs to another forward or revision")
    validate_grid(state.embedding, dims, name="state embedding")
    block_targets = dims.block_size * dims.points_per_layer
    cursor = _validated_completed_cursor(state, dims, block_targets)
    cursor = _validated_partial_cursor(state, dims, block_targets, cursor)
    if cursor > dims.transformations:
        raise ValueError("depth state contains future transformation outputs")


def source_partition(
    state: BlockState,
    dims: Dimensions = RUNNING_DIMS,
) -> tuple[tuple[int, ...], ...]:
    """Return output-index groups represented by the current sources."""

    validate_block_state(state, dims)
    groups: list[tuple[int, ...]] = [()]
    groups.extend(state.completed_indices)
    if state.partial is not None:
        groups.append(state.partial_indices)
    return tuple(groups)


def block_sources(
    state: BlockState,
    dims: Dimensions = RUNNING_DIMS,
) -> tuple[tuple[Grid, ...], tuple[str, ...]]:
    """Return the ordered embedding/completed/partial source collection."""

    validate_block_state(state, dims)
    values: list[Grid] = [state.embedding]
    labels: list[str] = ["embedding"]
    for index, completed in enumerate(state.completed, start=1):
        values.append(completed)
        labels.append(f"block-{index}")
    if state.partial is not None:
        values.append(state.partial)
        labels.append(f"partial-{len(state.completed) + 1}")
    return tuple(values), tuple(labels)


def append_partial(
    state: BlockState,
    output: Grid,
    target: int,
    dims: Dimensions = RUNNING_DIMS,
    *,
    expected: DepthIdentity | None = None,
) -> BlockState:
    """Add exactly the next transformation output to the current partial."""

    validate_block_state(state, dims, expected=expected)
    validate_grid(output, dims, name="new transformation output")
    represented = sum(len(indices) for indices in state.completed_indices)
    represented += len(state.partial_indices)
    if target != represented:
        raise ValueError(
            "transformation output is missing, duplicated, or out of order"
        )
    partial = (
        output if state.partial is None else add_grids(state.partial, output, dims)
    )
    updated = replace(
        state,
        partial=partial,
        partial_indices=state.partial_indices + (target,),
    )
    validate_block_state(updated, dims, expected=expected)
    return updated


def commit_partial(
    state: BlockState,
    dims: Dimensions = RUNNING_DIMS,
    *,
    expected: DepthIdentity | None = None,
) -> BlockState:
    """Commit one complete partial exactly once, then reset it."""

    validate_block_state(state, dims, expected=expected)
    required = dims.block_size * dims.points_per_layer
    if state.partial is None or len(state.partial_indices) != required:
        raise ValueError("only one complete nonempty block may be committed")
    updated = replace(
        state,
        completed=state.completed + (state.partial,),
        completed_indices=state.completed_indices + (state.partial_indices,),
        partial=None,
        partial_indices=(),
    )
    validate_block_state(updated, dims, expected=expected)
    return updated


def _block_forward_with_mixer(
    mixer: Mixer,
    dims: Dimensions,
) -> BlockTrace:
    queries = running_queries(dims)
    state = new_block_state(dims=dims)
    expected = state.identity
    steps: list[BlockStep] = []
    for decoder_layer in range(dims.decoder_layers):
        if decoder_layer > 0 and decoder_layer % dims.block_size == 0:
            state = commit_partial(state, dims, expected=expected)
        for point in range(dims.points_per_layer):
            target = decoder_layer * dims.points_per_layer + point
            values, source_ids = block_sources(state, dims)
            mixture = mixer(queries[target], values, dims)
            output = transform_grid(mixture.output, target, dims)
            state = append_partial(
                state,
                output,
                target,
                dims,
                expected=expected,
            )
            steps.append(
                BlockStep(
                    target=target,
                    decoder_layer=decoder_layer,
                    point=_point_name(target, dims),
                    block_index=decoder_layer // dims.block_size,
                    source_ids=source_ids,
                    mixture=mixture,
                    transformation_output=output,
                    state_after=state,
                )
            )
    values, source_ids = block_sources(state, dims)
    terminal = mixer(queries[-1], values, dims)
    if len(source_ids) != dims.terminal_sources:
        raise ValueError("terminal source count disagrees with the block partition")
    return BlockTrace(
        steps=tuple(steps),
        terminal_source_ids=source_ids,
        terminal=terminal,
        terminal_state=state,
    )


def block_forward(dims: Dimensions = RUNNING_DIMS) -> BlockTrace:
    """Evaluate the source-resolved Block AttnRes schedule."""

    return _block_forward_with_mixer(mix_depth, dims)


def block_forward_oracle(dims: Dimensions = RUNNING_DIMS) -> BlockTrace:
    """Evaluate the same schedule with the independent source-major mixer."""

    return _block_forward_with_mixer(mix_depth_oracle, dims)


def traces_close(
    left: FullTrace | BlockTrace,
    right: FullTrace | BlockTrace,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Compare corresponding Full or Block traces."""

    if type(left) is not type(right) or len(left.steps) != len(right.steps):
        return False
    left_steps = cast(Sequence[FullStep | BlockStep], left.steps)
    right_steps = cast(Sequence[FullStep | BlockStep], right.steps)
    for left_step, right_step in zip(left_steps, right_steps, strict=True):
        if (
            left_step.source_ids != right_step.source_ids
            or not _mix_close(left_step.mixture, right_step.mixture, tolerance)
            or max_abs_difference(
                left_step.transformation_output,
                right_step.transformation_output,
            )
            > tolerance
        ):
            return False
    return left.terminal_source_ids == right.terminal_source_ids and _mix_close(
        left.terminal, right.terminal, tolerance
    )


__all__ = [
    "append_partial",
    "block_forward",
    "block_forward_oracle",
    "block_sources",
    "commit_partial",
    "full_forward",
    "full_forward_oracle",
    "new_block_state",
    "source_partition",
    "traces_close",
    "validate_block_state",
]
