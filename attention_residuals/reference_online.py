"""Online-softmax merging for Block Attention Residuals."""

from __future__ import annotations

from math import exp
from typing import Sequence

from .reference_core import validate_grid
from .reference_math import dot, rms_norm
from .reference_runtime import validate_block_state
from .reference_types import BlockState, Dimensions, Grid, OnlineState, RUNNING_DIMS


def online_summary(
    query: Sequence[float],
    sources: Sequence[Grid],
    dims: Dimensions = RUNNING_DIMS,
) -> tuple[OnlineState, ...]:
    """Summarize a nonempty source group for every token."""

    if len(query) != dims.model:
        raise ValueError("online-softmax query has the wrong width")
    if not sources:
        raise ValueError("online-softmax group must be nonempty")
    for index, source in enumerate(sources):
        validate_grid(source, dims, name=f"online source {index}")
    states: list[OnlineState] = []
    for token in range(dims.tokens):
        logits = tuple(dot(query, rms_norm(source[token])) for source in sources)
        maximum = max(logits)
        shifted = tuple(exp(logit - maximum) for logit in logits)
        normalizer = sum(shifted)
        numerator = tuple(
            sum(
                weight * source[token][channel]
                for weight, source in zip(shifted, sources, strict=True)
            )
            for channel in range(dims.model)
        )
        states.append(OnlineState(maximum, normalizer, numerator))
    return tuple(states)


def merge_online_state(left: OnlineState, right: OnlineState) -> OnlineState:
    """Merge two disjoint online-softmax groups in exact-real semantics."""

    if (
        left.normalizer <= 0
        or right.normalizer <= 0
        or len(left.numerator) != len(right.numerator)
    ):
        raise ValueError("online-softmax states are incompatible")
    maximum = max(left.maximum, right.maximum)
    left_scale = exp(left.maximum - maximum)
    right_scale = exp(right.maximum - maximum)
    normalizer = left_scale * left.normalizer + right_scale * right.normalizer
    numerator = tuple(
        left_scale * left_value + right_scale * right_value
        for left_value, right_value in zip(
            left.numerator,
            right.numerator,
            strict=True,
        )
    )
    return OnlineState(maximum, normalizer, numerator)


def two_phase_block_mix(
    query: Sequence[float],
    state: BlockState,
    dims: Dimensions = RUNNING_DIMS,
) -> Grid:
    """Merge batched historical blocks with the sequential current partial."""

    validate_block_state(state, dims)
    historical = (state.embedding,) + state.completed
    inter = online_summary(query, historical, dims)
    merged = inter
    if state.partial is not None:
        intra = online_summary(query, (state.partial,), dims)
        merged = tuple(
            merge_online_state(left, right)
            for left, right in zip(inter, intra, strict=True)
        )
    return tuple(
        tuple(value / item.normalizer for value in item.numerator) for item in merged
    )
