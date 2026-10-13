"""Dependency-free reference semantics for Chapter 7 Attention Residuals.

The oracle keeps the sequence grid explicit while applying Attention
Residuals independently at every token position.  It implements Full and
Block schedules, a source-resolved two-point decoder-layer trace, exact
online-softmax merging, and logical resource ledgers.  The deterministic toy
transforms expose contracts; they are not trained Kimi weights or production
kernels.
"""

from __future__ import annotations

from math import cos, exp, sin, tanh
from typing import Sequence, cast

from .reference_math import dot, rms_norm, softmax
from .reference_types import (
    Dimensions,
    Grid,
    MixResult,
    RUNNING_DIMS,
    Vector,
)


def _require_width(vector: Sequence[object], width: int, name: str) -> None:
    if len(vector) != width:
        raise ValueError(f"{name} must have width {width}, got {len(vector)}")


def validate_grid(
    grid: Sequence[Sequence[float]],
    dims: Dimensions = RUNNING_DIMS,
    *,
    name: str = "token grid",
) -> None:
    """Validate one T-by-d grid."""

    if len(grid) != dims.tokens:
        raise ValueError(f"{name} must have {dims.tokens} token rows")
    if any(len(row) != dims.model for row in grid):
        raise ValueError(f"{name} rows must have width {dims.model}")


def add_grids(
    left: Grid,
    right: Grid,
    dims: Dimensions = RUNNING_DIMS,
) -> Grid:
    """Return the elementwise sum of two validated token grids."""
    validate_grid(left, dims, name="left grid")
    validate_grid(right, dims, name="right grid")
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def mix_depth(
    query: Sequence[float],
    sources: Sequence[Grid],
    dims: Dimensions = RUNNING_DIMS,
) -> MixResult:
    """Mix a source stack over depth, independently for each token."""

    _require_width(query, dims.model, "pseudo-query")
    if not sources:
        raise ValueError("at least one depth source is required")
    for index, source in enumerate(sources):
        validate_grid(source, dims, name=f"source {index}")
    token_logits: list[Vector] = []
    token_weights: list[Vector] = []
    output_rows: list[Vector] = []
    for token in range(dims.tokens):
        logits = tuple(dot(query, rms_norm(source[token])) for source in sources)
        weights = softmax(logits)
        output = tuple(
            sum(
                weight * source[token][channel]
                for weight, source in zip(weights, sources, strict=True)
            )
            for channel in range(dims.model)
        )
        token_logits.append(logits)
        token_weights.append(weights)
        output_rows.append(output)
    return MixResult(
        tuple(output_rows),
        tuple(token_logits),
        tuple(token_weights),
    )


def mix_depth_oracle(
    query: Sequence[float],
    sources: Sequence[Grid],
    dims: Dimensions = RUNNING_DIMS,
) -> MixResult:
    """Evaluate the same map with an independent source-major loop."""

    _require_width(query, dims.model, "oracle pseudo-query")
    if not sources:
        raise ValueError("oracle requires at least one source")
    for index, source in enumerate(sources):
        validate_grid(source, dims, name=f"oracle source {index}")
    all_logits: list[Vector] = []
    all_weights: list[Vector] = []
    output = [[0.0 for _ in range(dims.model)] for _ in range(dims.tokens)]
    for token in range(dims.tokens):
        logits: list[float] = []
        for source in sources:
            normalized = rms_norm(source[token])
            logits.append(
                sum(
                    query[channel] * normalized[channel]
                    for channel in range(dims.model)
                )
            )
        maximum = max(logits)
        exponentials = [exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        weights = [value / denominator for value in exponentials]
        for source_index, source in enumerate(sources):
            for channel in range(dims.model):
                output[token][channel] += weights[source_index] * source[token][channel]
        all_logits.append(tuple(logits))
        all_weights.append(tuple(weights))
    return MixResult(
        tuple(tuple(row) for row in output),
        tuple(all_logits),
        tuple(all_weights),
    )


def token_vector(
    position: int,
    dims: Dimensions = RUNNING_DIMS,
) -> Vector:
    """Create one deterministic embedding row for a one-indexed token."""

    if position < 1 or position > dims.tokens:
        raise ValueError("position is outside the declared token grid")
    values = [
        (((position * (channel + 2) + 3 * channel) % 19) - 9) / 8
        for channel in range(dims.model)
    ]
    if position <= 4:
        values[0] += 0.75
    if position == dims.tokens:
        values[1] += 0.75
    return tuple(values)


def running_embedding(
    dims: Dimensions = RUNNING_DIMS,
) -> Grid:
    """Return the four-visual/six-text running request embedding grid."""

    return tuple(token_vector(position, dims) for position in range(1, dims.tokens + 1))


def query_vector(
    target: int,
    dims: Dimensions = RUNNING_DIMS,
) -> Vector:
    """Return one deterministic layer-specific pseudo-query."""

    if target < 0:
        raise ValueError("target must be nonnegative")
    return tuple(
        0.22 * sin((target + 1) * (channel + 1))
        + 0.11 * cos((target + 2) * (channel + 1))
        for channel in range(dims.model)
    )


def running_queries(
    dims: Dimensions = RUNNING_DIMS,
) -> tuple[Vector, ...]:
    """Return one query per transformation plus one terminal query."""

    return tuple(
        query_vector(target, dims) for target in range(dims.transformations + 1)
    )


def transform_grid(
    hidden: Grid,
    target: int,
    dims: Dimensions = RUNNING_DIMS,
) -> Grid:
    """Apply a deterministic shape-preserving toy transformation."""

    validate_grid(hidden, dims, name="transformation input")
    if target < 0:
        raise ValueError("target must be nonnegative")
    rows: list[Vector] = []
    for token, row in enumerate(hidden):
        output: list[float] = []
        for channel in range(dims.model):
            first = row[(channel + target) % dims.model]
            second = row[(channel + target + 1) % dims.model]
            bias = 0.015 * (target + 1) * (channel + 1)
            token_term = 0.01 * (token + 1)
            output.append(tanh(0.58 * first - 0.21 * second + bias + token_term))
        rows.append(tuple(output))
    return tuple(rows)


def standard_residual_state(
    embedding: Grid,
    outputs: Sequence[Grid],
    dims: Dimensions = RUNNING_DIMS,
) -> Grid:
    """Add the embedding and every transformation output with unit weight."""

    validate_grid(embedding, dims, name="residual embedding")
    state = embedding
    for output in outputs:
        state = add_grids(state, output, dims)
    return state


def max_abs_difference(
    left: Sequence[float] | Grid,
    right: Sequence[float] | Grid,
) -> float:
    """Return maximum absolute difference for vectors or token grids."""

    if len(left) != len(right):
        raise ValueError("compared objects must have equal outer length")
    if not left:
        return 0.0
    left_first = left[0]
    right_first = right[0]
    if isinstance(left_first, tuple):
        if not isinstance(right_first, tuple):
            raise ValueError("compared objects must have equal rank")
        left_grid = cast(Grid, left)
        right_grid = cast(Grid, right)
        return max(
            max_abs_difference(left_row, right_row)
            for left_row, right_row in zip(left_grid, right_grid, strict=True)
        )
    if isinstance(right_first, tuple):
        raise ValueError("compared objects must have equal rank")
    left_vector = cast(Sequence[float], left)
    right_vector = cast(Sequence[float], right)
    return max(
        (abs(a - b) for a, b in zip(left_vector, right_vector, strict=True)),
        default=0.0,
    )


def _mix_close(left: MixResult, right: MixResult, tolerance: float) -> bool:
    return (
        max_abs_difference(left.output, right.output) <= tolerance
        and max(
            (
                max_abs_difference(a, b)
                for a, b in zip(left.weights, right.weights, strict=True)
            ),
            default=0.0,
        )
        <= tolerance
    )
