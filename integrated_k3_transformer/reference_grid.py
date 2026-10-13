"""Grid arithmetic and depth-state construction for the integrated K3 oracle."""

from __future__ import annotations

from fractions import Fraction

from .reference_types import DepthState, Grid, _validate_grid


def add(left: Grid, right: Grid) -> Grid:
    """Add two equally shaped token grids."""

    _validate_grid(left, "left grid")
    _validate_grid(right, "right grid")
    if len(left) != len(right):
        raise ValueError("grid token counts disagree")
    return tuple(
        tuple(a + b for a, b in zip(lrow, rrow, strict=True))
        for lrow, rrow in zip(left, right, strict=True)
    )


def scale(grid: Grid, weight: Fraction) -> Grid:
    """Scale every coordinate in a token grid."""

    return tuple(tuple(weight * value for value in row) for row in grid)


def new_depth_state(forward_id: str, embedding: Grid) -> DepthState:
    """Start a fresh one-forward depth history."""

    if not forward_id:
        raise ValueError("forward identity must be nonempty")
    _validate_grid(embedding, "depth embedding")
    return DepthState(forward_id=forward_id, embedding=embedding)


__all__ = ["add", "new_depth_state", "scale"]
