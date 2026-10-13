"""Small matrix oracles for the Chapter 14 Per-Head Muon fixture."""

from __future__ import annotations

import math
from typing import Sequence


Matrix = tuple[tuple[float, ...], ...]
NS5_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
DEFAULT_C0 = 0.2


def _shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("matrix must be nonempty and rectangular")
    return len(matrix), len(matrix[0])


def _transpose(matrix: Matrix) -> Matrix:
    rows, columns = _shape(matrix)
    return tuple(tuple(matrix[i][j] for i in range(rows)) for j in range(columns))


def _scale(matrix: Matrix, scalar: float) -> Matrix:
    return tuple(tuple(scalar * value for value in row) for row in matrix)


def _add(left: Matrix, right: Matrix) -> Matrix:
    if _shape(left) != _shape(right):
        raise ValueError("matrix shapes must agree")
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, shared = _shape(left)
    right_rows, _ = _shape(right)
    if shared != right_rows:
        raise ValueError("matrix inner dimensions must agree")
    columns = _transpose(right)
    return tuple(
        tuple(sum(left[i][k] * column[k] for k in range(shared)) for column in columns)
        for i in range(left_rows)
    )


def _frobenius(matrix: Matrix) -> float:
    return math.sqrt(sum(value * value for row in matrix for value in row))


def max_abs_difference(left: Matrix, right: Matrix) -> float:
    """Return the maximum entrywise absolute difference."""

    if _shape(left) != _shape(right):
        raise ValueError("matrix shapes must agree")
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def finite_ns_transform(matrix: Matrix, steps: int = 5) -> Matrix:
    """Apply the unscaled finite NS5 map used by the matrix oracle."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    original_rows, original_columns = _shape(matrix)
    transposed = original_rows > original_columns
    work = _transpose(matrix) if transposed else matrix
    norm = _frobenius(work)
    if norm == 0:
        return tuple(
            tuple(0.0 for _ in range(original_columns)) for _ in range(original_rows)
        )
    work = _scale(work, 1.0 / (norm + 1e-7))
    a, b, c = NS5_COEFFICIENTS
    for _ in range(steps):
        gram = _matmul(work, _transpose(work))
        polynomial = _add(_scale(gram, b), _scale(_matmul(gram, gram), c))
        work = _add(_scale(work, a), _matmul(polynomial, work))
    return _transpose(work) if transposed else work


def shape_scale(matrix: Matrix, c0: float = DEFAULT_C0) -> float:
    """Return Chapter 12's c0*sqrt(max(m,n)) multiplier."""

    if c0 <= 0.0:
        raise ValueError("c0 must be positive")
    rows, columns = _shape(matrix)
    return c0 * math.sqrt(max(rows, columns))


def _muon_direction(matrix: Matrix, c0: float) -> Matrix:
    """Apply the finite map and its declared shape multiplier."""

    return _scale(finite_ns_transform(matrix), shape_scale(matrix, c0))


def per_head_muon(
    blocks: Sequence[Matrix], c0: float = DEFAULT_C0
) -> tuple[Matrix, ...]:
    """Apply shape-scaled Muon independently to every logical head block."""

    if not blocks:
        raise ValueError("at least one head block is required")
    shape = _shape(blocks[0])
    if any(_shape(block) != shape for block in blocks):
        raise ValueError("the fixture requires equal head-block shapes")
    return tuple(_muon_direction(block, c0) for block in blocks)


def global_stacked_muon(
    blocks: Sequence[Matrix], c0: float = DEFAULT_C0
) -> tuple[Matrix, ...]:
    """Return the intentionally different shape-scaled stacked transform."""

    if not blocks:
        raise ValueError("at least one head block is required")
    rows, columns = _shape(blocks[0])
    if any(_shape(block) != (rows, columns) for block in blocks):
        raise ValueError("the fixture requires equal head-block shapes")
    stacked = tuple(row for block in blocks for row in block)
    transformed = _muon_direction(stacked, c0)
    return tuple(
        tuple(transformed[index * rows : (index + 1) * rows])
        for index in range(len(blocks))
    )


def per_head_fixture() -> dict[str, object]:
    """Return the two-head block/locality counterexample."""

    heads = (
        ((4.0, 0.0), (0.0, 1.0)),
        ((40.0, 0.0), (0.0, 10.0)),
    )
    perturbed = (heads[0], ((80.0, 0.0), (0.0, 5.0)))
    blockwise = per_head_muon(heads)
    global_blocks = global_stacked_muon(heads)
    isolated = per_head_muon(perturbed)
    globally_perturbed = global_stacked_muon(perturbed)
    rank_deficient = per_head_muon((((3.0, 0.0), (0.0, 0.0)),))[0]
    return {
        "heads": heads,
        "blockwise": blockwise,
        "global": global_blocks,
        "head_0_blockwise_perturbation": max_abs_difference(blockwise[0], isolated[0]),
        "head_0_global_perturbation": max_abs_difference(
            global_blocks[0], globally_perturbed[0]
        ),
        "blockwise_global_difference": max_abs_difference(
            blockwise[0], global_blocks[0]
        ),
        "rank_deficient_shape": _shape(rank_deficient),
        "rank_deficient_finite": all(
            math.isfinite(value) for row in rank_deficient for value in row
        ),
    }


__all__ = (
    "Matrix",
    "DEFAULT_C0",
    "finite_ns_transform",
    "global_stacked_muon",
    "max_abs_difference",
    "per_head_fixture",
    "per_head_muon",
    "shape_scale",
)
