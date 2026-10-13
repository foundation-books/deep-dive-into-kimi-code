"""Matrix and finite polar-transform primitives for Chapter 12."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .reference_types import FlatTensor, Matrix, NS5_COEFFICIENTS


def as_matrix(rows: Iterable[Iterable[float]]) -> Matrix:
    """Freeze and validate a nonempty rectangular matrix."""

    matrix = tuple(tuple(float(value) for value in row) for row in rows)
    if not matrix or not matrix[0]:
        raise ValueError("a matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have equal width")
    return matrix


def shape(matrix: Matrix) -> tuple[int, int]:
    """Return the matrix row and column counts."""
    return len(matrix), len(matrix[0])


def zeros(rows: int, columns: int) -> Matrix:
    """Return a zero matrix with the requested shape."""
    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def constant_matrix(rows: int, columns: int, value: float) -> Matrix:
    """Return a matrix filled with one constant value."""
    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    return tuple(tuple(float(value) for _ in range(columns)) for _ in range(rows))


def transpose(matrix: Matrix) -> Matrix:
    """Return the matrix transpose."""
    rows, columns = shape(matrix)
    return tuple(tuple(matrix[i][j] for i in range(rows)) for j in range(columns))


def add(left: Matrix, right: Matrix) -> Matrix:
    """Add two same-shaped matrices."""
    if shape(left) != shape(right):
        raise ValueError("matrix shapes must agree")
    return tuple(
        tuple(x + y for x, y in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def subtract(left: Matrix, right: Matrix) -> Matrix:
    """Subtract one matrix from another."""
    return add(left, scale(right, -1.0))


def scale(matrix: Matrix, scalar: float) -> Matrix:
    """Scale every value by the supplied factor."""
    return tuple(tuple(scalar * value for value in row) for row in matrix)


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two compatible matrices."""
    left_rows, shared = shape(left)
    right_rows, _right_columns = shape(right)
    if shared != right_rows:
        raise ValueError("inner dimensions must agree")
    right_t = transpose(right)
    return tuple(
        tuple(sum(left[i][k] * column[k] for k in range(shared)) for column in right_t)
        for i in range(left_rows)
    )


def frobenius_norm(matrix: Matrix) -> float:
    """Return the matrix Frobenius norm."""
    return math.sqrt(sum(value * value for row in matrix for value in row))


def rms(matrix: Matrix) -> float:
    """Return the matrix root-mean-square magnitude."""
    rows, columns = shape(matrix)
    return frobenius_norm(matrix) / math.sqrt(rows * columns)


def flatten(matrix: Matrix) -> FlatTensor:
    """Flatten a matrix in row-major order."""
    return tuple(value for row in matrix for value in row)


def flat_norm(values: Sequence[float]) -> float:
    """Return the Euclidean norm of flattened values."""
    return math.sqrt(sum(float(value) ** 2 for value in values))


def flat_rms(values: Sequence[float]) -> float:
    """Return the root-mean-square of flattened values."""
    if not values:
        raise ValueError("a tensor must be nonempty")
    return flat_norm(values) / math.sqrt(len(values))


def max_abs_difference(left: Matrix, right: Matrix) -> float:
    """Return the largest elementwise absolute difference."""
    if shape(left) != shape(right):
        raise ValueError("matrix shapes must agree")
    return max(
        abs(x - y)
        for left_row, right_row in zip(left, right, strict=True)
        for x, y in zip(left_row, right_row, strict=True)
    )


def diagonal_matrix(rows: int, columns: int, entries: Sequence[float]) -> Matrix:
    """Return a rectangular diagonal fixture with the supplied entries."""

    if len(entries) > min(rows, columns):
        raise ValueError("too many diagonal entries")
    return tuple(
        tuple(
            float(entries[i]) if i == j and i < len(entries) else 0.0
            for j in range(columns)
        )
        for i in range(rows)
    )


def diagonal_polar(rows: int, columns: int, entries: Sequence[float]) -> Matrix:
    """Return the exact partial polar factor of a rectangular diagonal matrix."""

    signs = tuple(0.0 if value == 0 else math.copysign(1.0, value) for value in entries)
    return diagonal_matrix(rows, columns, signs)


def diagonal_rank(entries: Sequence[float]) -> int:
    """Return the count of nonzero diagonal entries."""
    return sum(value != 0.0 for value in entries)


def polar_rms(rows: int, columns: int, rank: int) -> float:
    """RMS of a rank-r partial polar factor: sqrt(r/(rows*columns))."""

    if not 0 <= rank <= min(rows, columns):
        raise ValueError("rank is outside the matrix dimensions")
    return math.sqrt(rank / (rows * columns))


def shape_scale(rows: int, columns: int, base: float = 0.2) -> float:
    """Moonlight's shape multiplier c0*sqrt(max(rows, columns))."""

    if rows <= 0 or columns <= 0 or base < 0:
        raise ValueError("invalid shape scale")
    return base * math.sqrt(max(rows, columns))


def corrected_polar_rms(rows: int, columns: int, rank: int, base: float = 0.2) -> float:
    """Compute the finite-rank polar-update RMS estimate."""
    return shape_scale(rows, columns, base) * polar_rms(rows, columns, rank)


def newton_schulz_zeroth_power(
    matrix: Matrix,
    steps: int = 5,
    coefficients: tuple[float, float, float] = NS5_COEFFICIENTS,
    epsilon: float = 1e-7,
) -> Matrix:
    """Finite Newton--Schulz polynomial used by the official toy code.

    Tall matrices are transposed for the smaller Gram matrix, normalized by
    their Frobenius norm, iterated, and transposed back.  A finite iterate is
    an approximation and is not silently identified with the exact polar
    factor.
    """

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    original_rows, original_columns = shape(matrix)
    was_transposed = original_rows > original_columns
    work = transpose(matrix) if was_transposed else matrix
    norm = frobenius_norm(work)
    if norm == 0.0:
        return zeros(original_rows, original_columns)
    work = scale(work, 1.0 / (norm + epsilon))
    a, b, c = coefficients
    for _ in range(steps):
        gram = matmul(work, transpose(work))
        polynomial = add(scale(gram, b), scale(matmul(gram, gram), c))
        work = add(scale(work, a), matmul(polynomial, work))
    return transpose(work) if was_transposed else work


__all__ = """
add as_matrix constant_matrix corrected_polar_rms diagonal_matrix diagonal_polar
diagonal_rank flat_norm flat_rms flatten frobenius_norm matmul max_abs_difference
newton_schulz_zeroth_power polar_rms rms scale shape shape_scale subtract transpose zeros
""".split()
