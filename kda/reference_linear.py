"""Exact numeric primitives shared by the Chapter 4 KDA oracle."""

from __future__ import annotations

from fractions import Fraction
from math import isclose


Number = Fraction | float
Vector = tuple[Number, ...]
Matrix = tuple[tuple[Number, ...], ...]


def f(numerator: int, denominator: int = 1) -> Fraction:
    """Return one exact rational used by the frozen examples."""

    return Fraction(numerator, denominator)


ZERO = f(0)
E1: Vector = (f(1), f(0))
E2: Vector = (f(0), f(1))
Q34: Vector = (f(3, 5), f(4, 5))
Q4M3: Vector = (f(4, 5), f(-3, 5))


def _zero_matrix(rows: int, columns: int, zero: Number = ZERO) -> Matrix:
    return tuple(tuple(zero for _ in range(columns)) for _ in range(rows))


def _identity(size: int) -> Matrix:
    return tuple(
        tuple(f(1) if row == column else f(0) for column in range(size))
        for row in range(size)
    )


def _matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(a + b for a, b in zip(left_row, right_row, strict=True))
        for left_row, right_row in zip(left, right, strict=True)
    )


def _matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    columns = tuple(zip(*right, strict=True))
    return tuple(
        tuple(
            sum((a * b for a, b in zip(row, column, strict=True)), f(0))
            for column in columns
        )
        for row in left
    )


def _matrix_vector_transpose_read(state: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum((state[row][column] * vector[row] for row in range(len(state))), f(0))
        for column in range(len(state[0]))
    )


def _outer(left: Vector, right: Vector) -> Matrix:
    return tuple(tuple(a * b for b in right) for a in left)


def _scale_matrix(value: Number, matrix: Matrix) -> Matrix:
    return tuple(tuple(value * item for item in row) for row in matrix)


def _norm_is_one(vector: Vector) -> bool:
    value = sum((item * item for item in vector), f(0))
    if isinstance(value, Fraction):
        return value == 1
    return isclose(float(value), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def matrix_as_floats(matrix: Matrix, digits: int = 6) -> list[list[float]]:
    """Round an exact matrix for JSON-friendly reporting."""

    return [[round(float(value), digits) for value in row] for row in matrix]


def vector_as_floats(vector: Vector, digits: int = 6) -> list[float]:
    """Round an exact vector for JSON-friendly reporting."""

    return [round(float(value), digits) for value in vector]


__all__ = [
    "E1",
    "E2",
    "Matrix",
    "Number",
    "Q34",
    "Q4M3",
    "Vector",
    "ZERO",
    "_identity",
    "_matrix_add",
    "_matrix_multiply",
    "_matrix_vector_transpose_read",
    "_norm_is_one",
    "_outer",
    "_scale_matrix",
    "_zero_matrix",
    "f",
    "matrix_as_floats",
    "vector_as_floats",
]
