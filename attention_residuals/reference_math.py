"""Numerical primitives for the dependency-free Attention Residuals reference."""

from __future__ import annotations

from math import exp, sqrt
from typing import Sequence


Vector = tuple[float, ...]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two vectors."""
    if len(left) != len(right):
        raise ValueError(
            f"dot-product left operand must have width {len(right)}, got {len(left)}"
        )
    return sum(a * b for a, b in zip(left, right, strict=True))


def rms_norm(vector: Sequence[float], epsilon: float = 1e-6) -> Vector:
    """Apply unweighted RMSNorm to one nonempty vector."""

    if not vector:
        raise ValueError("RMSNorm requires a nonempty vector")
    inverse = 1.0 / sqrt(sum(value * value for value in vector) / len(vector) + epsilon)
    return tuple(value * inverse for value in vector)


def softmax(values: Sequence[float]) -> Vector:
    """Return a stable probability vector."""

    if not values:
        raise ValueError("softmax requires a nonempty source axis")
    maximum = max(values)
    numerators = tuple(exp(value - maximum) for value in values)
    denominator = sum(numerators)
    return tuple(value / denominator for value in numerators)
