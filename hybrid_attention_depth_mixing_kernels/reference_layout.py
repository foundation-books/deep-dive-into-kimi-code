"""Exact vector, ownership, and layout primitives for Chapter 34."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction


Number = int | Fraction
Vector = tuple[Fraction, ...]


def f(numerator: int, denominator: int = 1) -> Fraction:
    """Construct an exact rational number."""

    return Fraction(numerator, denominator)


def vector_add(left: Vector, right: Vector) -> Vector:
    """Add same-width vectors."""

    if len(left) != len(right):
        raise ValueError("vector widths must match")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def vector_scale(scale: Fraction, value: Vector) -> Vector:
    """Multiply a vector by an exact scalar."""

    return tuple(scale * item for item in value)


def vector_sum(values: Iterable[Vector], width: int) -> Vector:
    """Sum vectors while making the empty width explicit."""

    result = tuple(f(0) for _ in range(width))
    for value in values:
        result = vector_add(result, value)
    return result


@dataclass(frozen=True)
class OwnerTag:
    """Semantic identity attached to a state-bearing object."""

    request: str
    endpoint_or_forward: str
    layer: int
    mixing_point: int | None
    revision: str
    layout: str
    dtype: str
    lifetime: str


def validate_owner(expected: OwnerTag, actual: OwnerTag) -> None:
    """Reject a same-shaped object whose semantic identity differs."""

    if expected != actual:
        differing = tuple(
            name
            for name in OwnerTag.__dataclass_fields__
            if getattr(expected, name) != getattr(actual, name)
        )
        raise ValueError("owner-tag mismatch: " + ", ".join(differing))


def logical_to_feature_major(records: Sequence[Vector]) -> tuple[Vector, ...]:
    """Map [token, feature] logical records to [feature, token] storage."""

    if not records or not records[0]:
        raise ValueError("records must have positive token and feature extents")
    width = len(records[0])
    if any(len(record) != width for record in records):
        raise ValueError("all logical records must share a width")
    return tuple(
        tuple(records[token][feature] for token in range(len(records)))
        for feature in range(width)
    )


def feature_major_to_logical(storage: Sequence[Vector]) -> tuple[Vector, ...]:
    """Invert [feature, token] storage to logical [token, feature] records."""

    if not storage or not storage[0]:
        raise ValueError("storage must have positive feature and token extents")
    tokens = len(storage[0])
    if any(len(feature) != tokens for feature in storage):
        raise ValueError("all physical feature rows must share a token extent")
    return tuple(
        tuple(storage[feature][token] for feature in range(len(storage)))
        for token in range(tokens)
    )


__all__ = [
    "Number",
    "OwnerTag",
    "Vector",
    "f",
    "feature_major_to_logical",
    "logical_to_feature_major",
    "validate_owner",
    "vector_add",
    "vector_scale",
    "vector_sum",
]
