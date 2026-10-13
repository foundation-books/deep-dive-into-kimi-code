"""Floating-point SiTU fixtures for the routed-computation oracle."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, tanh


FloatVector = tuple[float, ...]


def dot(row: Sequence[float], column: Sequence[float]) -> float:
    """Return the strict pairwise dot product of two float vectors."""

    return sum(left * right for left, right in zip(row, column, strict=True))


def situ_expert(
    rows: Sequence[FloatVector],
    gate: FloatVector,
    up: FloatVector,
    down: FloatVector,
    beta_gate: float = 1.0,
    beta_up: float = 1.0,
) -> tuple[FloatVector, ...]:
    """Evaluate the Chapter 35 one-hidden-unit SiTU fixture."""

    if not rows or beta_gate <= 0 or beta_up <= 0:
        raise ValueError("rows and positive cap scales are required")
    outputs: list[FloatVector] = []
    for row in rows:
        g = dot(row, gate)
        u = dot(row, up)
        sigmoid = 1.0 / (1.0 + exp(-g))
        hidden = beta_gate * tanh(g / beta_gate) * sigmoid
        hidden *= beta_up * tanh(u / beta_up)
        outputs.append(tuple(hidden * component for component in down))
    return tuple(outputs)


def grouped_situ_fixture() -> tuple[tuple[FloatVector, ...], tuple[FloatVector, ...]]:
    """Evaluate two ragged experts independently and in descriptor order."""

    segment1 = ((1.0, 0.0), (0.0, 1.0))
    segment2 = ((1.0, 1.0),)
    independent = (
        *situ_expert(segment1, (1.0, 2.0), (2.0, 1.0), (1.0, -1.0)),
        *situ_expert(segment2, (1.0, -1.0), (1.0, 1.0), (2.0, 1.0)),
    )
    descriptors = (
        (segment1, (1.0, 2.0), (2.0, 1.0), (1.0, -1.0)),
        (segment2, (1.0, -1.0), (1.0, 1.0), (2.0, 1.0)),
    )
    grouped = tuple(
        output
        for rows, gate, up, down in descriptors
        for output in situ_expert(rows, gate, up, down)
    )
    return tuple(independent), grouped


__all__ = ["FloatVector", "dot", "grouped_situ_fixture", "situ_expert"]
