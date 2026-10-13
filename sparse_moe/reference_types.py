"""Immutable route records for the dependency-free sparse-MoE reference."""

from __future__ import annotations

from dataclasses import dataclass


Vector = tuple[float, ...]


@dataclass(frozen=True)
class RouterResult:
    """One routing decision with one-based expert identifiers."""

    scores: Vector
    biased_scores: Vector
    selected: tuple[int, ...]
    weights: Vector


@dataclass(frozen=True, order=True)
class RouteIdentity:
    """Immutable identity that survives dispatch and combine."""

    origin: str
    expert: int
    ordinal: int


@dataclass(frozen=True)
class RouteRecord:
    """A dispatched or returned route record."""

    identity: RouteIdentity
    weight: float
    payload: Vector
