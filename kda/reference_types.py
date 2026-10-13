"""Typed records shared by the dependency-free KDA reference."""

from __future__ import annotations

from dataclasses import dataclass

from .reference_linear import Matrix, Number, Vector


@dataclass(frozen=True)
class KdaInput:
    """One head's projected KDA quantities at one accepted position."""

    q: Vector
    k: Vector
    v: Vector
    alpha: Vector
    beta: Number


@dataclass(frozen=True)
class StepResult:
    """All observable intermediates of one KDA update."""

    retained: Matrix
    prediction: Vector
    error: Vector
    correction: Matrix
    state: Matrix
    output: Vector


@dataclass(frozen=True)
class Trace:
    """A position-major trace with one result per head and position."""

    positions: tuple[tuple[StepResult, ...], ...]
    final_states: tuple[Matrix, ...]

    @property
    def outputs(self) -> tuple[tuple[Vector, ...], ...]:
        return tuple(
            tuple(result.output for result in position) for position in self.positions
        )
