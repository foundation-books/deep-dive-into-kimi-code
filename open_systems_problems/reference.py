"""Original finite arithmetic; no model kernels or measured performance."""

from dataclasses import dataclass
from fractions import Fraction as F


def resource_floor(work: F, traffic: F, compute: F, bandwidth: F, path: F) -> F:
    """Seconds, with nonnegative demands and genuine positive capacities."""
    if min(work, traffic, path) < 0 or min(compute, bandwidth) <= 0:
        raise ValueError("invalid demand or capacity")
    return max(work / compute, traffic / bandwidth, path)


def pareto(costs: tuple[tuple[F, ...], ...]) -> tuple[int, ...]:
    """Indices nondominated among already feasible minimized cost vectors."""
    if costs and (not costs[0] or any(len(c) != len(costs[0]) for c in costs)):
        raise ValueError("cost vectors must have the same positive dimension")
    return tuple(
        i
        for i, candidate in enumerate(costs)
        if not any(
            all(a <= b for a, b in zip(other, candidate, strict=True))
            and any(a < b for a, b in zip(other, candidate, strict=True))
            for other in costs
        )
    )


def retention_value(size: F, duration: F, probability: F, saved: F, price: F) -> F:
    """Expected net GPU work for one possible compatible reuse."""
    if min(size, duration, saved, price) < 0 or not 0 <= probability <= 1:
        raise ValueError("invalid retention inputs")
    return probability * saved - size * duration * price


def placement(queues: tuple[F, ...], service: tuple[F, ...]) -> tuple[int, F]:
    """Minimum FCFS estimate; ties choose the earliest worker index."""
    if not queues or len(queues) != len(service) or min(queues + service) < 0:
        raise ValueError("nonempty matching nonnegative estimates required")
    totals = tuple(q + s for q, s in zip(queues, service, strict=True))
    chosen = min(range(len(totals)), key=lambda i: (totals[i], i))
    return chosen, totals[chosen]


def migration_cost(size: F, bandwidth: F, setup: F, convert: F, validate: F) -> F:
    """Serial destination-ready time; does not certify continuation validity."""
    if min(size, setup, convert, validate) < 0 or bandwidth <= 0:
        raise ValueError("invalid serial migration costs")
    return setup + size / bandwidth + convert + validate


def recurrence(inputs: tuple[F, ...], quantum: F | None = None) -> tuple[F, ...]:
    """Scalar half-decay; optional nearest-even quantization at every update."""
    if quantum is not None and quantum <= 0:
        raise ValueError("quantum must be positive")
    state = F(0)
    states = []
    for value in inputs:
        state = state / 2 + value
        if quantum is not None:
            state = round(state / quantum) * quantum
        states.append(state)
    return tuple(states)


def readout(state: F) -> str:
    """Greedy two-token readout for the scalar witness."""
    return "A" if state > F(1, 2) else "B"


def amplification(extra: F, lost: F) -> F | None:
    """Extra work / invalidated work; undefined when nothing was lost."""
    if min(extra, lost) < 0:
        raise ValueError("work cannot be negative")
    return extra / lost if lost else None


def cohort_metrics(
    offered: int, admitted: int, timely: int, horizon: F
) -> tuple[F, F | None, F | None]:
    """Unique requests; timely means correct by deadline and observation cutoff."""
    if not 0 <= timely <= admitted <= offered or horizon <= 0:
        raise ValueError("invalid cohort")
    return (
        F(timely) / horizon,
        F(timely, offered) if offered else None,
        F(timely, admitted) if admitted else None,
    )


@dataclass(frozen=True)
class Observation:
    """Only information available when selecting a worker."""

    queues: tuple[F, ...]
    service_estimates: tuple[F, ...]


@dataclass(frozen=True)
class TraceCase:
    """Observed history and future tape kept as distinct fields."""

    observed: Observation
    future_tool_return: F
    future_output_length: int


def online_choice(history: Observation) -> int:
    """Receives the observation projection, never a trace's future fields."""
    return placement(history.queues, history.service_estimates)[0]
