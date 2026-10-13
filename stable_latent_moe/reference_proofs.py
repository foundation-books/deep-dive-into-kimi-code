"""Small executable witnesses for Chapter 8's algebraic claims."""

from __future__ import annotations

from itertools import combinations
from math import exp, tanh
from typing import Sequence


Vector = tuple[float, ...]
ScoreMatrix = tuple[Vector, ...]


def sigmoid(value: float) -> float:
    """Evaluate the logistic sigmoid without overflow."""

    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    scaled = exp(value)
    return scaled / (1.0 + scaled)


def softcap(value: float, beta: float) -> float:
    """Return beta*tanh(value/beta) for positive beta."""

    if beta <= 0.0:
        raise ValueError("softcap scale must be positive")
    return beta * tanh(value / beta)


def swiglu_coordinate(gate: float, up: float) -> float:
    """One scalar SwiGLU product before its output projection."""

    return gate * sigmoid(gate) * up


def situ_coordinate(
    gate: float,
    up: float,
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> float:
    """One scalar SiTU-GLU product before its output projection."""

    return softcap(gate, beta_gate) * sigmoid(gate) * softcap(up, beta_up)


def situ_coordinate_bound(beta_gate: float = 4.0, beta_up: float = 25.0) -> float:
    """Return the exact-real coordinate upper bound used by the chapter."""

    if min(beta_gate, beta_up) <= 0.0:
        raise ValueError("SiTU cap scales must be positive")
    return beta_gate * beta_up


def max_situ_grid_ratio(
    values: Sequence[float],
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> float:
    """Evaluate a Cartesian grid and return max |SiTU|/(beta1*beta2)."""

    if not values:
        raise ValueError("SiTU grid must be nonempty")
    bound = situ_coordinate_bound(beta_gate, beta_up)
    return max(
        abs(situ_coordinate(gate, up, beta_gate, beta_up)) / bound
        for gate in values
        for up in values
    )


def balanced_assignment_dp(
    scores: ScoreMatrix,
    selected: int,
) -> tuple[float, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Find the maximum-score exactly balanced assignment by dynamic programming.

    Expert identifiers in the returned route sets are one based.  The dynamic
    program is an exhaustive state merge over legal per-token subsets, not a
    production routing algorithm.
    """

    if not scores or not scores[0]:
        raise ValueError("score matrix must be nonempty")
    experts = len(scores[0])
    if any(len(row) != experts for row in scores):
        raise ValueError("score matrix must be rectangular")
    if not 1 <= selected < experts:
        raise ValueError("selected count must lie between one and E-1")
    routes = len(scores) * selected
    if routes % experts:
        raise ValueError("exact target load must be integral")
    target = routes // experts
    subsets = tuple(combinations(range(experts), selected))
    states: dict[tuple[int, ...], tuple[float, tuple[tuple[int, ...], ...]]] = {
        (0,) * experts: (0.0, ())
    }
    for row in scores:
        next_states: dict[
            tuple[int, ...], tuple[float, tuple[tuple[int, ...], ...]]
        ] = {}
        for loads, (objective, assignment) in states.items():
            for subset in subsets:
                candidate_loads = list(loads)
                for expert in subset:
                    candidate_loads[expert] += 1
                if any(load > target for load in candidate_loads):
                    continue
                key = tuple(candidate_loads)
                value = objective + sum(row[expert] for expert in subset)
                incumbent = next_states.get(key)
                one_based = tuple(expert + 1 for expert in subset)
                candidate = (value, assignment + (one_based,))
                if incumbent is None or value > incumbent[0]:
                    next_states[key] = candidate
        states = next_states
    final_loads = (target,) * experts
    if final_loads not in states:
        raise ValueError("no balanced assignment exists")
    objective, assignment = states[final_loads]
    return objective, assignment, final_loads


def local_quantile_average_counterexample() -> dict[str, object]:
    """Show that averaging rank-local medians need not give the pooled median."""

    first = (0.0, 100.0, 101.0)
    second = (1.0, 2.0, 3.0)
    local_average = (first[1] + second[1]) / 2.0
    pooled = tuple(sorted((*first, *second)))
    pooled_median = (pooled[2] + pooled[3]) / 2.0
    return {
        "shards": (first, second),
        "average_local_medians": local_average,
        "pooled_median": pooled_median,
        "different": local_average != pooled_median,
    }


__all__ = [
    "balanced_assignment_dp",
    "local_quantile_average_counterexample",
    "max_situ_grid_ratio",
    "sigmoid",
    "situ_coordinate",
    "situ_coordinate_bound",
    "softcap",
    "swiglu_coordinate",
]
