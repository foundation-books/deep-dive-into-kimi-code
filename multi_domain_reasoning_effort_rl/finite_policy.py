"""Exact four-route teaching policy for Chapter 21.

The routes and costs are declared assumptions, not observed Kimi K3 behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

from .reference_core import (
    BaselineRecord,
    BudgetState,
    MultiplierRecord,
    ResourceVector,
    budget_state,
    effective_reward,
)


@dataclass(frozen=True)
class Route:
    """A fixed-cost macro-route with conditional binary task success."""

    name: str
    thinking_cost: int
    success_probability: float

    def validate(self) -> None:
        if not self.name or not isinstance(self.thinking_cost, int):
            raise ValueError("route needs a name and integer thinking cost")
        if isinstance(self.thinking_cost, bool) or self.thinking_cost <= 0:
            raise ValueError("route thinking cost must be positive")
        if not math.isfinite(self.success_probability) or not (
            0 <= self.success_probability <= 1
        ):
            raise ValueError("route success probability must be in [0, 1]")


ROUTES = (
    Route("A", 40, 0.35),
    Route("B", 400, 0.95),
    Route("C", 120, 0.85),
    Route("D", 70, 0.75),
)
INITIAL_PROBABILITIES = (0.40, 0.40, 0.15, 0.05)
BASELINE = Fraction(200)
MULTIPLIERS = {"max": Fraction(5, 2), "high": Fraction(4, 5), "low": Fraction(2, 5)}


def validate_fixture(routes: Sequence[Route], probabilities: Sequence[float]) -> None:
    """Reject malformed route and initial-probability declarations."""

    if not routes or len(routes) != len(probabilities):
        raise ValueError("routes and probabilities need the same nonzero length")
    if len({route.name for route in routes}) != len(routes):
        raise ValueError("route names must be unique")
    for route in routes:
        route.validate()
    if any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("initial probabilities must be finite and nonnegative")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("initial probabilities must sum to one")


def cap(baseline: Fraction, multiplier: Fraction) -> Fraction:
    """Compute the positive finite per-problem cap exactly."""

    if isinstance(baseline, bool) or isinstance(multiplier, bool):
        raise ValueError("budget factors must be numeric")
    for value in (baseline, multiplier):
        if not isinstance(value, (int, float, Fraction)):
            raise ValueError("budget factors must be numeric")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError("budget factors must be finite and positive")
    return Fraction(baseline) * Fraction(multiplier)


def initial_logits(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Turn a strictly positive route distribution into softmax logits."""

    if any(p <= 0 for p in probabilities):
        raise ValueError("logit initialization requires positive probabilities")
    validate_fixture(
        tuple(Route(str(i), 1, 0.5) for i in range(len(probabilities))), probabilities
    )
    return tuple(math.log(p) for p in probabilities)


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite logits after subtracting their maximum."""

    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("logits must be a nonempty finite vector")
    shifted = tuple(math.exp(value - max(logits)) for value in logits)
    total = sum(shifted)
    return tuple(value / total for value in shifted)


def route_states(routes: Sequence[Route], budget: Fraction) -> tuple[BudgetState, ...]:
    """Use the existing Chapter 21 budget projection for each declared route."""
    baseline = BaselineRecord(
        "four-route-general", "v1", "fixture-p0", "declared", budget
    )
    multiplier = MultiplierRecord("g", "H", "fixture", Fraction(1), "declared")
    return tuple(
        budget_state(
            "four-route-general",
            "v1",
            "g",
            "H",
            ResourceVector(route.thinking_cost, 0, 0, 1, 0),
            baseline,
            multiplier,
        )
        for route in routes
    )


def expected_rewards(routes: Sequence[Route], budget: Fraction) -> tuple[float, ...]:
    """Apply strict overrun replacement to each route's expected reward."""

    return tuple(
        float(effective_reward(Fraction(str(route.success_probability)), state))
        for route, state in zip(routes, route_states(routes, budget), strict=True)
    )


def metrics(
    routes: Sequence[Route], logits: Sequence[float], budget: Fraction
) -> dict[str, float | list[float]]:
    """Evaluate reward, task success, compliance, and full-route cost."""

    p = softmax(logits)
    q = expected_rewards(routes, budget)
    within = tuple(route.thinking_cost <= budget for route in routes)
    return {
        "probabilities": list(p),
        "effective_reward": sum(pi * qi for pi, qi in zip(p, q, strict=True)),
        "raw_success": sum(
            pi * route.success_probability for pi, route in zip(p, routes, strict=True)
        ),
        "success_within_budget": sum(
            pi * route.success_probability
            for pi, route, ok in zip(p, routes, within, strict=True)
            if ok
        ),
        "compliance": sum(pi for pi, ok in zip(p, within, strict=True) if ok),
        "expected_cost": sum(
            pi * route.thinking_cost for pi, route in zip(p, routes, strict=True)
        ),
    }


def exact_gradient(
    routes: Sequence[Route], logits: Sequence[float], budget: Fraction
) -> tuple[float, ...]:
    """Return the exact objective gradient in route-logit coordinates."""

    p = softmax(logits)
    q = expected_rewards(routes, budget)
    objective = sum(pi * qi for pi, qi in zip(p, q, strict=True))
    return tuple(pi * (qi - objective) for pi, qi in zip(p, q, strict=True))


def exact_step(
    routes: Sequence[Route], logits: Sequence[float], budget: Fraction, rate: float
) -> tuple[float, ...]:
    """Apply one plain gradient-ascent step to the route logits."""

    if not math.isfinite(rate) or rate < 0:
        raise ValueError("learning rate must be finite and nonnegative")
    return tuple(
        value + rate * gradient
        for value, gradient in zip(
            logits, exact_gradient(routes, logits, budget), strict=True
        )
    )


def cutoff_metrics(
    routes: Sequence[Route], logits: Sequence[float], cutoff: int
) -> dict[str, float]:
    """A route reveals its answer only when all required thinking work finishes."""
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
        raise ValueError("cutoff must be a positive integer")
    p = softmax(logits)
    complete = tuple(route.thinking_cost <= cutoff for route in routes)
    return {
        "completion": sum(pi for pi, ok in zip(p, complete, strict=True) if ok),
        "raw_success": sum(
            pi * route.success_probability
            for pi, route, ok in zip(p, routes, complete, strict=True)
            if ok
        ),
        "consumed_cost": sum(
            pi * min(route.thinking_cost, cutoff)
            for pi, route in zip(p, routes, strict=True)
        ),
    }
