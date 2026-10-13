"""Exact finite-group reward-error example, not a model-learning result."""

from __future__ import annotations

import math
from typing import Sequence, TypedDict


class UpdateResult(TypedDict):
    """One mean-reduced softmax step and its numerical derivative check."""

    rewards: tuple[float, ...]
    coefficients: tuple[float, ...]
    gradient: tuple[float, ...]
    finite_difference: tuple[float, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


class ComparisonResult(TypedDict):
    """Correct and proxy-error steps from the same zero-logit policy."""

    evidence_class: str
    truth: UpdateResult
    false_acceptance: UpdateResult


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite logits after a common stabilizing shift."""
    maximum = max(logits)
    weights = tuple(math.exp(x - maximum) for x in logits)
    total = sum(weights)
    return tuple(x / total for x in weights)


def loss(logits: Sequence[float], coefficients: Sequence[float]) -> float:
    """Compute the declared mean loss with fixed detached coefficients."""
    probabilities = softmax(logits)
    return -sum(
        c * math.log(p) for c, p in zip(coefficients, probabilities, strict=True)
    ) / len(coefficients)


def update(rewards: Sequence[float], learning_rate: float = 0.3) -> UpdateResult:
    """Center a nonempty group and take its independently checked step."""
    if not rewards:
        raise ValueError("empty reward group")
    mean = sum(rewards) / len(rewards)
    coefficients = tuple(r - mean for r in rewards)
    # At zero logits and sum(c)=0 the mean-reduced derivative is -c/B.
    gradient = tuple(-c / len(rewards) for c in coefficients)
    logits = tuple(-learning_rate * g for g in gradient)
    epsilon = 1e-5
    finite_difference = []
    for k in range(len(rewards)):
        plus, minus = [0.0] * len(rewards), [0.0] * len(rewards)
        plus[k], minus[k] = epsilon, -epsilon
        finite_difference.append(
            (loss(plus, coefficients) - loss(minus, coefficients)) / (2 * epsilon)
        )
    return {
        "rewards": tuple(rewards),
        "coefficients": coefficients,
        "gradient": gradient,
        "finite_difference": tuple(finite_difference),
        "logits": logits,
        "probabilities": softmax(logits),
    }


def reward_error_comparison() -> ComparisonResult:
    """Compare true rewards with false acceptance of response two."""
    return {
        "evidence_class": "constructed independently checked arithmetic",
        "truth": update((1.0, 0.0, 0.0)),
        "false_acceptance": update((1.0, 1.0, 0.0)),
    }
