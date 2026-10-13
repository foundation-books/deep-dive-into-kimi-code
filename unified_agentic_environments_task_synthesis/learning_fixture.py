"""A one-parameter learning decision after a completed teaching episode."""

from __future__ import annotations

import json
from math import exp, log


INITIAL_P = 0.2
LEARNING_RATE = 0.25


def probability(logit: float) -> float:
    """Map the one trainable logit to reconciliation probability."""

    return 1.0 / (1.0 + exp(-logit))


def sample_gradient(action: int, reward: int, logit: float) -> float:
    """Raw-reward score-function sample; no baseline or regularizer."""
    if action not in (0, 1) or reward not in (0, 1):
        raise ValueError("binary action and reward required")
    return reward * (action - probability(logit))


def finite_update() -> dict[str, float]:
    """Only the reconcile-vs-calendar-only choice is trainable."""
    before_logit = log(INITIAL_P / (1 - INITIAL_P))
    successful_gradient = sample_gradient(1, 1, before_logit)
    failure_gradient = sample_gradient(0, 0, before_logit)
    next_logit = before_logit + LEARNING_RATE * successful_gradient
    return {
        "before_logit": before_logit,
        "before_probability": probability(before_logit),
        "successful_sample_gradient": successful_gradient,
        "failure_sample_gradient": failure_gradient,
        "expected_gradient": INITIAL_P * (1 - INITIAL_P),
        "learning_rate": LEARNING_RATE,
        "after_logit": next_logit,
        "after_probability": probability(next_logit),
    }


def annotated_record() -> list[dict[str, object]]:
    """An observation conditions later action, but carries no action loss."""
    return [
        {
            "kind": "instruction",
            "content": "Coordinate Ada through Day 3.",
            "assistant_action_mask": 0,
        },
        {
            "kind": "assistant_action",
            "content": "request_provision(Day 4, Mira)",
            "assistant_action_mask": 1,
        },
        {
            "kind": "tool_observation",
            "content": "accepted_pending",
            "assistant_action_mask": 0,
        },
        {
            "kind": "tool_observation",
            "content": "Day 4 grant became available after date moved to Day 6",
            "assistant_action_mask": 0,
        },
        {
            "kind": "assistant_action",
            "content": "reconcile_grant(Day 6, Tomas)",
            "assistant_action_mask": 1,
        },
        {
            "kind": "evaluator",
            "content": "completed task: 1",
            "assistant_action_mask": 0,
        },
    ]


if __name__ == "__main__":
    print(json.dumps(finite_update(), indent=2, sort_keys=True))
