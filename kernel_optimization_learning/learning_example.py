"""Finite first-decision returns and one exact binary-policy update.

All scores are constructed teaching values. The updated policy selects a
strategy at the episode's first decision; later candidate scores determine
that decision's return but are not extra policy choices in this tiny model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math


SCORES_D = (0.4, 0.4, 0.4)
SCORES_E = (0.0, 0.4, 1.0)


def teaching_score(
    outcome: str, *, runtime: float | None = None, baseline: float | None = None
) -> float | None:
    """Score a rejected or checked candidate; leave unresolved work unscored.

    ``passed`` means that the caller has already completed all required
    numerical checks. Build and numerical failures are separate records even
    though both use the local ``candidate_rejected`` score branch.
    """
    if outcome == "candidate_rejected":
        return 0.0
    if outcome == "unresolved":
        return None
    if outcome != "passed":
        raise ValueError("unknown evaluation outcome")
    if runtime is None or baseline is None:
        return None
    if not all(math.isfinite(value) and value > 0 for value in (runtime, baseline)):
        return None
    return min(1.0, 0.4 * baseline / runtime)


@dataclass(frozen=True)
class PolicyStep:
    """One ascent step for the distribution over first decisions D and E."""

    return_d: float
    return_e: float
    baseline: float
    advantage_d: float
    advantage_e: float
    probability_before: float
    expected_return_before: float
    gradient: float
    logit_after: float
    probability_after: float
    expected_return_after: float


def discounted_returns(scores: Sequence[float], gamma: float) -> tuple[float, ...]:
    """Assign each turn its sum of later discounted candidate scores."""
    if not scores or not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("need scores and a finite discount between zero and one")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    running = 0.0
    backward: list[float] = []
    for score in reversed(scores):
        running = float(score) + gamma * running
        backward.append(running)
    return tuple(reversed(backward))


def grouped_turn_coefficients(
    returns: Sequence[float],
    group_keys: Sequence[tuple[str, int | None]],
    loss_mask: Sequence[bool],
) -> tuple[float, ...]:
    """Subtract the mean of other returns in each group, then mask actor terms.

    A key is (prompt, turn); ``(prompt, None)`` models a prompt-only grouping
    for comparison. The mask excludes an assistant turn from the loss without
    deleting its return from a complete group's baseline. A value of zero is
    therefore either a zero advantage or a masked-out actor coefficient.
    """
    if not returns or len(returns) != len(group_keys) or len(returns) != len(loss_mask):
        raise ValueError("returns, group keys, and loss mask must align")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("returns must be finite")
    if not all(isinstance(flag, bool) for flag in loss_mask):
        raise ValueError("loss mask must contain booleans")
    groups: dict[tuple[str, int | None], list[int]] = {}
    for index, key in enumerate(group_keys):
        groups.setdefault(key, []).append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("each leave-one-out group needs at least two turns")
    coefficients = [0.0] * len(returns)
    for indices in groups.values():
        total = math.fsum(returns[index] for index in indices)
        for index in indices:
            if loss_mask[index]:
                other_mean = (total - returns[index]) / (len(indices) - 1)
                coefficients[index] = returns[index] - other_mean
    return tuple(coefficients)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def first_decision_policy_step(
    scores_d: Sequence[float],
    scores_e: Sequence[float],
    *,
    gamma: float,
    logit: float,
    learning_rate: float,
) -> PolicyStep:
    """Update P(E)=sigmoid(logit) using the same first-turn return objective.

    The policy-weighted baseline is frozen for this score-function step. Its
    terms cancel in the exact expected gradient. This enumerates both first
    actions; it does not differentiate compiler output or GPU execution.
    """
    if not math.isfinite(logit) or not math.isfinite(learning_rate):
        raise ValueError("logit and learning rate must be finite")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    return_d = discounted_returns(scores_d, gamma)[0]
    return_e = discounted_returns(scores_e, gamma)[0]
    probability_before = _sigmoid(logit)
    expected_return_before = (
        1.0 - probability_before
    ) * return_d + probability_before * return_e
    baseline = expected_return_before
    advantage_d = return_d - baseline
    advantage_e = return_e - baseline
    gradient = probability_before * (1.0 - probability_before) * (return_e - return_d)
    logit_after = logit + learning_rate * gradient
    probability_after = _sigmoid(logit_after)
    expected_return_after = (
        1.0 - probability_after
    ) * return_d + probability_after * return_e
    return PolicyStep(
        return_d,
        return_e,
        baseline,
        advantage_d,
        advantage_e,
        probability_before,
        expected_return_before,
        gradient,
        logit_after,
        probability_after,
        expected_return_after,
    )


def whole_operation_speedup(
    unchanged_fraction: float, component_speedup: float
) -> float:
    """Return speedup under a declared additive-time cost model, not a timing."""
    if not 0 <= unchanged_fraction <= 1 or component_speedup <= 0:
        raise ValueError("need a fraction in [0,1] and positive component speedup")
    if not math.isfinite(unchanged_fraction) or not math.isfinite(component_speedup):
        raise ValueError("cost-model inputs must be finite")
    return 1.0 / (unchanged_fraction + (1.0 - unchanged_fraction) / component_speedup)


if __name__ == "__main__":
    for discount in (0.5, 1.0):
        print(
            "gamma",
            discount,
            discounted_returns(SCORES_D, discount),
            discounted_returns(SCORES_E, discount),
        )
    print(
        first_decision_policy_step(
            SCORES_D, SCORES_E, gamma=1.0, logit=0.0, learning_rate=0.5
        )
    )
    turn_returns = (
        *discounted_returns((1.0, 4.0), 1.0),
        *discounted_returns((3.0, 2.0), 1.0),
    )
    print(
        "grouped-turn probe:",
        turn_returns,
        grouped_turn_coefficients(
            turn_returns,
            (("task", 1), ("task", 2), ("task", 1), ("task", 2)),
            (True, True, True, True),
        ),
    )
