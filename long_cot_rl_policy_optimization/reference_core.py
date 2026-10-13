"""Dependency-free semantic oracles for Chapter 19.

The module reproduces the chapter-created rollout-group arithmetic,
finite-space exponential tilt, sequence/token ratio identities, explicit
local-gradient convention, length ledgers, and complete studio trace.  It is a
teaching reference, not a Moonshot AI trainer or a reproduction of Kimi K3.
"""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Sequence, cast

from .reference_group import (
    STUDIO_BEHAVIOR_VERSION as STUDIO_BEHAVIOR_VERSION,
    STUDIO_LENGTHS,
    STUDIO_REWARDS,
    STUDIO_UPDATE_VERSION as STUDIO_UPDATE_VERSION,
    GroupSpec as GroupSpec,
    _fraction,
    group_advantages,
    studio_group as studio_group,
    validate_group as validate_group,
)


STUDIO_BOUNDS = (-0.20, 0.20)
STUDIO_TAU = 0.1
STUDIO_LOG_RATIOS = (
    (0.00, 0.08, -0.04, 0.12),
    (-0.15, 0.05, 0.31, -0.08),
    (-0.18, -0.24, 0.07, 0.02),
    (0.03, 0.11, 0.19, 0.27),
)
# Preserve the historical 16-token diagnostic; explicitly complete all 48 tokens.
STUDIO_COMPLETE_LOG_RATIOS = tuple(
    values + (0.0,) * (length - len(values))
    for values, length in zip(STUDIO_LOG_RATIOS, STUDIO_LENGTHS, strict=True)
)
STUDIO_REDUCTION_COUNTS = (8, 6, 8, 10)
STUDIO_REDUCTION_LOG_RATIOS = tuple(
    (0.0,) * count + (0.3,) * (length - count)
    for count, length in zip(STUDIO_REDUCTION_COUNTS, STUDIO_LENGTHS, strict=True)
)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def exponential_tilt(
    reference_probabilities: Sequence[float],
    rewards: Sequence[float],
    eta: float,
) -> tuple[float, ...]:
    """Solve D12 on a finite, normalized trajectory support."""

    eta = _finite(eta, "eta")
    if eta <= 0:
        raise ValueError("eta must be positive")
    if len(reference_probabilities) != len(rewards) or not rewards:
        raise ValueError("probabilities and rewards need equal nonzero length")
    probabilities = tuple(
        _finite(value, "reference probability") for value in reference_probabilities
    )
    reward_values = tuple(_finite(value, "reward") for value in rewards)
    if any(value < 0 for value in probabilities):
        raise ValueError("reference probabilities must be nonnegative")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("reference probabilities must sum to one")
    log_weights = [
        -math.inf if probability == 0 else math.log(probability) + reward / eta
        for probability, reward in zip(probabilities, reward_values, strict=True)
    ]
    maximum = max(log_weights)
    if not math.isfinite(maximum):
        raise ValueError("reference support is empty")
    weights = [
        0.0 if value == -math.inf else math.exp(value - maximum)
        for value in log_weights
    ]
    normalizer = sum(weights)
    result = tuple(value / normalizer for value in weights)
    if not math.isclose(sum(result), 1.0, abs_tol=1e-12):
        raise AssertionError("exponential tilt did not normalize")
    return result


def ratio_factorization(
    current_probabilities: Sequence[float],
    behavior_probabilities: Sequence[float],
) -> dict[str, object]:
    """Compute matching token ratios and their sequence aggregation."""

    if (
        len(current_probabilities) != len(behavior_probabilities)
        or not current_probabilities
    ):
        raise ValueError("token supports must have equal nonzero length")
    current = tuple(
        _finite(value, "current probability") for value in current_probabilities
    )
    behavior = tuple(
        _finite(value, "behavior probability") for value in behavior_probabilities
    )
    if any(value <= 0 for value in current + behavior):
        raise ValueError("sampled-token probabilities must be positive")
    token_ratios = tuple(now / old for now, old in zip(current, behavior, strict=True))
    token_log_ratios = tuple(math.log(value) for value in token_ratios)
    sequence_ratio_product = math.prod(token_ratios)
    sequence_log_ratio = sum(token_log_ratios)
    sequence_ratio_from_log = math.exp(sequence_log_ratio)
    return {
        "token_ratios": token_ratios,
        "token_log_ratios": token_log_ratios,
        "sequence_ratio_product": sequence_ratio_product,
        "sequence_log_ratio": sequence_log_ratio,
        "sequence_ratio_from_log": sequence_ratio_from_log,
        "factorization_holds": math.isclose(
            sequence_ratio_product, sequence_ratio_from_log, rel_tol=1e-12
        ),
    }


def regularization_comparison(log_ratios: Sequence[float]) -> dict[str, float]:
    """Return the squared sequence value, local squares, and cross term."""

    if not log_ratios:
        raise ValueError("at least one token log ratio is required")
    values = tuple(_finite(value, "log ratio") for value in log_ratios)
    sequence_log_ratio = sum(values)
    local_square_sum = sum(value * value for value in values)
    cross_term = 2 * sum(
        values[left] * values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    sequence_square = sequence_log_ratio * sequence_log_ratio
    if not math.isclose(sequence_square, local_square_sum + cross_term, abs_tol=1e-12):
        raise AssertionError("squared-sum identity failed")
    return {
        "sequence_log_ratio": sequence_log_ratio,
        "sequence_square": sequence_square,
        "local_square_sum": local_square_sum,
        "cross_term": cross_term,
    }


def local_factor(log_ratio: float, lower: float, upper: float) -> float:
    """Return the continuous boundary-held factor phi of the declared local rule."""

    log_ratio = _finite(log_ratio, "log ratio")
    lower = _finite(lower, "lower bound")
    upper = _finite(upper, "upper bound")
    if lower >= upper:
        raise ValueError("local bounds must satisfy lower < upper")
    return math.exp(min(max(log_ratio, lower), upper))


def local_score(
    log_ratio: float,
    advantage: float,
    lower: float,
    upper: float,
    tau: float,
) -> float:
    """Return one token's chapter-created ascent score."""

    log_ratio = _finite(log_ratio, "log ratio")
    advantage = _finite(advantage, "advantage")
    tau = _finite(tau, "tau")
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    return advantage * local_factor(log_ratio, lower, upper) - tau * log_ratio**2


def local_gradient_coefficient(
    log_ratio: float,
    advantage: float,
    lower: float,
    upper: float,
    tau: float,
) -> float:
    """Return the scalar multiplying the sampled-token score vector."""

    log_ratio = _finite(log_ratio, "log ratio")
    advantage = _finite(advantage, "advantage")
    lower = _finite(lower, "lower bound")
    upper = _finite(upper, "upper bound")
    tau = _finite(tau, "tau")
    if lower >= upper or tau < 0:
        raise ValueError("invalid local-gradient configuration")
    eligible = lower <= log_ratio <= upper
    reward = advantage * math.exp(log_ratio) if eligible else 0.0
    return reward - 2 * tau * log_ratio


def local_diagnostics(
    advantage: float,
    log_ratios: Sequence[float],
    lower: float,
    upper: float,
    tau: float,
) -> dict[str, object]:
    """Return reward eligibility, forward factors, and gradient coefficients."""

    if not log_ratios:
        raise ValueError("at least one log ratio is required")
    records = []
    for raw in log_ratios:
        log_ratio = _finite(raw, "log ratio")
        eligible = lower <= log_ratio <= upper
        records.append(
            {
                "log_ratio": log_ratio,
                "probability_ratio": math.exp(log_ratio),
                "eligible": eligible,
                "local_factor": local_factor(log_ratio, lower, upper),
                "reward_gradient_coefficient": (
                    advantage * math.exp(log_ratio) if eligible else 0.0
                ),
                "penalty_loss": tau * log_ratio**2,
                "restoring_gradient_coefficient": -2 * tau * log_ratio,
                "total_gradient_coefficient": local_gradient_coefficient(
                    log_ratio, advantage, lower, upper, tau
                ),
            }
        )
    return {
        "records": records,
        "eligible_count": sum(bool(record["eligible"]) for record in records),
        "penalty_mass": sum(float(record["penalty_loss"]) for record in records),
    }


def length_ledger(
    lengths: Sequence[int],
    advantages: Sequence[int | float | Fraction],
    ratio: float = 1.0,
    active_counts: Sequence[int] | None = None,
) -> dict[str, object]:
    """Compare total-token, trajectory, and active-token coefficients.

    ``active_counts[j]`` is the number of reward-eligible tokens in trajectory
    ``j``; ``ratio`` is common to eligible tokens only. All three reductions
    use that same eligible reward sum. Omitting counts makes every token active.
    Advantages are supplied frozen coefficients, not recentered here. This
    oracle rejects an empty active set; a caller that wants a null-reward
    fallback must give that fallback a separate name and implementation.
    """

    if len(lengths) != len(advantages) or not lengths:
        raise ValueError("lengths and advantages need equal nonzero length")
    if any(
        isinstance(length, bool) or not isinstance(length, int) or length <= 0
        for length in lengths
    ):
        raise ValueError("lengths must be positive integers")
    ratio = _finite(ratio, "ratio")
    if ratio <= 0:
        raise ValueError("probability ratio must be positive")
    values = tuple(_fraction(value) for value in advantages)
    active = tuple(lengths if active_counts is None else active_counts)
    if len(active) != len(lengths):
        raise ValueError("active counts must match the trajectory count")
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or count > length
        for count, length in zip(active, lengths, strict=True)
    ):
        raise ValueError("active counts must be integers between zero and length")
    total_tokens = sum(lengths)
    active_tokens = sum(active)
    if active_tokens == 0:
        raise ValueError("active-token normalization has an empty denominator")
    total_token_coefficients = tuple(
        float(Fraction(count) * advantage / total_tokens) * ratio
        for count, advantage in zip(active, values, strict=True)
    )
    per_trajectory_coefficients = tuple(
        float(Fraction(count, length) * advantage / len(values)) * ratio
        for count, length, advantage in zip(active, lengths, values, strict=True)
    )
    active_token_coefficients = tuple(
        float(Fraction(count) * advantage / active_tokens) * ratio
        for count, advantage in zip(active, values, strict=True)
    )
    return {
        "total_tokens": total_tokens,
        "active_counts": active,
        "active_tokens": active_tokens,
        "total_token_coefficients": total_token_coefficients,
        "per_trajectory_coefficients": per_trajectory_coefficients,
        "active_token_coefficients": active_token_coefficients,
        "total_token_sum": sum(total_token_coefficients),
        "per_trajectory_sum": sum(per_trajectory_coefficients),
        "active_token_sum": sum(active_token_coefficients),
    }


def complete_studio_score() -> dict[str, object]:
    """Compute the full zero-tail fixture, including every token coefficient."""

    _, advantages = group_advantages(STUDIO_REWARDS)
    paths = [
        local_diagnostics(float(advantage), values, *STUDIO_BOUNDS, STUDIO_TAU)
        for advantage, values in zip(
            advantages, STUDIO_COMPLETE_LOG_RATIOS, strict=True
        )
    ]
    forward = sum(
        float(advantage) * local_factor(value, *STUDIO_BOUNDS)
        for advantage, values in zip(
            advantages, STUDIO_COMPLETE_LOG_RATIOS, strict=True
        )
        for value in values
    )
    penalty = STUDIO_TAU * sum(
        value**2 for values in STUDIO_COMPLETE_LOG_RATIOS for value in values
    )
    reward = sum(
        float(advantage) * math.exp(value)
        for advantage, values in zip(
            advantages, STUDIO_COMPLETE_LOG_RATIOS, strict=True
        )
        for value in values
        if STUDIO_BOUNDS[0] <= value <= STUDIO_BOUNDS[1]
    )
    restoring = -2 * STUDIO_TAU * sum(map(sum, STUDIO_COMPLETE_LOG_RATIOS))
    return {
        "fixture_id": "studio-zero-tail-v1",
        "lengths": tuple(map(len, STUDIO_COMPLETE_LOG_RATIOS)),
        "log_ratios": STUDIO_COMPLETE_LOG_RATIOS,
        "N": sum(STUDIO_LENGTHS),
        "U": sum(cast(int, item["eligible_count"]) for item in paths),
        "paths": paths,
        "forward_reward_sum": forward,
        "penalty_sum": penalty,
        "score": (forward - penalty) / sum(STUDIO_LENGTHS),
        "reward_coefficient_sum": reward,
        "restoring_coefficient_sum": restoring,
        "normalized_coefficient_sum": (reward + restoring) / sum(STUDIO_LENGTHS),
    }


def matched_reduction_fixture() -> dict[str, object]:
    """Hold paths, advantages, log ratios, and eligibility fixed across reductions."""

    _, advantages = group_advantages(STUDIO_REWARDS)
    counts = tuple(
        sum(STUDIO_BOUNDS[0] <= value <= STUDIO_BOUNDS[1] for value in values)
        for values in STUDIO_REDUCTION_LOG_RATIOS
    )
    return {
        **length_ledger(STUDIO_LENGTHS, advantages, active_counts=counts),
        "log_ratios": STUDIO_REDUCTION_LOG_RATIOS,
        "regularizer_total_token_mean": STUDIO_TAU
        * sum(value**2 for values in STUDIO_REDUCTION_LOG_RATIOS for value in values)
        / sum(STUDIO_LENGTHS),
    }


def tiny_policy_step(logit: float = math.log(0.3 / 0.7)) -> dict[str, float]:
    """One actual ascent step for a shared binary-answer logit at a fixed history.

    The saved group contains blue and red once each, sampled from (1/4, 3/4).
    Its rewards (1, 0) give advantages (1/2, -1/2); N=2. The same logit controls
    both probabilities. Bounds and regularization use the studio convention.
    """

    logit = _finite(logit, "logit")
    blue = 1 / (1 + math.exp(-logit))
    red = 1 - blue
    blue_log_ratio, red_log_ratio = math.log(blue / 0.25), math.log(red / 0.75)
    blue_coefficient = local_gradient_coefficient(
        blue_log_ratio, 0.5, *STUDIO_BOUNDS, STUDIO_TAU
    )
    red_coefficient = local_gradient_coefficient(
        red_log_ratio, -0.5, *STUDIO_BOUNDS, STUDIO_TAU
    )
    score = (
        local_score(blue_log_ratio, 0.5, *STUDIO_BOUNDS, STUDIO_TAU)
        + local_score(red_log_ratio, -0.5, *STUDIO_BOUNDS, STUDIO_TAU)
    ) / 2
    gradient = (blue_coefficient * red - red_coefficient * blue) / 2
    next_logit = logit + 0.4 * gradient
    next_blue = 1 / (1 + math.exp(-next_logit))
    return {
        "blue": blue,
        "red": red,
        "blue_log_ratio": blue_log_ratio,
        "red_log_ratio": red_log_ratio,
        "blue_coefficient": blue_coefficient,
        "red_coefficient": red_coefficient,
        "score": score,
        "parameter_gradient": gradient,
        "learning_rate": 0.4,
        "next_logit": next_logit,
        "next_blue": next_blue,
        "next_red": 1 - next_blue,
    }
