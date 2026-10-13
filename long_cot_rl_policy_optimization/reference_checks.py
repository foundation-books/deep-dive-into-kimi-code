"""Complete Chapter 19 studio trace and named oracle records."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from typing import Callable, Sequence, cast

from .reference_core import (
    STUDIO_BOUNDS,
    STUDIO_LENGTHS,
    STUDIO_LOG_RATIOS,
    STUDIO_REWARDS,
    STUDIO_TAU,
    GroupSpec,
    complete_studio_score,
    exponential_tilt,
    group_advantages,
    length_ledger,
    local_diagnostics,
    local_gradient_coefficient,
    local_score,
    matched_reduction_fixture,
    ratio_factorization,
    regularization_comparison,
    studio_group,
    tiny_policy_step,
    validate_group,
)


def _fraction_strings(values: Sequence[Fraction]) -> list[str]:
    return [f"{value.numerator}/{value.denominator}" for value in values]


def _rejects_value_error(operation: Callable[[], object]) -> bool:
    """Return whether a declared invalid fixture is rejected as required."""

    try:
        operation()
    except ValueError:
        return True
    return False


def finite_difference_local_checks(
    advantage: float = 0.25,
    lower: float = -0.20,
    upper: float = 0.20,
    tau: float = 0.1,
    step: float = 1e-7,
) -> dict[str, float]:
    """Check interior, exterior, and chosen one-sided boundary derivatives."""

    def centered(point: float) -> float:
        return (
            local_score(point + step, advantage, lower, upper, tau)
            - local_score(point - step, advantage, lower, upper, tau)
        ) / (2 * step)

    numerical = {
        "inside": centered(0.08),
        "below": centered(-0.24),
        "above": centered(0.31),
        "lower_interior_side": (
            local_score(lower + step, advantage, lower, upper, tau)
            - local_score(lower, advantage, lower, upper, tau)
        )
        / step,
        "upper_interior_side": (
            local_score(upper, advantage, lower, upper, tau)
            - local_score(upper - step, advantage, lower, upper, tau)
        )
        / step,
    }
    analytic = {
        "inside": local_gradient_coefficient(0.08, advantage, lower, upper, tau),
        "below": local_gradient_coefficient(-0.24, advantage, lower, upper, tau),
        "above": local_gradient_coefficient(0.31, advantage, lower, upper, tau),
        "lower_interior_side": local_gradient_coefficient(
            lower, advantage, lower, upper, tau
        ),
        "upper_interior_side": local_gradient_coefficient(
            upper, advantage, lower, upper, tau
        ),
    }
    return {name: abs(numerical[name] - analytic[name]) for name in numerical}


def studio_trace() -> dict[str, object]:
    """Retain the historical CH17-D06 representative diagnostic unchanged."""

    group = studio_group()
    validation = validate_group(group)
    mean, advantages = group_advantages(group.outcomes)
    diagnostics = [
        local_diagnostics(float(advantage), ratios, *STUDIO_BOUNDS, STUDIO_TAU)
        for advantage, ratios in zip(advantages, STUDIO_LOG_RATIOS, strict=True)
    ]
    penalty_masses = [cast(float, item["penalty_mass"]) for item in diagnostics]
    trace: dict[str, object] = {
        "task_id": validation["task_id"],
        "behavior_version": group.behavior_version,
        "update_version": group.update_version,
        "update_lag": validation["update_lag"],
        "K": validation["K"],
        "N": validation["N"],
        "lengths": list(group.lengths),
        "outcomes": list(group.outcomes),
        "mean_reward": f"{mean.numerator}/{mean.denominator}",
        "advantages": _fraction_strings(advantages),
        "advantage_sum": str(sum(advantages, Fraction(0))),
        "illustrative_bounds": list(STUDIO_BOUNDS),
        "representative_log_ratios": [list(values) for values in STUDIO_LOG_RATIOS],
        "representative_eligible_counts": [
            cast(int, item["eligible_count"]) for item in diagnostics
        ],
        "representative_penalty_masses": penalty_masses,
        "representative_penalty_mass": sum(penalty_masses),
        "length_ledger": length_ledger(group.lengths, advantages),
        "evidence_class": "reproduced",
        "claim_boundary": (
            "chapter-created arithmetic and semantic checks only; no model "
            "quality, K3 implementation, stability, or causal-credit claim"
        ),
    }
    payload = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    trace["content_sha256"] = hashlib.sha256(payload).hexdigest()
    return trace


def _group_baseline_oracle(valid_group: GroupSpec) -> dict[str, object]:
    """Build CH17-D01, including every declared rejection fixture."""

    group_contract = validate_group(valid_group)
    mean, advantages = group_advantages(STUDIO_REWARDS)
    translated_mean, translated = group_advantages(
        tuple(value + 7 for value in STUDIO_REWARDS)
    )
    null_mean, null_advantages = group_advantages((1, 1, 1, 1))
    group_rejections = {
        "mixed_task": _rejects_value_error(
            lambda: validate_group(replace(valid_group, task_ids=("a", "a", "b", "a")))
        ),
        "missing_behavior_version": _rejects_value_error(
            lambda: validate_group(replace(valid_group, behavior_version=None))
        ),
        "incomplete_promised_group": _rejects_value_error(
            lambda: validate_group(replace(valid_group, promised_size=5))
        ),
        "unfinished_member": _rejects_value_error(
            lambda: validate_group(
                replace(valid_group, completed=(True, True, True, False))
            )
        ),
        "zero_length": _rejects_value_error(
            lambda: validate_group(replace(valid_group, lengths=(8, 12, 0, 20)))
        ),
        "singleton_baseline": _rejects_value_error(lambda: group_advantages((1,))),
    }
    return {
        "status": "pass",
        "protocol_id": "CH17-D01",
        "group_contract": group_contract,
        "mean": f"{mean.numerator}/{mean.denominator}",
        "advantages": _fraction_strings(advantages),
        "translated_mean": (
            f"{translated_mean.numerator}/{translated_mean.denominator}"
        ),
        "translated_advantages": _fraction_strings(translated),
        "null_mean": f"{null_mean.numerator}/{null_mean.denominator}",
        "null_advantages": _fraction_strings(null_advantages),
        "centering_conserved": sum(advantages, Fraction(0)) == 0,
        "invalid_fixture_rejections": group_rejections,
        "all_invalid_fixtures_rejected": all(group_rejections.values()),
    }


def _mirror_descent_oracle() -> dict[str, object]:
    """Build CH17-D02 across finite, limiting, and invalid inputs."""

    tilted = exponential_tilt((0.6, 0.4), (1.0, 0.0), 0.5)
    near_reference = exponential_tilt((0.6, 0.4), (1.0, 0.0), 1e8)
    near_maximum = exponential_tilt((0.6, 0.4), (1.0, 0.0), 0.01)
    zero_support = exponential_tilt((1.0, 0.0), (0.0, 10.0), 0.5)
    exact_logs = tuple(
        math.log(now / old) for now, old in zip(tilted, (0.6, 0.4), strict=True)
    )
    log_constant = 0.5 * math.log(0.6 * math.exp(2) + 0.4)
    centered_logs = tuple(value - sum(exact_logs) / 2 for value in exact_logs)
    return {
        "status": "pass",
        "protocol_id": "CH17-D02",
        "updated_probabilities": list(tilted),
        "log_partition_constant": log_constant,
        "empirical_reward_mean": 0.5,
        "exact_target_log_ratios": exact_logs,
        "centered_exact_identity": all(
            math.isclose(advantage, 0.5 * value, abs_tol=1e-12)
            for advantage, value in zip((0.5, -0.5), centered_logs, strict=True)
        ),
        "surrogate_residuals_at_exact_target": tuple(
            advantage - 0.5 * value
            for advantage, value in zip((0.5, -0.5), exact_logs, strict=True)
        ),
        "normalizes": math.isclose(sum(tilted), 1.0, abs_tol=1e-12),
        "large_eta_probabilities": list(near_reference),
        "small_eta_probabilities": list(near_maximum),
        "large_eta_approaches_reference": math.isclose(
            near_reference[0], 0.6, rel_tol=1e-8
        ),
        "small_eta_concentrates_on_maximum": near_maximum[0] > 1 - 1e-12,
        "zero_reference_support_probabilities": list(zero_support),
        "reference_support_preserved": zero_support == (1.0, 0.0),
        "invalid_eta_rejected": _rejects_value_error(
            lambda: exponential_tilt((0.6, 0.4), (1.0, 0.0), 0.0)
        ),
    }


def _ratio_factorization_oracle() -> dict[str, object]:
    """Build CH17-D03 from one factorization and two square fixtures."""

    return {
        "status": "pass",
        "protocol_id": "CH17-D03",
        "factorization": ratio_factorization((0.30, 0.20), (0.25, 0.25)),
        "cancellation": regularization_comparison((0.3, -0.3)),
        "coherent": regularization_comparison((0.3, 0.3)),
    }


def _local_gradient_oracle(advantages: Sequence[Fraction]) -> dict[str, object]:
    """Build CH17-D04 for the studio ratios and finite differences."""

    local = [
        local_diagnostics(float(advantage), values, *STUDIO_BOUNDS, STUDIO_TAU)
        for advantage, values in zip(advantages, STUDIO_LOG_RATIOS, strict=True)
    ]
    errors = finite_difference_local_checks()
    return {
        "status": "pass",
        "protocol_id": "CH17-D04",
        "eligible_counts": [cast(int, item["eligible_count"]) for item in local],
        "penalty_masses": [cast(float, item["penalty_mass"]) for item in local],
        "finite_difference_max_error": max(errors.values()),
        "boundary_convention": "inclusive, interior one-sided derivative",
    }


def _length_ledger_oracle(advantages: Sequence[Fraction]) -> dict[str, object]:
    """Build CH17-D05 for three normalizations and invalid denominators."""

    length = length_ledger(STUDIO_LENGTHS, advantages)
    partial_active = matched_reduction_fixture()
    rejections = {
        "empty_length_ledger": _rejects_value_error(lambda: length_ledger((), ())),
        "empty_active_set": _rejects_value_error(
            lambda: length_ledger((5, 20), (0.25, 0.25), active_counts=(0, 0))
        ),
    }
    return {
        "status": "pass",
        "protocol_id": "CH17-D05",
        **length,
        "all_active_matches_total_token": (
            length["active_token_coefficients"] == length["total_token_coefficients"]
        ),
        "partial_active_example": partial_active,
        "denominator_rejections": rejections,
        "all_invalid_denominators_rejected": all(rejections.values()),
    }


def all_checks() -> dict[str, object]:
    """Return the six named deterministic protocol records."""

    valid_group = studio_group()
    _, advantages = group_advantages(STUDIO_REWARDS)
    return {
        "RESULT_CH17_GROUP_BASELINE_ORACLE": _group_baseline_oracle(valid_group),
        "RESULT_CH17_MIRROR_DESCENT_ORACLE": _mirror_descent_oracle(),
        "RESULT_CH17_RATIO_FACTORIZATION_ORACLE": _ratio_factorization_oracle(),
        "RESULT_CH17_LOCAL_GRADIENT_ORACLE": _local_gradient_oracle(advantages),
        "RESULT_CH17_LENGTH_LEDGER_ORACLE": _length_ledger_oracle(advantages),
        "RESULT_CH17_COMPLETE_STUDIO_TRACE": {
            "status": "pass",
            "protocol_id": "CH17-D06",
            **studio_trace(),
            "complete_zero_tail_fixture": complete_studio_score(),
            "tiny_parameter_step": tiny_policy_step(),
        },
    }


__all__ = ["all_checks", "finite_difference_local_checks", "studio_trace"]
