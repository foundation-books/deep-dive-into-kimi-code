"""Finite deterministic oracles for Chapter 30 evaluation methodology.

The module implements only the chapter-created ``studio-eval-001`` fixtures.
It performs no model inference, network access, repository mutation, or
empirical evaluation.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Mapping, Sequence


FIXTURE_VERSION = "ch30-studio-eval-v1"


REQUIRED_SYSTEM_FIELDS = {"model", "decoding", "harness", "tools", "context"}
REQUIRED_TASK_FIELDS = {
    "id",
    "request",
    "initial_state",
    "authority",
    "verifier",
    "version",
}
REQUIRED_PROTOCOL_FIELDS = {
    "system",
    "population",
    "sample",
    "environment",
    "budget",
    "evaluator",
    "retry",
    "normalization",
    "version",
}


def validate_record(
    system: Mapping[str, object],
    task: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[bool, list[str]]:
    """Validate required identities without converting defects into task failures."""
    missing: list[str] = []
    for prefix, required, record in (
        ("system", REQUIRED_SYSTEM_FIELDS, system),
        ("task", REQUIRED_TASK_FIELDS, task),
        ("protocol", REQUIRED_PROTOCOL_FIELDS, protocol),
    ):
        missing.extend(f"{prefix}.{field}" for field in sorted(required - set(record)))
    return not missing, missing


def compare_protocols(
    left: Mapping[str, object],
    right: Mapping[str, object],
    relevant_fields: set[str],
    declared_delta: set[str],
    bridged_fields: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Classify one claim-relative protocol comparison with a diff witness."""
    diff = sorted(
        field for field in relevant_fields if left.get(field) != right.get(field)
    )
    changed = set(diff)
    if not changed:
        return "equivalent", diff
    if changed == declared_delta:
        return "controlled_difference", diff
    if changed <= (bridged_fields or set()):
        return "bridge_required", diff
    return "incomparable", diff


def pass_at_k(n: int, c: int, k: int) -> Fraction:
    """Return the unbiased finite-sample pass@k estimator exactly."""
    if not (n >= k >= 1 and 0 <= c <= n):
        raise ValueError("require n >= k >= 1 and 0 <= c <= n")
    failures = n - c
    numerator = 0 if failures < k else comb(failures, k)
    return Fraction(1, 1) - Fraction(numerator, comb(n, k))


def paired_effect(
    left: Sequence[int], right: Sequence[int]
) -> tuple[Fraction, list[int]]:
    """Return an equal-weight paired effect and one-based discordant indices."""
    if not left or len(left) != len(right):
        raise ValueError("paired sequences must have equal positive length")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return Fraction(sum(differences), len(differences)), [
        index for index, value in enumerate(differences, 1) if value != 0
    ]


def disposition_rates(
    dispositions: Sequence[str],
) -> tuple[Fraction, Fraction, dict[str, int]]:
    """Compute semantic and all-dispatch rates while retaining source states."""
    counts = {name: dispositions.count(name) for name in sorted(set(dispositions))}
    semantic = counts.get("pass", 0) + counts.get("fail", 0)
    if semantic == 0 or not dispositions:
        raise ValueError("both semantic and dispatch denominators must be positive")
    passes = counts.get("pass", 0)
    return Fraction(passes, semantic), Fraction(passes, len(dispositions)), counts


def weighted_rate(values: Sequence[Fraction], weights: Sequence[Fraction]) -> Fraction:
    """Return a weighted rate after validating a probability vector."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must have equal positive length")
    if any(weight < 0 for weight in weights) or sum(weights) != 1:
        raise ValueError("weights must be nonnegative and sum to one")
    return sum(
        (value * weight for value, weight in zip(values, weights, strict=True)),
        Fraction(),
    )


def judge_order_consistency(
    forward: Sequence[str], reversed_normalized: Sequence[str]
) -> Fraction:
    """Compute exact order consistency after candidates are normalized by identity."""
    if not forward or len(forward) != len(reversed_normalized):
        raise ValueError("judge sequences must have equal positive length")
    agreements = sum(a == b for a, b in zip(forward, reversed_normalized, strict=True))
    return Fraction(agreements, len(forward))


def pareto_relation(
    left: Sequence[Fraction | int], right: Sequence[Fraction | int]
) -> str:
    """Compare vectors whose coordinates are all oriented larger-is-better."""
    if not left or len(left) != len(right):
        raise ValueError("vectors must have equal positive length")
    left_no_worse = all(a >= b for a, b in zip(left, right, strict=True))
    right_no_worse = all(b >= a for a, b in zip(left, right, strict=True))
    if left_no_worse and any(a > b for a, b in zip(left, right, strict=True)):
        return "left_dominates"
    if right_no_worse and any(b > a for a, b in zip(left, right, strict=True)):
        return "right_dominates"
    if left_no_worse and right_no_worse:
        return "equal"
    return "incomparable"


def swarm_accounting(
    serial_work: int, serial_critical: int, work: int, critical: int
) -> dict[str, Fraction]:
    """Return exact work factor and critical-step speedup for a matched fixture."""
    if min(serial_work, serial_critical, work, critical) <= 0 or critical > work:
        raise ValueError(
            "work and critical steps must be positive with critical <= work"
        )
    return {
        "critical_step_speedup": Fraction(serial_critical, critical),
        "work_factor": Fraction(work, serial_work),
    }


def claim_admissible(
    claim: Mapping[str, object],
    evidence: Mapping[str, object],
    comparable: bool,
    intervention_complete: bool,
) -> tuple[bool, list[str]]:
    """Check identity, comparison, and mechanism gates for one candidate claim."""
    reasons: list[str] = []
    for field in ("system", "population", "protocol", "metric"):
        if claim.get(field) != evidence.get(field):
            reasons.append(f"mismatched_{field}")
    if claim.get("kind") == "ranking" and not comparable:
        reasons.append("incomparable_ranking")
    if claim.get("kind") == "mechanism" and not intervention_complete:
        reasons.append("missing_intervention")
    if evidence.get("status") in {"pending", "unknown"}:
        reasons.append(f"evidence_{evidence['status']}")
    return not reasons, reasons


def _result(passed: bool, **payload: object) -> dict[str, object]:
    return {
        "status": "pass" if passed else "fail",
        "fixture_version": FIXTURE_VERSION,
        **payload,
    }


def _record_protocol_checks() -> dict[str, dict[str, object]]:
    system = {
        "model": "K",
        "decoding": "max",
        "harness": "H0",
        "tools": ["shell"],
        "context": "k0",
    }
    task = {
        "id": "A2",
        "request": "release",
        "initial_state": "studio@r0",
        "authority": "local",
        "verifier": "v1",
        "version": "r1",
    }
    protocol = {
        "system": "S_M",
        "population": "studio-eval-001",
        "sample": "twelve",
        "environment": "E0",
        "budget": "B0",
        "evaluator": "J0",
        "retry": "R0",
        "normalization": "N0",
        "version": "P1",
    }
    valid, missing = validate_record(system, task, protocol)
    invalid, invalid_missing = validate_record(
        {key: value for key, value in system.items() if key != "harness"},
        task,
        protocol,
    )

    left = {"effort": "low", "harness": "H0", "tools": "T0", "tasks": "D0"}
    effort = {**left, "effort": "max"}
    collision = {**effort, "tools": "T1", "tasks": "D1"}
    relevant = {"effort", "harness", "tools", "tasks"}
    return {
        "record_validation": _result(
            valid and not invalid and invalid_missing == ["system.harness"],
            missing=missing,
            invalid_missing=invalid_missing,
        ),
        "protocol_comparison": _result(
            compare_protocols(left, left, relevant, set())[0] == "equivalent"
            and compare_protocols(left, effort, relevant, {"effort"})[0]
            == "controlled_difference"
            and compare_protocols(left, collision, relevant, {"effort"})[0]
            == "incomparable"
        ),
    }


def _metric_checks() -> dict[str, dict[str, object]]:
    paired, discordant = paired_effect((1, 0, 1, 0), (1, 1, 0, 0))
    semantic, dispatch, counts = disposition_rates(
        ["pass"] * 7 + ["fail"] * 2 + ["invalid", "timeout", "infrastructure_error"]
    )
    a_values = (Fraction(9, 10), Fraction(6, 10))
    b_values = (Fraction(8, 10), Fraction(5, 10))
    a_incompatible = weighted_rate(a_values, (Fraction(1, 10), Fraction(9, 10)))
    b_incompatible = weighted_rate(b_values, (Fraction(9, 10), Fraction(1, 10)))
    judge = judge_order_consistency(("A", "B", "tie", "A"), ("A", "B", "tie", "B"))
    pareto = pareto_relation(
        (Fraction(8, 10), -40, -40, -2), (Fraction(8, 10), -24, -52, -5)
    )
    swarm = swarm_accounting(32, 32, 36, 18)
    return {
        "pass_at_k": _result(
            pass_at_k(4, 1, 2) == Fraction(1, 2)
            and [pass_at_k(6, 2, k) for k in (1, 2, 4)]
            == [Fraction(1, 3), Fraction(3, 5), Fraction(14, 15)],
            exercise=["1/3", "3/5", "14/15"],
        ),
        "paired_effect": _result(
            paired == 0 and discordant == [2, 3],
            effect=str(paired),
            discordant=discordant,
        ),
        "disposition_denominators": _result(
            semantic == Fraction(7, 9) and dispatch == Fraction(7, 12),
            semantic=str(semantic),
            dispatch=str(dispatch),
            counts=counts,
        ),
        "aggregation_reversal": _result(
            a_incompatible == Fraction(63, 100) and b_incompatible == Fraction(77, 100),
            a=str(a_incompatible),
            b=str(b_incompatible),
        ),
        "judge_order": _result(judge == Fraction(3, 4), consistency=str(judge)),
        "pareto": _result(pareto == "incomparable", relation=pareto),
        "swarm_accounting": _result(
            swarm
            == {
                "critical_step_speedup": Fraction(16, 9),
                "work_factor": Fraction(9, 8),
            },
            speedup=str(swarm["critical_step_speedup"]),
            work_factor=str(swarm["work_factor"]),
        ),
    }


def _claim_gate_checks() -> dict[str, dict[str, object]]:
    candidate_claim = {
        "system": "S_M",
        "population": "studio",
        "protocol": "P1",
        "metric": "success",
        "kind": "ranking",
    }
    evidence = {**candidate_claim, "status": "checked"}
    admitted, reasons = claim_admissible(candidate_claim, evidence, True, False)
    rejected, rejection_reasons = claim_admissible(
        {**candidate_claim, "kind": "mechanism"},
        {**evidence, "status": "pending"},
        True,
        False,
    )
    return {
        "dossier_claim_gate": _result(
            admitted
            and not rejected
            and rejection_reasons == ["missing_intervention", "evidence_pending"],
            admitted_reasons=reasons,
            rejected_reasons=rejection_reasons,
        ),
    }


def all_checks() -> dict[str, dict[str, object]]:
    """Run the ten exact Chapter 30 fixture checks."""
    return {
        **_record_protocol_checks(),
        **_metric_checks(),
        **_claim_gate_checks(),
    }
