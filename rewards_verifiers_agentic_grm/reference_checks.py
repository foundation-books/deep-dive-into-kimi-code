"""Seven deterministic protocol records for the Chapter 22 companion."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from .reference_comparison import ComparisonRecord, compare_candidates, win_fraction
from .reference_core import (
    DISPOSITIONS,
    RewardLedger,
    Rubric,
    ScorepadCell,
    VerifierResult,
    compose_reward,
    exact_verifier,
    hard_gate,
    mask_iou,
    missing,
    observed,
    public_hidden_gap,
    relation_verifier,
    tolerance_verifier,
    verbosity_event,
)
from .reference_studio import (
    center_record,
    centered_studio_group,
    studio_handoffs,
    studio_ledgers,
)


def _studio_relations() -> dict[str, str]:
    red_a = center_record("A", "red", (50, 40))
    blue_a = center_record("A", "blue", (50, 100))
    red_d = center_record("D", "red", (50, 40))
    blue_d = center_record("D", "blue", (50, 100))
    return {
        "A": relation_verifier("below", red_a, blue_a).disposition,
        "C": "not_evaluated",  # Evidence admission stops before the task call.
        "D": relation_verifier("above", red_d, blue_d).disposition,
    }


def _tournament_records() -> tuple[ComparisonRecord, ...]:
    return (
        compare_candidates("A", "B", 8, 18, lambda: "A"),
        compare_candidates("B", "A", 18, 8, lambda: "A"),
        compare_candidates("X", "Y", 8, 8, lambda: "X"),
        compare_candidates("Y", "Z", 8, 8, lambda: "Y"),
        compare_candidates("Z", "X", 8, 8, lambda: "Z"),
        compare_candidates("A", "X", 8, 8, lambda: "abstain"),
        compare_candidates("A", "Y", 8, 8, lambda: "tie"),
    )


def _studio_rubric() -> Rubric:
    criteria = ("correctness", "grounding", "clarity")
    return Rubric(
        criteria,
        {name: (Fraction(0), Fraction(1)) for name in criteria},
        ("fabricated_required_evidence",),
        {
            "correctness": ("relation-witness",),
            "grounding": ("red-event", "blue-event"),
            "clarity": ("artifact-span",),
        },
        "chapter-created",
        "studio-rubric-v1",
    )


def _scorepad_ok() -> bool:
    rubric = _studio_rubric()
    cells = (
        ScorepadCell(
            "A_ctl", "correctness", Fraction(1), ("relation-witness",), "observed"
        ),
        ScorepadCell(
            "A_ctl",
            "grounding",
            Fraction(1),
            ("red-event", "blue-event"),
            "observed",
        ),
        ScorepadCell(
            "B_ctl", "clarity", Fraction(3, 4), ("artifact-span",), "observed"
        ),
    )
    return rubric.validate() and all(cell.validate(rubric) for cell in cells)


def _studio_payload() -> dict[str, object]:
    handoffs = studio_handoffs()
    payload = {
        name: {
            "gate": handoff.ledger.gate,
            "scalar": None if handoff.scalar is None else str(handoff.scalar),
            "final_status": handoff.final_status,
            "evidence_disposition": handoff.evidence_disposition,
            "scalar_purpose": handoff.scalar_purpose,
            "training_eligible": handoff.training_eligible,
            "slot": handoff.slot_id,
            "group_action": handoff.group_action,
        }
        for name, handoff in handoffs.items()
        if handoff.validate()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "candidates": payload,
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "handoff_count": len(payload),
    }


def _disposition_check() -> dict[str, object]:
    pass_result, fail_result = (
        VerifierResult("pass", "ok", "v1"),
        VerifierResult("fail", "false", "v1"),
    )
    invalid_result = VerifierResult("invalid", "malformed", "v1")
    unresolved_result = VerifierResult("indeterminate", "timeout", "v1")
    return {
        "status": "pass",
        "all_dispositions": sorted(DISPOSITIONS),
        "all_pass_gate": hard_gate((pass_result, pass_result)),
        "fail_gate": hard_gate((pass_result, fail_result)),
        "invalid_precedence_gate": hard_gate((unresolved_result, invalid_result)),
        "indeterminate_gate": hard_gate((pass_result, unresolved_result)),
    }


def _domain_check() -> dict[str, object]:
    exact = exact_verifier(37 * 14, 518)
    tolerance = tolerance_verifier(
        0.3000004,
        0.3,
        absolute=0.000001,
        relative=0,
        unit="ratio",
        expected_unit="ratio",
    )
    unit_invalid = tolerance_verifier(
        1,
        1,
        absolute=0,
        relative=0,
        unit="meters",
        expected_unit="seconds",
    )
    iou = mask_iou({(0, 0), (0, 1)}, {(0, 1), (1, 1)})
    return {
        "status": "pass",
        "exact_product": exact.disposition,
        "tolerance_boundary": tolerance.disposition,
        "unit_mismatch": unit_invalid.disposition,
        "mask_iou": str(iou),
        "studio_relations": _studio_relations(),
    }


def _composition_check() -> dict[str, object]:
    ledgers = studio_ledgers()
    missing_ledger = RewardLedger(
        1,
        observed(1),
        observed(0),
        missing(),
        observed(0),
        observed(0),
        observed(0),
        "studio-profile-v1",
    )
    return {
        "status": "pass",
        "scalars": {
            name: str(compose_reward(ledger)) for name, ledger in ledgers.items()
        },
        "invalid_with_unbounded_soft_score": str(compose_reward(ledgers["C"])),
        "required_missing_propagates": compose_reward(missing_ledger) is None,
    }


def _public_hidden_check() -> dict[str, object]:
    public, hidden, studio_gap = public_hidden_gap(1, 1, 0, 1)
    _, _, coding_gap = public_hidden_gap(2, 2, 1, 2)
    return {
        "status": "pass",
        "studio_public_rate": str(public),
        "studio_hidden_rate": str(hidden),
        "studio_proxy_gap": str(studio_gap),
        "coding_proxy_gap": str(coding_gap),
    }


def _tournament_check() -> dict[str, object]:
    tournament = _tournament_records()
    abstention_only = (ComparisonRecord("A", "B", ("A", "B"), "abstain", "v1", "v1"),)
    return {
        "status": "pass",
        "studio_A_win_fraction": str(win_fraction(tournament, "A")),
        "cycle_win_fractions": {
            name: str(win_fraction(tournament, name)) for name in ("X", "Y", "Z")
        },
        "abstention_only_is_missing": win_fraction(abstention_only, "A") is None,
        "order_records_retained": [
            list(record.displayed_order) for record in tournament[:2]
        ],
    }


def _rubric_verbosity_check() -> dict[str, object]:
    return {
        "status": "pass",
        "scorepad_certificate_valid": _scorepad_ok(),
        "threshold": 16,
        "events": {
            str(length): verbosity_event(length, 8, Fraction(2))
            for length in (15, 16, 18)
        },
        "both_over_behavior": "not publicly specified",
    }


def _studio_handoff_check() -> dict[str, object]:
    mean, coefficients = centered_studio_group(tuple(studio_handoffs().values()))
    return {
        "status": "pass",
        **_studio_payload(),
        "inclusive_mean": str(mean),
        "centered_coefficients": [str(value) for value in coefficients],
        "measured_protocol_outcomes": "pending",
    }


def all_checks() -> dict[str, dict[str, object]]:
    """Return seven JSON-safe exact fixture records."""
    return {
        "RESULT_CH20_DISPOSITION_SCHEMA_ORACLE": _disposition_check(),
        "RESULT_CH20_DOMAIN_VERIFIER_ORACLE": _domain_check(),
        "RESULT_CH20_HARD_FIRST_COMPOSITION_ORACLE": _composition_check(),
        "RESULT_CH20_PUBLIC_HIDDEN_HACK_ORACLE": _public_hidden_check(),
        "RESULT_CH20_TOURNAMENT_ORACLE": _tournament_check(),
        "RESULT_CH20_RUBRIC_VERBOSITY_ORACLE": _rubric_verbosity_check(),
        "RESULT_CH20_STUDIO_HANDOFF_ORACLE": _studio_handoff_check(),
    }
