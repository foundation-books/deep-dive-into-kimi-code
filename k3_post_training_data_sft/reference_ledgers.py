"""Mixture ledgers and sequential-error fixtures for Chapter 17."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def normalized_shares(values: Mapping[str, int]) -> dict[str, str]:
    """Normalize a nonnegative ledger and reject an empty denominator."""

    if any(value < 0 for value in values.values()):
        raise ValueError("ledger values must be nonnegative")
    denominator = sum(values.values())
    if denominator <= 0:
        raise ValueError("ledger denominator must be positive")
    return {
        name: _fraction_string(Fraction(value, denominator))
        for name, value in values.items()
    }


def mixture_fixture() -> dict[str, object]:
    """Reproduce all six ledgers for the two-record example."""

    empty_denominator_rejected = False
    try:
        normalized_shares({"direct": 0, "agentic": 0})
    except ValueError:
        empty_denominator_rejected = True
    return {
        "record_share": normalized_shares({"direct": 1, "agentic": 1}),
        "serialized_token_share": normalized_shares({"direct": 20, "agentic": 200}),
        "supervised_token_share": normalized_shares({"direct": 10, "agentic": 50}),
        "action_share": {"direct": "0/1", "agentic": "1/1"},
        "source_lineage": {
            "unique_sources": 1,
            "source_reuse": {"studio-seed-1": 2},
            "family_shares_defined": False,
        },
        "approximate_compute_share": normalized_shares({"direct": 20, "agentic": 320}),
        "raw_totals": {
            "records": 2,
            "serialized_tokens": 220,
            "supervised_tokens": 60,
            "actions": 8,
            "unique_sources": 1,
            "approximate_compute_units": 340,
        },
        "empty_denominator_rejected": empty_denominator_rejected,
    }


def sequential_error_fixture(
    horizon: int = 25, epsilon: float = 0.01
) -> dict[str, object]:
    """Return the chapter's qualified illustrative bound terms."""

    if horizon <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("invalid horizon or epsilon")
    return {
        "horizon": horizon,
        "epsilon": epsilon,
        "quadratic_term": horizon * horizon * epsilon,
        "linear_term": horizon * epsilon,
        "terms_are_probabilities": False,
        "claim_boundary": (
            "illustrative cost-bound arithmetic under stated imitation-learning "
            "assumptions"
        ),
        "recovery_trajectories": [
            {
                "record_id": "studio-recovery-schema",
                "learner_induced_failure": "malformed locate_shapes image_id",
                "retained_feedback": "typed CALL_SCHEMA error",
                "recovery": "reissue c1 with studio-layout-001, then execute c2",
                "terminal_claim": "below with cited c1 and c2 evidence",
            },
            {
                "record_id": "studio-recovery-stale-state",
                "learner_induced_failure": "pre-crop boxes supplied after crop",
                "retained_feedback": "STALE_STATE_ARGUMENTS rejection",
                "recovery": (
                    "re-locate shapes in the cropped state, then compare current boxes"
                ),
                "terminal_claim": "below only after the recovered observation",
            },
        ],
    }


__all__ = ["mixture_fixture", "normalized_shares", "sequential_error_fixture"]
