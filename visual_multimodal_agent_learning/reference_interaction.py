"""Active-perception, interaction, and handoff fixtures for Chapter 25."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import cast


HANDOFF_INVARIANTS = (
    "versioned-identity",
    "observation-lineage",
    "coordinate-convention",
    "actionable-grounding",
    "disjoint-action-semantics",
    "trace-legality-terminal-independence",
    "vision-admission",
    "zero-vision-scope",
    "curriculum-visibility",
    "outcome-path-separation",
    "active-perception-accountability",
    "consistency-before-credit",
    "set-valued-attribution",
    "fresh-state-binding",
    "matched-transfer-attribution",
    "retention-visibility",
    "diagnostic-scope",
    "evidence-terminal-scope",
)
EVIDENCE_STATUSES = (
    "measured",
    "reported",
    "derived",
    "illustrative",
    "pending",
    "unknown",
    "not-applicable",
)


def active_perception_check() -> dict[str, object]:
    """Check child lineage and separated information costs."""

    calls = (
        {
            "source": "obs-root-v11",
            "tool": "crop-v1",
            "arguments": [10, 20, 60, 120],
            "output": "obs-crop-v11",
            "derivation": "root-to-crop-v1",
            "cost": 1,
            "state_version": 11,
        },
        {
            "source": "obs-crop-v11",
            "tool": "ocr-v2",
            "arguments": ["normalize-upper"],
            "output": "ocr-blue-v11",
            "derivation": "nongeometric-ocr-v2",
            "cost": 2,
            "state_version": 11,
        },
    )
    required = {
        "source",
        "tool",
        "arguments",
        "output",
        "derivation",
        "cost",
        "state_version",
    }
    version_bound = all(call["state_version"] == 11 for call in calls)
    derivations_declared = all(bool(call["derivation"]) for call in calls)
    complete = (
        all(set(call) == required for call in calls)
        and version_bound
        and derivations_declared
    )
    cost = sum(cast(int, call["cost"]) for call in calls)
    broken = {key: value for key, value in calls[0].items() if key != "tool"}
    broken_rejected = set(broken) != required
    passed = complete and cost == 3 and broken_rejected
    return {
        "status": "pass" if passed else "fail",
        "call_count": len(calls),
        "total_information_cost": cost,
        "within_budget": cost <= 3,
        "complete_lineage": complete,
        "state_version_bound": version_bound,
        "missing_tool_version_rejected": broken_rejected,
    }


def gui_staleness_check() -> dict[str, object]:
    """Check state-version binding and a fresh action-effect witness."""

    stale_action = {
        "observation_version": 11,
        "current_version": 12,
        "point": [90, 40],
        "explicit_revalidation": False,
    }
    stale_rejected = (
        stale_action["observation_version"] != stale_action["current_version"]
        and not stale_action["explicit_revalidation"]
    )
    fresh_witness = {
        "observation": "gui-v12",
        "grounding": [90, 80],
        "transform": "identity-edge-v1",
        "action": [90, 80],
        "before_version": 12,
        "after_version": 13,
        "effect_predicate": True,
    }
    witness_complete = set(fresh_witness) == {
        "observation",
        "grounding",
        "transform",
        "action",
        "before_version",
        "after_version",
        "effect_predicate",
    }
    passed = stale_rejected and witness_complete and fresh_witness["effect_predicate"]
    return {
        "status": "pass" if passed else "fail",
        "stale_action_rejected": stale_rejected,
        "fresh_action_point": fresh_witness["action"],
        "fresh_witness_complete": witness_complete,
        "effect_verified": fresh_witness["effect_predicate"],
    }


def diagnostic_certificate_check() -> dict[str, object]:
    """Apply only candidate reductions licensed by the registered replays."""

    initial = {
        "capture",
        "perception",
        "grounding",
        "transform",
        "planning",
        "execution",
        "transition",
        "verification",
    }
    outcomes = {
        "oracle_grounding": "success",
        "oracle_transform": "failure",
        "fresh_view_with_old_grounding": "failure",
    }
    # Under the declared isolation assumptions, oracle-grounding success
    # localizes the fault to the representation/grounding boundary. Neither
    # remaining failure separates capture from perception from grounding.
    licensed = initial & {"capture", "perception", "grounding"}
    passed = outcomes["oracle_grounding"] == "success" and licensed == {
        "capture",
        "perception",
        "grounding",
    }
    return {
        "status": "pass" if passed else "fail",
        "initial_candidate_count": len(initial),
        "replay_outcomes": outcomes,
        "remaining_candidates": sorted(licensed),
        "unique_cause_claimed": False,
        "residual_uncertainty_retained": True,
    }


def handoff_check() -> dict[str, object]:
    """Check the complete evidence-qualified Chapter 25 packet."""

    record = {
        "task": "studio-ops-001",
        "schema": "chapter-23-handoff-v1",
        "version": 1,
        "invariants": list(HANDOFF_INVARIANTS),
        "evidence_statuses": list(EVIDENCE_STATUSES),
        "pending_protocols": 8,
        "terminal_truth": "independent-verifier",
    }
    digest = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    passed = (
        len(set(HANDOFF_INVARIANTS)) == 18
        and len(set(EVIDENCE_STATUSES)) == 7
        and record["pending_protocols"] == 8
        and record["terminal_truth"] == "independent-verifier"
    )
    return {
        "status": "pass" if passed else "fail",
        "task": record["task"],
        "invariant_count": len(HANDOFF_INVARIANTS),
        "evidence_status_count": len(EVIDENCE_STATUSES),
        "pending_protocol_count": record["pending_protocols"],
        "record_digest": digest,
    }


__all__ = [
    "active_perception_check",
    "diagnostic_certificate_check",
    "gui_staleness_check",
    "handoff_check",
]
