"""Deterministic Chapter 21 studio fixtures and composed trace."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from .reference_core import (
    CELL_KEYS,
    DOMAINS,
    EFFORTS,
    AllocationRow,
    BaselineRecord,
    MultiplierRecord,
    ParetoPoint,
    ResourceVector,
    SpecialistHandoff,
    allocation_totals,
    budget_state,
    domain_effort_cells,
    effective_reward,
    pareto_frontier,
)


BASELINES = {"g": Fraction(12), "a": Fraction(30), "c": Fraction(60)}
TASK_NAMES = {"g": "general", "a": "agent", "c": "code"}
MULTIPLIERS = {"L": Fraction(2, 3), "H": Fraction(4, 3), "M": Fraction(2)}
STUDIO_COSTS = dict(zip(CELL_KEYS, (7, 12, 26, 18, 36, 58, 39, 74, 110), strict=True))
STUDIO_REWARDS = dict(zip(CELL_KEYS, (1, 1, 1, 0, 1, 1, 0, 1, 1), strict=True))
STUDIO_RESOURCES = {
    ("g", "L"): ResourceVector(7, 3, 0, 1, 0, Fraction(1, 5)),
    ("g", "H"): ResourceVector(12, 5, 0, 1, 0, Fraction(2, 5)),
    ("g", "M"): ResourceVector(26, 7, 0, 1, 0, Fraction(9, 10)),
    ("a", "L"): ResourceVector(8, 4, 6, 2, 1, Fraction(1)),
    ("a", "H"): ResourceVector(14, 8, 14, 3, 2, Fraction(17, 10)),
    ("a", "M"): ResourceVector(20, 18, 20, 4, 3, Fraction(5, 2)),
    ("c", "L"): ResourceVector(12, 5, 22, 3, 2, Fraction(3)),
    ("c", "H"): ResourceVector(20, 8, 46, 4, 3, Fraction(6)),
    ("c", "M"): ResourceVector(30, 12, 68, 5, 4, Fraction(9)),
}


def _baseline(domain: str) -> BaselineRecord:
    return BaselineRecord(
        f"studio-{TASK_NAMES[domain]}-001",
        "r1",
        f"cold-{domain}-v1",
        "teaching",
        BASELINES[domain],
    )


def _multiplier(domain: str, effort: str) -> MultiplierRecord:
    return MultiplierRecord(
        domain, effort, f"stage-{effort}", MULTIPLIERS[effort], "fixture"
    )


def _cell_records(
    domain: str, effort: str
) -> tuple[str, dict[str, object], AllocationRow, SpecialistHandoff]:
    resources = STUDIO_RESOURCES[(domain, effort)]
    state = budget_state(
        f"studio-{TASK_NAMES[domain]}-001",
        "r1",
        domain,
        effort,
        resources,
        _baseline(domain),
        _multiplier(domain, effort),
    )
    reward = STUDIO_REWARDS[(domain, effort)]
    key = f"{domain}-{effort}"
    budget_row = {
        "cap": str(state.cap),
        "cost": state.cost,
        "slack": str(state.slack),
        "overrun": str(state.overrun),
        "relation": state.relation,
        "task_reward": reward,
        "effective_reward": str(effective_reward(reward, state)),
    }
    allocation_row = AllocationRow(
        1,
        1,
        reward,
        int(state.relation != "over"),
        resources.thinking,
        resources.assistant_output,
        resources.turns,
        resources.tools,
    )
    handoff = SpecialistHandoff(
        f"policy-{key}-fixture",
        domain,
        effort,
        f"curriculum-{domain}-fixture",
        f"baseline-{domain}-fixture",
        f"reward-{domain}-fixture",
        f"summary-{key}-fixture",
        ("fixture", "deduced", "pending", "unknown"),
        ("K3 numerical schedule", "population capability curve"),
    )
    return key, budget_row, allocation_row, handoff


def _studio_rows() -> tuple[
    dict[str, dict[str, object]], list[AllocationRow], list[SpecialistHandoff]
]:
    budget_rows: dict[str, dict[str, object]] = {}
    allocation_rows: list[AllocationRow] = []
    handoffs: list[SpecialistHandoff] = []
    for domain, effort in domain_effort_cells():
        key, budget_row, allocation_row, handoff = _cell_records(domain, effort)
        budget_rows[key] = budget_row
        allocation_rows.append(allocation_row)
        handoffs.append(handoff)
    return budget_rows, allocation_rows, handoffs


def _studio_frontiers() -> dict[str, tuple[str, ...]]:
    return {
        domain: pareto_frontier(
            tuple(
                ParetoPoint(
                    f"{domain}-{effort}",
                    (Fraction(STUDIO_REWARDS[(domain, effort)]),),
                    (Fraction(STUDIO_COSTS[(domain, effort)]),),
                )
                for effort in EFFORTS
            )
        )
        for domain in DOMAINS
    }


def studio_trace() -> dict[str, object]:
    """Compose all nine rows, exact totals, frontiers, and handoff records."""

    budget_rows, allocation_rows, handoffs = _studio_rows()
    payload = {
        "budget_rows": budget_rows,
        "allocation_totals": allocation_totals(allocation_rows).__dict__,
        "frontiers": _studio_frontiers(),
        "handoff_count": len(
            tuple(handoff for handoff in handoffs if handoff.validate())
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "BASELINES",
    "MULTIPLIERS",
    "STUDIO_COSTS",
    "STUDIO_RESOURCES",
    "STUDIO_REWARDS",
    "TASK_NAMES",
    "studio_trace",
]
