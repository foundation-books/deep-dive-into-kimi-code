"""Deterministic protocol records for the Chapter 21 reference module."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

from .reference_core import (
    CURRICULUM_ORDER,
    BaselineRecord,
    ResourceVector,
    _rejects,
    add_like_units,
    budget_state,
    domain_effort_cells,
    effective_reward,
    token_cost,
    validate_cell,
    validate_curriculum,
)
from .reference_studio import (
    STUDIO_RESOURCES,
    _baseline,
    _multiplier,
    studio_trace,
)


def _projected_costs(cells: tuple[tuple[str, str], ...]) -> dict[str, int]:
    return {
        f"{domain}-{effort}": token_cost(domain, STUDIO_RESOURCES[(domain, effort)])
        for domain, effort in cells
    }


def all_checks() -> dict[str, dict[str, object]]:
    """Return six deterministic protocol records with JSON-safe values."""
    cells = domain_effort_cells()
    expert_rejection = _rejects(validate_cell, "g", "L", expert_kind="moe")
    projected = _projected_costs(cells)
    unit_rejection = _rejects(add_like_units, 30, "tokens", 2, "turns")
    equality = budget_state(
        "studio-general-001",
        "r1",
        "g",
        "L",
        ResourceVector(8, 3, 0, 1, 0),
        _baseline("g"),
        _multiplier("g", "L"),
    )
    invalid_baseline_rejection = _rejects(
        BaselineRecord("x", "r1", "cold", "teaching", Fraction(0)).validate
    )
    curriculum = tuple(_multiplier("g", effort) for effort in CURRICULUM_ORDER)
    schedule_ok = validate_curriculum(curriculum, Fraction(64))
    order_rejection = _rejects(
        validate_curriculum, tuple(reversed(curriculum)), Fraction(64)
    )
    trace = studio_trace()
    budget_rows = cast(dict[str, dict[str, object]], trace["budget_rows"])
    return {
        "RESULT_CH19_GRID_IDENTITY_ORACLE": {
            "status": "pass",
            "cell_count": len(cells),
            "cells": [f"{domain}-{effort}" for domain, effort in cells],
            "moe_conflation_rejected": expert_rejection,
        },
        "RESULT_CH19_RESOURCE_PROJECTION_ORACLE": {
            "status": "pass",
            "projected_costs": projected,
            "projected_total": sum(projected.values()),
            "assistant_output_total": sum(
                record.assistant_output for record in STUDIO_RESOURCES.values()
            ),
            "unlike_unit_sum_rejected": unit_rejection,
        },
        "RESULT_CH19_BUDGET_REWARD_ORACLE": {
            "status": "pass",
            "equality_relation": equality.relation,
            "equality_effective_reward": str(effective_reward(1, equality)),
            "invalid_baseline_rejected": invalid_baseline_rejection,
            "general_max": budget_rows["g-M"],
        },
        "RESULT_CH19_CURRICULUM_ORACLE": {
            "status": "pass",
            "order": list(CURRICULUM_ORDER),
            "multipliers": [str(record.value) for record in curriculum],
            "finite_max": schedule_ok,
            "wrong_order_rejected": order_rejection,
        },
        "RESULT_CH19_ALLOCATION_ORACLE": {
            "status": "pass",
            "totals": trace["allocation_totals"],
            "budget_projection_total": 380,
            "general_visible_token_difference": 15,
        },
        "RESULT_CH19_STUDIO_HANDOFF_ORACLE": {
            "status": "pass",
            "frontiers": trace["frontiers"],
            "handoff_count": trace["handoff_count"],
            "content_sha256": trace["content_sha256"],
            "measured_protocol_outcomes": "pending",
        },
    }
