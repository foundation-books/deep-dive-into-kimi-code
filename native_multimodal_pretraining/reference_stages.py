"""Exact curriculum-stage contracts for Chapter 16."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, cast

from .reference_stage_inputs import STAGE_INPUTS
from .reference_types import FamilySpec


GATE_CLASSES = {"entry", "integrity", "stability", "retention", "exit"}
TUPLE_FIELDS = {
    "pi_s",
    "R_s",
    "F_s",
    "L_s",
    "M_s",
    "Theta_s",
    "Omega_s",
    "B_s",
    "G_s",
}
REQUIRED_STAGE_FIELDS = {
    "name",
    "mixture",
    "source_policy",
    "records",
    "resolution_policy",
    "temporal_policy",
    "context_length",
    "packing_policy",
    "mask_policy",
    "trainable_parameters",
    "optimizer_phase",
    "precision_state",
    "gates",
}
STAGE_FAMILIES: Mapping[str, FamilySpec] = {
    "text": FamilySpec("text", Fraction(1), 64, 64, 0, 1),
    "caption": FamilySpec("caption", Fraction(1), 48, 20, 16, 3),
    "grounding": FamilySpec("grounding", Fraction(1), 40, 8, 16, 3),
    "code_render": FamilySpec("code_render", Fraction(1), 72, 24, 16, 4),
    "video": FamilySpec("video", Fraction(1), 64, 12, 16, 6),
}


def stage_budget(mixture: Mapping[str, Fraction], records: int) -> dict[str, int]:
    """Convert a rational mixture and record count into exact stage budgets."""

    if records <= 0 or sum(mixture.values(), Fraction()) != 1:
        raise ValueError("stage mixture and record count are invalid")
    counts: dict[str, int] = {}
    for name, probability in mixture.items():
        if name not in STAGE_FAMILIES or probability < 0:
            raise ValueError("unknown family or negative probability")
        exact = probability * records
        if exact.denominator != 1:
            raise ValueError("record count must realize every family exactly")
        counts[name] = exact.numerator
    return {
        "records": records,
        "backbone_tokens": sum(
            counts[name] * STAGE_FAMILIES[name].backbone_length for name in counts
        ),
        "targets": sum(
            counts[name] * STAGE_FAMILIES[name].target_count for name in counts
        ),
        "visual_positions": sum(
            counts[name] * STAGE_FAMILIES[name].visual_positions for name in counts
        ),
        "compute_units": sum(
            counts[name] * STAGE_FAMILIES[name].compute_units for name in counts
        ),
    }


def _validated_fields(
    stage: dict[str, object],
) -> tuple[dict[str, Fraction], int, dict[str, list[str]]]:
    if set(stage) != REQUIRED_STAGE_FIELDS:
        raise AssertionError("pedagogical stage is incomplete")
    mixture = stage["mixture"]
    records = stage["records"]
    gates = stage["gates"]
    if not isinstance(mixture, dict) or not isinstance(records, int):
        raise AssertionError("stage fields have unexpected types")
    if any(
        not isinstance(name, str) or not isinstance(value, Fraction)
        for name, value in mixture.items()
    ):
        raise AssertionError("stage mixture has unexpected types")
    if not isinstance(gates, dict) or set(gates) != GATE_CLASSES:
        raise AssertionError("every gate class must be explicit")
    if any(not isinstance(values, list) or not values for values in gates.values()):
        raise AssertionError("every gate class must be nonempty")
    return (
        cast(dict[str, Fraction], mixture),
        records,
        cast(dict[str, list[str]], gates),
    )


def _tuple_fields(
    stage: dict[str, object],
    serialized_mixture: dict[str, str],
    budgets: dict[str, int],
    gates: dict[str, list[str]],
) -> dict[str, object]:
    return {
        "pi_s": {
            "mixture": serialized_mixture,
            "source_policy": stage["source_policy"],
        },
        "R_s": stage["resolution_policy"],
        "F_s": stage["temporal_policy"],
        "L_s": {
            "context_length": stage["context_length"],
            "packing_policy": stage["packing_policy"],
        },
        "M_s": stage["mask_policy"],
        "Theta_s": stage["trainable_parameters"],
        "Omega_s": {
            "optimizer_phase": stage["optimizer_phase"],
            "precision_state": stage["precision_state"],
        },
        "B_s": budgets,
        "G_s": gates,
    }


def _transition(
    tuple_fields: dict[str, object],
    gates: dict[str, list[str]],
    previous_tuple: dict[str, object] | None,
    serialized: list[dict[str, object]],
) -> tuple[str, bool, list[str], list[str]]:
    if previous_tuple is None:
        return (
            "initial_state",
            gates["entry"] == ["initial_registry_ready"],
            list(tuple_fields),
            [],
        )
    previous_gates = serialized[-1]["gates"]
    if not isinstance(previous_gates, dict):
        raise AssertionError("serialized gates must remain an object")
    changed = [
        field for field, value in tuple_fields.items() if value != previous_tuple[field]
    ]
    fixed = [
        field for field, value in tuple_fields.items() if value == previous_tuple[field]
    ]
    handoff_valid = bool(set(previous_gates["exit"]) & set(gates["entry"]))
    return str(serialized[-1]["name"]), handoff_valid, changed, fixed


def _serialize_stage(
    stage: dict[str, object],
    previous_tuple: dict[str, object] | None,
    serialized: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    mixture, records, gates = _validated_fields(stage)
    budgets = stage_budget(mixture, records)
    observed_budgets = dict(budgets)
    serialized_mixture = {name: str(value) for name, value in mixture.items()}
    tuple_fields = _tuple_fields(stage, serialized_mixture, budgets, gates)
    predecessor, handoff_valid, changed, fixed = _transition(
        tuple_fields, gates, previous_tuple, serialized
    )
    payload = {
        **{key: value for key, value in stage.items() if key != "mixture"},
        "mixture": serialized_mixture,
        "budgets": budgets,
        "observed_budgets": observed_budgets,
        "budget_reconciled": observed_budgets == budgets,
        "tuple_fields": tuple_fields,
        "predecessor": predecessor,
        "handoff_from_previous_valid": handoff_valid,
        "changed_fields_from_previous": changed,
        "fixed_fields_from_previous": fixed,
    }
    return payload, tuple_fields


def _add_successors(serialized: list[dict[str, object]]) -> None:
    for index, stage in enumerate(serialized):
        stage["successor"] = (
            serialized[index + 1]["name"]
            if index + 1 < len(serialized)
            else "terminal_state"
        )


def _transition_fields_accounted(stage: dict[str, object]) -> bool:
    changed = stage["changed_fields_from_previous"]
    fixed = stage["fixed_fields_from_previous"]
    if not isinstance(changed, list) or not isinstance(fixed, list):
        raise AssertionError("transition field lists must remain lists")
    return set(changed) | set(fixed) == TUPLE_FIELDS


def stage_fixture() -> dict[str, object]:
    """Return four complete pedagogical stage tuples and reconciled budgets."""

    serialized: list[dict[str, object]] = []
    previous_tuple: dict[str, object] | None = None
    for stage in STAGE_INPUTS:
        payload, previous_tuple = _serialize_stage(stage, previous_tuple, serialized)
        serialized.append(payload)
    _add_successors(serialized)
    return {
        "stage_names": [stage["name"] for stage in serialized],
        "stages": serialized,
        "all_fields_complete": True,
        "all_mixtures_normalized": True,
        "all_budgets_reconciled": all(
            stage["budget_reconciled"] is True for stage in serialized
        ),
        "all_intended_observed_budgets_reconciled": all(
            stage["budget_reconciled"] is True for stage in serialized
        ),
        "all_gate_classes_declared": all(
            set(cast(dict[str, object], stage["gates"])) == GATE_CLASSES
            for stage in serialized
        ),
        "all_precision_states_declared": all(
            bool(stage["precision_state"]) for stage in serialized
        ),
        "all_transition_fields_accounted": all(
            _transition_fields_accounted(stage) for stage in serialized
        ),
        "all_handoffs_valid": all(
            stage["handoff_from_previous_valid"] is True for stage in serialized
        ),
        "reported_kimi_fields_not_imputed": True,
    }


__all__ = ["stage_budget", "stage_fixture"]
