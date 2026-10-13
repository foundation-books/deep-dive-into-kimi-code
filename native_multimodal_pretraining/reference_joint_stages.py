"""Deterministic joint-from-start studio stages used by printed Chapter 16."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping

from .reference_stages import stage_budget


_TEXT_CAPTION_GROUNDING: Mapping[str, Fraction] = {
    "text": Fraction(1, 2),
    "caption": Fraction(1, 4),
    "grounding": Fraction(1, 4),
}
_TEXT_GROUNDING_REPAIR: Mapping[str, Fraction] = {
    "text": Fraction(1, 2),
    "grounding": Fraction(1, 4),
    "code_render": Fraction(1, 4),
}
_TEXT_REPAIR_VIDEO: Mapping[str, Fraction] = {
    "text": Fraction(1, 2),
    "code_render": Fraction(1, 4),
    "video": Fraction(1, 4),
}
_OPTIMIZER = {"method": "toy_sgd", "learning_rate": "1/1000", "precision": "fp32"}
_MASK = {"rule": "canonical_family_targets", "reduction": "pooled_active_targets"}
_TRAINABLE = ("vision_encoder", "projector", "backbone")
_SOURCE = "studio_semantic_source_reuse_cap_40"


def _stage_tuple(
    full_law: dict[str, object],
    temporal_policy: str,
    context_cap: int,
    budgets: dict[str, int],
    gates: Mapping[str, object],
) -> dict[str, object]:
    return {
        "Pi_s": full_law,
        "R_s": {"image_grid": "low_8x8", "visual_position_cap": 16},
        "F_s": temporal_policy,
        "L_s": {
            "context_cap": context_cap,
            "packing": "one_record_with_explicit_boundary",
            "step_token_capacity": context_cap,
            "position_map": "consecutive_from_zero",
            "attention": "causal",
        },
        "M_s": _MASK,
        "Theta_s": _TRAINABLE,
        "Omega_s": _OPTIMIZER,
        "B_s": {
            "intended": budgets,
            "frozen_trace": budgets,
            "observed_training": None,
        },
        "G_s": dict(gates),
    }


def _stage(
    name: str,
    mixture: Mapping[str, Fraction],
    records: int,
    context_cap: int,
    *,
    source_policy: str,
    temporal_policy: str,
    gates: Mapping[str, object],
) -> dict[str, object]:
    if records <= 0 or context_cap < 64:
        raise ValueError("record and context budgets must be positive")
    counts = {family: probability * records for family, probability in mixture.items()}
    if any(count.denominator != 1 for count in counts.values()):
        raise ValueError("the frozen trace requires integral family counts")
    family_counts = {family: count.numerator for family, count in counts.items()}
    if set(gates) != {"entry", "integrity", "stability", "retention", "exit"}:
        raise ValueError("all five gate classes must be declared")
    if name == "initial_joint" and gates["retention"] is not None:
        raise ValueError("initial retention must be not applicable")
    if name != "initial_joint" and gates["retention"] is None:
        raise ValueError("later stages need a retention predicate")

    media_choices = {
        family: ("four_frame_low" if family == "video" else "low_grid")
        for family in mixture
        if family != "text"
    }
    full_law = {
        "family_probabilities": {
            family: str(probability) for family, probability in mixture.items()
        },
        "source_policy": source_policy,
        "within_family_media": media_choices,
        "accepted_trace_only": True,
    }
    budgets = stage_budget(mixture, records)
    tuple_fields = _stage_tuple(full_law, temporal_policy, context_cap, budgets, gates)
    if any(
        family != "text" and media_choices[family] not in {"low_grid", "four_frame_low"}
        for family in mixture
    ):
        raise ValueError("the realized media must be admitted by the stage")
    return {
        "name": name,
        "family_counts": family_counts,
        "media_choices": media_choices,
        "budgets": budgets,
        "tuple_fields": tuple_fields,
        "observed_training_budgets": None,
        "training_gate_verdict": None,
        "retention_status": "not_applicable"
        if name == "initial_joint"
        else "unmeasured",
    }


def _add_transition_partitions(stages: list[dict[str, object]]) -> list[str]:
    first = stages[0]["tuple_fields"]
    if not isinstance(first, dict):
        raise AssertionError("stage tuple is not a mapping")
    field_names = set(first)
    for index, stage in enumerate(stages):
        current = stage["tuple_fields"]
        if not isinstance(current, dict):
            raise AssertionError("stage tuple is not a mapping")
        previous = stages[index - 1]["tuple_fields"] if index else None
        if previous is None:
            stage["changed_fields"] = sorted(field_names)
            stage["fixed_fields"] = []
        else:
            if not isinstance(previous, dict):
                raise AssertionError("previous stage tuple is not a mapping")
            stage["changed_fields"] = sorted(
                field for field in field_names if current[field] != previous[field]
            )
            stage["fixed_fields"] = sorted(
                field for field in field_names if current[field] == previous[field]
            )
    return sorted(field_names)


def joint_stage_fixture() -> dict[str, object]:
    """Return the four printed stages and exact changed/fixed field sets."""

    stages = [
        _stage(
            "initial_joint",
            _TEXT_CAPTION_GROUNDING,
            20,
            64,
            source_policy=_SOURCE,
            temporal_policy="image_only",
            gates={
                "entry": "initial_registry_and_baseline_ready",
                "integrity": "valid_records_and_conserved_ledgers",
                "stability": "finite_groupwise_updates",
                "retention": None,
                "exit": "initial_grounding_acquisition",
            },
        ),
        _stage(
            "joint_acquisition",
            _TEXT_CAPTION_GROUNDING,
            40,
            192,
            source_policy=_SOURCE,
            temporal_policy="image_only",
            gates={
                "entry": "initial_joint_handoff_and_text_baseline",
                "integrity": "valid_records_and_conserved_ledgers",
                "stability": "finite_groupwise_updates",
                "retention": "text_drop_at_most_2_percentage_points",
                "exit": "grounding_gain_under_declared_uncertainty_rule",
            },
        ),
        _stage(
            "quality_cooldown",
            _TEXT_GROUNDING_REPAIR,
            40,
            192,
            source_policy="quality_filtered_" + _SOURCE,
            temporal_policy="image_only",
            gates={
                "entry": "joint_acquisition_handoff",
                "integrity": "valid_records_quality_floor_and_conserved_ledgers",
                "stability": "finite_groupwise_updates",
                "retention": "protected_text_code_and_grounding",
                "exit": "cooldown_quality_gate",
            },
        ),
        _stage(
            "long_cap_continuation",
            _TEXT_REPAIR_VIDEO,
            40,
            1024,
            source_policy="temporal_validated_" + _SOURCE,
            temporal_policy="ordered_four_frame_group",
            gates={
                "entry": "quality_cooldown_handoff",
                "integrity": "valid_records_temporal_support_and_conserved_ledgers",
                "stability": "finite_groupwise_updates",
                "retention": "protected_earlier_capabilities",
                "exit": "continuation_exit_gate",
            },
        ),
    ]
    return {"stages": stages, "field_names": _add_transition_partitions(stages)}


def two_loss_reductions(
    text_mean: Fraction, visual_mean: Fraction
) -> dict[str, Fraction]:
    """Compare equal-record and pooled-target losses for 64 and 32 targets."""

    return {
        "equal_record_means": (text_mean + visual_mean) / 2,
        "pooled_active_targets": (64 * text_mean + 32 * visual_mean) / 96,
    }


def accepted_family_shares() -> dict[str, Fraction]:
    """Normalize two equal attempted masses after 1 and 1/2 acceptance."""

    accepted = {"first": Fraction(1, 2), "second": Fraction(1, 4)}
    total = sum(accepted.values(), Fraction())
    return {family: mass / total for family, mass in accepted.items()}


__all__ = ["accepted_family_shares", "joint_stage_fixture", "two_loss_reductions"]
