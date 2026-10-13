"""Canonical four-stage curriculum inputs for Chapter 16."""

from __future__ import annotations

from fractions import Fraction


STAGE_INPUTS: tuple[dict[str, object], ...] = (
    {
        "name": "vision_preparation",
        "mixture": {"caption": Fraction(1)},
        "source_policy": "frozen studio registry with declared cap 20",
        "records": 20,
        "resolution_policy": "low_grid",
        "temporal_policy": "image_only",
        "context_length": 64,
        "packing_policy": "single_record_with_explicit_boundaries",
        "mask_policy": "caption_targets",
        "trainable_parameters": "vision_and_projector",
        "optimizer_phase": "preparation",
        "precision_state": "fixed_reference_precision",
        "gates": {
            "entry": ["initial_registry_ready"],
            "integrity": ["record_integrity", "caption_acquisition"],
            "stability": ["finite_groupwise_updates"],
            "retention": ["frozen_language_state_unchanged"],
            "exit": ["vision_preparation_complete"],
        },
    },
    {
        "name": "joint_acquisition",
        "mixture": {
            "text": Fraction(1, 2),
            "caption": Fraction(1, 4),
            "grounding": Fraction(1, 4),
        },
        "source_policy": "frozen studio registry with text floor and source caps",
        "records": 40,
        "resolution_policy": "low_and_high_grids",
        "temporal_policy": "image_only",
        "context_length": 192,
        "packing_policy": "family_aware_packing_with_media_boundaries",
        "mask_policy": "family_targets",
        "trainable_parameters": "vision_projector_backbone",
        "optimizer_phase": "joint",
        "precision_state": "fixed_reference_precision",
        "gates": {
            "entry": ["vision_preparation_complete"],
            "integrity": ["mixture_conservation", "geometry_integrity"],
            "stability": ["finite_groupwise_updates"],
            "retention": ["text_retention"],
            "exit": ["joint_acquisition_complete"],
        },
    },
    {
        "name": "quality_cooldown",
        "mixture": {
            "text": Fraction(1, 2),
            "grounding": Fraction(1, 4),
            "code_render": Fraction(1, 4),
        },
        "source_policy": "quality floor with text replay and source caps",
        "records": 40,
        "resolution_policy": "quality_weighted_high_grid",
        "temporal_policy": "image_only",
        "context_length": 192,
        "packing_policy": "family_aware_packing_with_media_boundaries",
        "mask_policy": "quality_filtered_family_targets",
        "trainable_parameters": "vision_projector_backbone",
        "optimizer_phase": "cooldown",
        "precision_state": "fixed_reference_precision",
        "gates": {
            "entry": ["joint_acquisition_complete"],
            "integrity": ["geometry_integrity", "quality_floor"],
            "stability": ["finite_groupwise_updates"],
            "retention": ["code_retention"],
            "exit": ["quality_cooldown_complete"],
        },
    },
    {
        "name": "multimodal_long_continuation",
        "mixture": {
            "text": Fraction(1, 2),
            "code_render": Fraction(1, 4),
            "video": Fraction(1, 4),
        },
        "source_policy": "long-form source caps with protected text replay",
        "records": 40,
        "resolution_policy": "budgeted_low_high_grid",
        "temporal_policy": "ordered_four_frame_groups",
        "context_length": 1024,
        "packing_policy": "long_context_dependency_preserving_packing",
        "mask_policy": "causal_family_targets",
        "trainable_parameters": "vision_projector_backbone",
        "optimizer_phase": "long_continuation",
        "precision_state": "fixed_reference_precision",
        "gates": {
            "entry": ["quality_cooldown_complete"],
            "integrity": ["temporal_validity", "dependency_survival"],
            "stability": ["finite_groupwise_updates"],
            "retention": ["earlier_stage_retention"],
            "exit": ["multimodal_long_continuation_complete"],
        },
    },
)


__all__ = ["STAGE_INPUTS"]
