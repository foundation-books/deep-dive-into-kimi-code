"""Dependency-free semantic oracles for Chapter 16.

The facade composes the chapter's typed serialization, geometry, mixture,
resolution, and curriculum ledgers through ordinary package imports. The
fixtures do not train a model, implement a Moonshot AI data
pipeline, or measure model quality or hardware performance.
"""

from __future__ import annotations

from .reference_mixture import (
    mixture_fixture,
    mixture_ledgers,
    source_reuse_ledger,
    two_family_specs,
)
from .reference_resolution import grid_ledger, resolution_fixture
from .reference_serialization import (
    caption_record,
    crop_box,
    media_identity_swap_detected,
    normalize_box,
    reflect_box_horizontally,
    validate_serialized_record,
)
from .reference_stages import stage_budget, stage_fixture
from .reference_studio import serialization_fixture
from .reference_types import Box, FamilySpec, GridLedger, SerializedRecord


def all_checks() -> dict[str, dict[str, object]]:
    """Expose the four deterministic Chapter 16 result payloads."""

    return {
        "RESULT_CH15_SERIALIZATION_FIXTURE": serialization_fixture(),
        "RESULT_CH15_MIXTURE_LEDGER": mixture_fixture(),
        "RESULT_CH15_RESOLUTION_LEDGER": resolution_fixture(),
        "RESULT_CH15_STAGE_LEDGER": stage_fixture(),
    }


__all__ = [
    "Box",
    "FamilySpec",
    "GridLedger",
    "SerializedRecord",
    "all_checks",
    "caption_record",
    "crop_box",
    "grid_ledger",
    "media_identity_swap_detected",
    "mixture_fixture",
    "mixture_ledgers",
    "normalize_box",
    "reflect_box_horizontally",
    "resolution_fixture",
    "serialization_fixture",
    "source_reuse_ledger",
    "stage_budget",
    "stage_fixture",
    "two_family_specs",
    "validate_serialized_record",
]
