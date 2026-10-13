"""Dependency-free semantic-oracle facade for Chapter 17.

The facade composes typed studio records, causal validation, frozen variants,
canonical symbolic serialization, masked-loss arithmetic, mixture ledgers, and
the sequential-error fixture.  It is teaching code, not a Kimi K3 pipeline.
"""

from __future__ import annotations

from .reference_ledgers import (
    mixture_fixture,
    normalized_shares,
    sequential_error_fixture,
)
from .reference_serialization import (
    masked_loss_fixture,
    parse_events,
    serialization_fixture,
    serialize_events,
    structural_tokens_and_mask,
    validate_mask,
    weighted_masked_loss,
)
from .reference_types import (
    BLUE_BOX,
    CROPPED_BLUE_BOX,
    CROPPED_RED_BOX,
    RED_BOX,
    Event,
    ValidationResult,
    environment_contract,
    recovery_events,
    task_contract,
    valid_events,
)
from .reference_validation import certificate_for, content_hash, validate_record
from .reference_variants import (
    EXPECTED_REJECTIONS,
    EXPECTED_VARIANTS,
    REPAIR_DEPENDENCY_GRAPH,
    corrupted_variant,
    record_fixture,
)


def all_checks() -> dict[str, dict[str, object]]:
    """Expose the four deterministic Chapter 17 result payloads."""

    return {
        "RESULT_CH16_RECORD_ORACLE": record_fixture(),
        "RESULT_CH16_SERIALIZATION_MASK_ORACLE": {
            **serialization_fixture(),
            "masked_loss": masked_loss_fixture(),
        },
        "RESULT_CH16_MIXTURE_LEDGER_ORACLE": mixture_fixture(),
        "RESULT_CH16_SEQUENTIAL_ERROR_ORACLE": sequential_error_fixture(),
    }


__all__ = [
    "BLUE_BOX",
    "CROPPED_BLUE_BOX",
    "CROPPED_RED_BOX",
    "EXPECTED_REJECTIONS",
    "EXPECTED_VARIANTS",
    "Event",
    "RED_BOX",
    "REPAIR_DEPENDENCY_GRAPH",
    "ValidationResult",
    "all_checks",
    "certificate_for",
    "content_hash",
    "corrupted_variant",
    "environment_contract",
    "masked_loss_fixture",
    "mixture_fixture",
    "normalized_shares",
    "parse_events",
    "record_fixture",
    "recovery_events",
    "sequential_error_fixture",
    "serialization_fixture",
    "serialize_events",
    "structural_tokens_and_mask",
    "task_contract",
    "valid_events",
    "validate_mask",
    "validate_record",
    "weighted_masked_loss",
]
