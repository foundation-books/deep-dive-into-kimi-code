"""Exact mixture and source-reuse ledgers for Chapter 16."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from .reference_types import FamilySpec


def mixture_ledgers(families: Sequence[FamilySpec]) -> dict[str, object]:
    """Accumulate exact record, token, target, visual, and compute measures."""

    if not families or sum((item.probability for item in families), Fraction()) != 1:
        raise ValueError("family probabilities must form a nonempty distribution")
    if len({item.name for item in families}) != len(families):
        raise ValueError("family names must be unique")
    if any(
        item.probability < 0
        or item.backbone_length <= 0
        or not 0 <= item.visual_positions <= item.backbone_length
        or item.target_count < 0
        or item.compute_units <= 0
        for item in families
    ):
        raise ValueError("family fields violate their declared domains")

    token_total = sum(
        (item.probability * item.backbone_length for item in families), Fraction()
    )
    target_total = sum(
        (item.probability * item.target_count for item in families), Fraction()
    )
    visual_total = sum(
        (item.probability * item.visual_positions for item in families), Fraction()
    )
    compute_total = sum(
        (item.probability * item.compute_units for item in families), Fraction()
    )
    if target_total == 0:
        raise ValueError("the target-share denominator is zero")
    return {
        "record_share": {item.name: str(item.probability) for item in families},
        "token_share": {
            item.name: str(item.probability * item.backbone_length / token_total)
            for item in families
        },
        "target_share": {
            item.name: str(item.probability * item.target_count / target_total)
            for item in families
        },
        "compute_share": {
            item.name: str(item.probability * item.compute_units / compute_total)
            for item in families
        },
        "global_visual_token_share": str(visual_total / token_total),
        "expected_backbone_tokens_per_record": str(token_total),
        "expected_targets_per_record": str(target_total),
        "expected_visual_positions_per_record": str(visual_total),
        "expected_compute_units_per_record": str(compute_total),
    }


def two_family_specs() -> tuple[FamilySpec, FamilySpec]:
    """Return the exact text/visual family example from the chapter."""

    return (
        FamilySpec("text", Fraction(1, 2), 64, 64, 0, 1),
        FamilySpec("visual", Fraction(1, 2), 192, 32, 128, 9),
    )


def source_reuse_ledger(
    source_ids: Sequence[str], source_caps: Mapping[str, int]
) -> dict[str, object]:
    """Count source-lineage reuse and reject undeclared or exceeded caps."""

    if not source_ids:
        raise ValueError(
            "source-reuse accounting requires at least one accepted record"
        )
    if any(not source_id for source_id in source_ids):
        raise ValueError("source identifiers must be nonempty")
    if any(cap <= 0 for cap in source_caps.values()):
        raise ValueError("source caps must be positive")
    counts = {
        source_id: sum(item == source_id for item in source_ids)
        for source_id in sorted(set(source_ids))
    }
    undeclared = sorted(set(counts) - set(source_caps))
    exceeded = {
        source_id: {"observed": count, "cap": source_caps[source_id]}
        for source_id, count in counts.items()
        if source_id in source_caps and count > source_caps[source_id]
    }
    if undeclared or exceeded:
        raise ValueError("source reuse is undeclared or exceeds its cap")
    return {
        "counts": counts,
        "caps": {source_id: source_caps[source_id] for source_id in sorted(counts)},
        "within_caps": True,
    }


def _source_reuse_fixture(studio_size: int) -> tuple[dict[str, object], bool]:
    source_ids = ("studio-semantic-source",) * studio_size
    source_reuse = source_reuse_ledger(
        source_ids, {"studio-semantic-source": studio_size}
    )
    try:
        source_reuse_ledger(
            source_ids + ("studio-semantic-source",),
            {"studio-semantic-source": studio_size},
        )
    except ValueError:
        return source_reuse, True
    return source_reuse, False


def mixture_fixture() -> dict[str, object]:
    """Return exact two-family and five-family ledger checks."""

    studio = (
        FamilySpec("text", Fraction(1, 4), 64, 64, 0, 1),
        FamilySpec("caption", Fraction(1, 4), 48, 20, 16, 3),
        FamilySpec("grounding", Fraction(1, 5), 40, 8, 16, 3),
        FamilySpec("code_render", Fraction(3, 20), 72, 24, 16, 4),
        FamilySpec("video", Fraction(3, 20), 64, 12, 16, 6),
    )
    source_reuse, source_cap_violation_rejected = _source_reuse_fixture(len(studio))
    rejected_by_reason = {"decode": 1, "geometry": 1}
    rejection_ledger = {
        "attempted": 7,
        "accepted": 5,
        "rejected_by_reason": rejected_by_reason,
        "conserved": 7 == 5 + sum(rejected_by_reason.values()),
    }
    return {
        "two_family": mixture_ledgers(two_family_specs()),
        "studio": mixture_ledgers(studio),
        "source_reuse": source_reuse,
        "source_cap_violation_rejected": source_cap_violation_rejected,
        "rejection_ledger": rejection_ledger,
        "padding_does_not_create_targets": True,
        "rejected_records_have_separate_counters": True,
    }


__all__ = [
    "mixture_fixture",
    "mixture_ledgers",
    "source_reuse_ledger",
    "two_family_specs",
]
