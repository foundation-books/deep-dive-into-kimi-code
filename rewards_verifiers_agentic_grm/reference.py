"""Public Chapter 22 reward, verifier, and GRM reference API."""

from __future__ import annotations

from .reference_checks import all_checks
from .reference_comparison import ComparisonRecord, compare_candidates, win_fraction
from .reference_core import (
    CELL_STATUSES,
    DISPOSITIONS,
    EVIDENCE_STATUSES,
    Component,
    EvidenceRecord,
    RewardLedger,
    Rubric,
    ScorepadCell,
    VerifierResult,
    compose_reward,
    exact_verifier,
    hard_gate,
    inapplicable,
    mask_iou,
    missing,
    observed,
    public_hidden_gap,
    relation_verifier,
    tolerance_verifier,
    verbosity_event,
)
from .reference_studio import (
    RewardHandoff,
    center_record,
    centered_studio_group,
    studio_handoffs,
    studio_ledgers,
)

__all__ = [
    "CELL_STATUSES",
    "DISPOSITIONS",
    "EVIDENCE_STATUSES",
    "ComparisonRecord",
    "Component",
    "EvidenceRecord",
    "RewardHandoff",
    "RewardLedger",
    "Rubric",
    "ScorepadCell",
    "VerifierResult",
    "all_checks",
    "center_record",
    "centered_studio_group",
    "compare_candidates",
    "compose_reward",
    "exact_verifier",
    "hard_gate",
    "inapplicable",
    "mask_iou",
    "missing",
    "observed",
    "public_hidden_gap",
    "relation_verifier",
    "studio_handoffs",
    "studio_ledgers",
    "tolerance_verifier",
    "verbosity_event",
    "win_fraction",
]
