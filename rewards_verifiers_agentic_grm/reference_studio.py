"""Studio records for the Chapter 22 reference module."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

from .reference_core import (
    EVIDENCE_STATUSES,
    Component,
    EvidenceRecord,
    RewardLedger,
    compose_reward,
    inapplicable,
    observed,
)


@dataclass(frozen=True)
class RewardHandoff:
    """Validated final status and reward ledger for one candidate."""

    candidate: str
    ledger: RewardLedger
    scalar: Fraction | None
    final_status: str
    evidence_statuses: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_disposition: str
    candidate_evidence_violation: bool
    group_id: str
    slot_id: int
    sampling_provenance_valid: bool = True
    infrastructure_resolved: bool = True
    admission_policy: str = "studio-admission-v1"

    @property
    def training_eligible(self) -> bool:
        """Admit a number only under the explicit sampling contract."""
        if (
            not self.sampling_provenance_valid
            or not self.infrastructure_resolved
            or self.scalar is None
        ):
            return False
        if self.ledger.gate == 1:
            return True
        return (
            self.ledger.gate == 0
            and self.admission_policy == "studio-admission-v1"
            and self.candidate_evidence_violation
            and self.evidence_disposition == "invalid"
        )

    @property
    def scalar_purpose(self) -> str:
        """Keep an eligible negative outcome's blocked diagnostic visible."""
        if self.scalar is None:
            return "no_scalar"
        if not self.training_eligible:
            return "diagnostic_only"
        if self.ledger.gate == 0:
            return "blocked_diagnostic_and_training_outcome"
        return "training_outcome"

    @property
    def group_action(self) -> str:
        if self.training_eligible:
            return "retain_slot"
        if not self.infrastructure_resolved:
            return "defer_original_slot"
        return "quarantine_group"

    def validate(self) -> bool:
        self.ledger.validate()
        if not self.candidate or not self.final_status or not self.limitations:
            raise ValueError("handoff identity, status, and limitations are required")
        if not self.evidence_statuses or any(
            status not in EVIDENCE_STATUSES for status in self.evidence_statuses
        ):
            raise ValueError("handoff evidence statuses are invalid")
        if self.scalar != compose_reward(self.ledger):
            raise ValueError("handoff scalar does not match its ledger")
        if not self.group_id or self.slot_id not in {1, 2, 3, 4}:
            raise ValueError("studio handoff requires one of four fixed slots")
        if self.admission_policy not in {"studio-admission-v1", "reject-only-v1"}:
            raise ValueError("unknown sampling admission policy")
        if self.evidence_disposition not in {"pass", "invalid", "indeterminate"}:
            raise ValueError("unexpected studio evidence disposition")
        return True


def center_record(candidate: str, color: str, value: tuple[int, int]) -> EvidenceRecord:
    """Build qualified image-center evidence for one candidate and color."""
    payload = json.dumps([candidate, color, *value], separators=(",", ":")).encode()
    return EvidenceRecord(
        f"{candidate}-{color}",
        "image_center",
        value,
        "studio-location-tool",
        f"{candidate}-event-{color}",
        "studio-tool-v1",
        hashlib.sha256(payload).hexdigest(),
    )


def studio_ledgers() -> dict[str, RewardLedger]:
    """Return the four frozen teaching ledgers."""
    zero = observed(0)
    omitted = inapplicable()
    skipped = Component(None, "not_evaluated")
    return {
        "A": RewardLedger(
            1,
            observed(1),
            omitted,
            observed(Fraction(9, 10)),
            omitted,
            zero,
            omitted,
            "studio-profile-v2",
        ),
        "B": RewardLedger(
            1,
            observed(1),
            omitted,
            observed(Fraction(7, 10)),
            omitted,
            observed(Fraction(1, 4)),
            omitted,
            "studio-profile-v2",
        ),
        "C": RewardLedger(
            0,
            skipped,
            omitted,
            skipped,
            omitted,
            skipped,
            omitted,
            "studio-profile-v2",
        ),
        "D": RewardLedger(
            1,
            observed(0),
            omitted,
            omitted,
            omitted,
            zero,
            omitted,
            "studio-profile-v2",
        ),
    }


def studio_handoffs() -> dict[str, RewardHandoff]:
    """Build the deterministic handoffs teaching fixture."""
    statuses = {
        "A": "grounded_concise_success",
        "B": "grounded_verbose_success",
        "C": "invalid_fabricated_evidence",
        "D": "valid_task_failure",
    }
    limitations = ("teaching fixture, not a model measurement",)
    return {
        name: RewardHandoff(
            name,
            ledger,
            compose_reward(ledger),
            statuses[name],
            ("fixture", "checked", "pending", "unknown"),
            limitations,
            "invalid" if name == "C" else "pass",
            name == "C",
            "studio-group-001",
            slot,
        )
        for slot, (name, ledger) in enumerate(studio_ledgers().items(), 1)
    }


def centered_studio_group(
    handoffs: tuple[RewardHandoff, ...],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Center exactly four eligible original slots; reject incomplete groups."""
    if len(handoffs) != 4 or {item.slot_id for item in handoffs} != {1, 2, 3, 4}:
        raise ValueError("promised four-slot group is not ready")
    if len({item.group_id for item in handoffs}) != 1:
        raise ValueError("group identities differ")
    if len({item.admission_policy for item in handoffs}) != 1:
        raise ValueError("group admission policies differ")
    rewards = []
    for item in sorted(handoffs, key=lambda value: value.slot_id):
        item.validate()
        if not item.training_eligible or item.scalar is None:
            raise ValueError("a promised slot is ineligible")
        rewards.append(item.scalar)
    mean = sum(rewards, Fraction(0)) / 4
    return mean, tuple(reward - mean for reward in rewards)
