"""Finite authoritative-store model with an enforced generation gateway."""

from dataclasses import dataclass, replace
from .reference import Cache, Record, cache_bytes, can_admit, compatible, valid_record


@dataclass
class Controller:
    """Single serialized authority; takeover models fencing and draining old execution."""

    record: Record = Record()
    generation: int = 4
    stage: str = "paused"
    candidate: Record | None = None
    active: int | None = None
    adopted_version: int = 8
    reserved: int = 0
    active_version: int | None = None

    def takeover(self) -> int:
        """Fence and drain prior execution before installing a new attempt generation."""
        self.generation += 1
        self.active = None
        self.reserved = 0
        self.active_version = None
        self.candidate = None
        self.stage = "paused"
        return self.generation

    def transition(
        self, generation: int, event: str, candidate: Record | None = None
    ) -> bool:
        """Execute one idempotent transition at the current authority revision."""
        if generation != self.generation:
            return False
        if event == "prepare":
            return self._prepare(generation, candidate)
        if event == "validate":
            return self._validate()
        if event == "commit":
            if self.stage in ("committed", "admitted"):
                return True
            if self.stage != "validated" or self.candidate is None:
                return False
            self.record, self.stage = self.candidate, "committed"
            return True
        return False

    def admit(
        self,
        generation: int,
        cache: Cache,
        capacity: int,
        other_reservations: tuple[int, ...] = (),
        margin: int = 0,
    ) -> bool:
        """Atomically check authority, compatible state and growth capacity for G4."""
        if generation != self.generation or self.stage not in ("committed", "admitted"):
            return False
        if not compatible(cache, self.adopted_version, self.record.tokens):
            return False
        required = cache_bytes(12)
        if not can_admit(other_reservations, (required,), capacity, margin):
            return False
        self.reserved = required
        self.active_version = self.adopted_version
        self.active, self.stage = generation, "admitted"
        return True

    def _prepare(self, generation: int, candidate: Record | None) -> bool:
        proposed = candidate or replace(self.record, generation=generation)
        if self.stage == "prepared":
            return self.candidate == proposed
        if self.stage != "paused" or proposed.generation != generation:
            return False
        self.candidate, self.stage = proposed, "prepared"
        return True

    def _validate(self) -> bool:
        if self.stage == "validated":
            return True
        if self.stage != "prepared" or self.candidate is None:
            return False
        expected = replace(self.record, generation=self.generation)
        if not valid_record(self.candidate) or self.candidate != expected:
            return False
        self.stage = "validated"
        return True

    def may_execute(self, generation: int) -> bool:
        """Model an effectful gateway's current-generation check."""
        return (
            self.active == generation == self.generation
            and self.stage == "admitted"
            and self.reserved == cache_bytes(12)
            and self.active_version == self.adopted_version
        )


def effect_retry_allowed(outcome: str, service_deduplicates: bool) -> bool:
    """An unknown external effect cannot be blindly retried."""
    return outcome == "confirmed_absent" or (
        outcome == "unknown" and service_deduplicates
    )
