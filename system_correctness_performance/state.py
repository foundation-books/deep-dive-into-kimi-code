"""Identity-only durable-state fixture, with no model or storage implementation."""

from dataclasses import dataclass, replace
from fractions import Fraction

PREFIX = tuple(10 + i % 17 for i in range(3072))
PROMPT = PREFIX + (30,) * 1024
ROLES = ("kda", "mla", "convolution", "position")
TAPE = (Fraction(1, 5), Fraction(4, 5), Fraction(3, 4))


@dataclass(frozen=True)
class StateObject:
    """A role's exact prefix and numerical/preprocessing identity witness."""

    role: str
    history: tuple[int, ...]
    version: str = "v7"
    preprocessing: str = "text-token-ids-v1"


@dataclass(frozen=True)
class Record:
    """One atomic durable prefix with aligned objects and replay cursor."""

    history: tuple[int, ...]
    objects: tuple[StateObject, ...]
    cursor: int
    epoch: int
    cache_generation: int = 12
    version: str = "v7"
    preprocessing: str = "text-token-ids-v1"


def materialize(history: tuple[int, ...], cursor: int, epoch: int) -> Record:
    """Build identity witnesses for all four text-only state roles."""
    return Record(
        history, tuple(StateObject(role, history) for role in ROLES), cursor, epoch
    )


def compatible(record: Record) -> bool:
    """Check every role against the durable prefix's execution identity."""
    return (
        record.cursor >= 0
        and len(record.objects) == len(ROLES)
        and {obj.role for obj in record.objects} == set(ROLES)
        and all(
            obj.history == record.history
            and obj.version == record.version
            and obj.preprocessing == record.preprocessing
            for obj in record.objects
        )
    )


def prepare_suffix(record: Record) -> Record:
    """Fixed conditioned proposal (1,2,3) and explicit acceptance decisions."""
    if not compatible(record) or record.history != PROMPT or record.cursor != 0:
        raise ValueError("fixture starts at the compatible full prompt and cursor zero")
    # At P, p=q; at Pa the b acceptance ratio is (.1/.4).
    if not TAPE[0] < 1 or TAPE[1] < Fraction(1, 4):
        raise AssertionError("decision tape no longer exercises the planned branch")
    residual = (Fraction(1, 2), Fraction(0), Fraction(0), Fraction(1, 2))
    mass = Fraction()
    correction = 0
    for token, probability in enumerate(residual, start=1):
        mass += probability
        if TAPE[2] < mass:
            correction = token
            break
    return materialize(record.history + (1, correction), 3, record.epoch)


class DurableFixture:
    """Atomic assignment and epoch fencing are assumptions of this model."""

    def __init__(self) -> None:
        self.authority = 8
        self.record = materialize(PROMPT, 0, 8)
        self.visible: dict[int, int] = {}

    def commit(self, candidate: Record, writer_epoch: int) -> bool:
        if writer_epoch != self.authority or candidate.epoch != self.authority:
            return False
        if not compatible(candidate):
            raise ValueError("incompatible state bundle")
        old = self.record
        if (
            candidate.version != old.version
            or candidate.preprocessing != old.preprocessing
        ):
            raise ValueError("execution identity changed without a new request")
        if candidate.cache_generation != old.cache_generation:
            raise ValueError("unexpected cache generation")
        if (
            candidate.history[: len(old.history)] != old.history
            or candidate.cursor < old.cursor
        ):
            raise ValueError("commit does not extend the durable history")
        self.record = candidate
        return True

    def takeover(self) -> Record:
        self.authority += 1
        return replace(self.record, epoch=self.authority)

    def deliver(self, writer_epoch: int) -> bool:
        if writer_epoch != self.authority:
            return False
        for index, token in enumerate(
            self.record.history[len(PROMPT) :], start=len(PROMPT)
        ):
            if index in self.visible and self.visible[index] != token:
                raise ValueError("conflicting visible token")
            self.visible[index] = token
        return True


def crash_fixture(after_commit: bool) -> DurableFixture:
    """Resume either crash branch and exercise stale writes and redelivery."""
    fixture = DurableFixture()
    prepared = prepare_suffix(fixture.record)
    if after_commit:
        fixture.commit(prepared, 8)
    recovered = fixture.takeover()
    if recovered.history == PROMPT:
        recovered = prepare_suffix(recovered)
    fixture.commit(recovered, 9)
    if fixture.commit(prepared, 8):
        raise AssertionError("obsolete worker changed durable state")
    fixture.deliver(9)
    fixture.deliver(9)  # retransmission deduplicates by token index
    return fixture
