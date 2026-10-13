"""A synchronous two-stage, equal-duration GPipe-style teaching schedule."""

from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Sequence
from itertools import pairwise


@dataclass(frozen=True)
class Event:
    """One stage occupies [slot, slot+1) for one microbatch operation."""

    stage: int
    microbatch: int
    direction: str
    slot: int


def flush_schedule(microbatches: int) -> tuple[Event, ...]:
    """All forward events finish before backward begins; no weight updates overlap."""
    if microbatches < 1:
        raise ValueError("at least one microbatch is required")
    events = []
    for stage in range(2):
        for microbatch in range(microbatches):
            events.append(Event(stage, microbatch, "F", microbatch + stage))
            events.append(
                Event(stage, microbatch, "B", 2 * microbatches - microbatch + 1 - stage)
            )
    return tuple(events)


def validate_schedule(
    events: Sequence[Event], microbatches: int, update_slot: int
) -> Fraction:
    """Check coverage, model dependencies, exclusive stage use, and update timing."""
    expected = {
        (s, m, d) for s in range(2) for m in range(microbatches) for d in ("F", "B")
    }
    observed = {(e.stage, e.microbatch, e.direction): e.slot for e in events}
    if len(events) != len(expected) or set(observed) != expected:
        raise ValueError("missing or duplicate event")
    if any(e.slot < 0 for e in events) or len(
        {(e.stage, e.slot) for e in events}
    ) != len(events):
        raise ValueError("invalid slot or overlapping stage use")
    for m in range(microbatches):
        chain = [(0, m, "F"), (1, m, "F"), (1, m, "B"), (0, m, "B")]
        if any(observed[a] + 1 > observed[b] for a, b in pairwise(chain)):
            raise ValueError("producer has not completed before its consumer")
    makespan = max(e.slot for e in events) + 1
    if update_slot < makespan:
        raise ValueError("update precedes completed gradients")
    return Fraction(len(events), 2 * makespan)
