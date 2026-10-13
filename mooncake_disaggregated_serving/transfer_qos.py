"""Finite non-preemptive scheduling and source-inspired promotion decisions."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Slice:
    """One independently queued teaching slice; lower priority number wins."""

    name: str
    arrival: float
    priority: int
    duration: float = 1.0


def schedule(
    slices: tuple[Slice, ...], priority: bool
) -> tuple[tuple[str, float, float], ...]:
    """Post exactly one slice at a time, selecting only among arrived work.

    The C++ worker can dequeue/post lists of slices. This declared one-slice
    limit isolates the effect of priority without claiming runtime latency.
    """
    if len({s.name for s in slices}) != len(slices):
        raise ValueError("unique slice names required")
    for item in slices:
        if not all(isfinite(v) for v in (item.arrival, item.duration)):
            raise ValueError("finite times required")
        if item.arrival < 0 or item.duration <= 0 or item.priority not in (0, 1, 2):
            raise ValueError("invalid slice")
    pending = list(enumerate(slices))
    now = 0.0
    result = []
    while pending:
        ready = [(i, s) for i, s in pending if s.arrival <= now]
        if not ready:
            now = min(s.arrival for _, s in pending)
            continue
        chosen = min(
            ready,
            key=lambda pair: (
                pair[1].priority if priority else 0,
                pair[1].arrival,
                pair[0],
            ),
        )
        pending.remove(chosen)
        item = chosen[1]
        result.append((item.name, now, now + item.duration))
        now += item.duration
    return tuple(result)


def promotion(
    timestamps: tuple[int, ...], now: int, timeout: int, per_entry: bool
) -> tuple[tuple[int, ...], int, int]:
    """Return promoted indices, age-predicate count and entries traversed.

    Zero means unknown timestamp, matching the pinned helper. Both worker
    paths traverse all drained entries; age-predicate counts are narrower.
    """
    if min((now, timeout, *timestamps)) < 0:
        raise ValueError("nonnegative clock values required")
    if not timestamps:
        return (), 0, 0

    def expired(stamp: int) -> bool:
        return stamp > 0 and now >= stamp and now - stamp >= timeout

    if per_entry:
        moved = tuple(i for i, stamp in enumerate(timestamps) if expired(stamp))
        return moved, len(timestamps), len(timestamps)
    moved = tuple(range(len(timestamps))) if expired(timestamps[0]) else ()
    return moved, 1, len(timestamps)
