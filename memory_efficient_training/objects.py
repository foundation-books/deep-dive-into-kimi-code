"""Versioned objects, lifetimes, tiers, and exact event sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class ObjectKey:
    """Immutable identity for one logical version of a training object."""

    identity: str
    step: int
    microbatch: int
    operator: str
    role: str
    version: int


@dataclass(frozen=True)
class MemoryObject:
    """Logical object with its consumers and full-representation size."""

    key: ObjectKey
    consumers: tuple[int, ...]
    full_units: int

    @property
    def lifetime(self) -> tuple[int, int]:
        if not self.consumers:
            raise ValueError("a memory object requires at least one consumer")
        return (0, max(self.consumers))


@dataclass(frozen=True)
class Residency:
    """One object's half-open residency interval on a rank and tier."""

    key: ObjectKey
    rank: int
    tier: str
    start: int
    stop: int
    units: int

    def __post_init__(self) -> None:
        if self.start >= self.stop:
            raise ValueError("residency must be a nonempty half-open interval")
        if self.units < 0:
            raise ValueError("residency units must be nonnegative")


@dataclass(frozen=True)
class SweepResult:
    """Peak, first violation, and ordered trace produced by an event sweep."""

    peak: int
    peak_event: tuple[int, str]
    first_violation: tuple[int, str] | None
    trace: tuple[tuple[int, str, int], ...]


def sweep_peak(
    fixed_units: int,
    residencies: Iterable[Residency],
    *,
    rank: int,
    tier: str,
    capacity: int,
    release_before_allocate: bool = True,
) -> SweepResult:
    """Return the exact ordered peak for constant half-open residencies."""
    events: list[tuple[int, int, str, int]] = []
    for item in residencies:
        if item.rank != rank or item.tier != tier:
            continue
        release_priority = 0 if release_before_allocate else 1
        allocate_priority = 1 if release_before_allocate else 0
        events.append(
            (item.start, allocate_priority, f"+{item.key.identity}", item.units)
        )
        events.append(
            (item.stop, release_priority, f"-{item.key.identity}", -item.units)
        )
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    resident = fixed_units
    peak = fixed_units
    peak_event = (-1, "fixed")
    violation: tuple[int, str] | None = None
    trace: list[tuple[int, str, int]] = []
    for timestamp, _, name, delta in events:
        resident += delta
        trace.append((timestamp, name, resident))
        if resident > peak:
            peak = resident
            peak_event = (timestamp, name)
        if resident > capacity and violation is None:
            violation = (timestamp, name)
    return SweepResult(peak, peak_event, violation, tuple(trace))


def running_object(
    microbatch: int, role: str, units: int, consumers: Sequence[int]
) -> MemoryObject:
    """Construct a versioned object from the chapter's running fixture."""
    return MemoryObject(
        ObjectKey(role, 7, microbatch, "block11", role, 0),
        tuple(consumers),
        units,
    )


def naive_warmup_residencies() -> tuple[Residency, ...]:
    """Three rank-0 bundles and one rank-1 bundle live at event 4."""
    rows: list[Residency] = []
    for microbatch, start in enumerate((1, 3, 4)):
        key = ObjectKey(f"bundle{microbatch}", 7, microbatch, "stage0", "bundle", 0)
        rows.append(Residency(key, 0, "hbm", start, 5 + microbatch, 13))
    key = ObjectKey("bundle0", 7, 0, "stage1", "bundle", 0)
    rows.append(Residency(key, 1, "hbm", 2, 5, 13))
    return tuple(rows)


def naive_peaks() -> tuple[int, int]:
    """Return the two naive running-example HBM peaks."""
    rows = naive_warmup_residencies()
    return tuple(
        sweep_peak(18, rows, rank=rank, tier="hbm", capacity=32).peak for rank in (0, 1)
    )  # type: ignore[return-value]
