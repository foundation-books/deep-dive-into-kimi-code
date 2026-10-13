"""Dependency-ready interleaved toy pipeline, with an independent checker."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    """Immutable work, placement, dependency, priority, and release contract."""

    name: str
    duration: int
    ranks: tuple[int, ...]
    dependencies: tuple[str, ...]
    priority: tuple[int | str, ...]
    release: int = 0


def _backbone_jobs(m: str, has_encoder: bool, exposed: bool) -> list[Job]:
    result = []
    ef = tuple("E" + m for m in "BCD")
    for c in range(4):
        dep: tuple[str, ...] = (
            (("E" + m,) if has_encoder else ()) if c == 0 else ("F" + m + str(c - 1),)
        )
        if exposed and c == 0:
            dep = ef
        result.append(Job("F" + m + str(c), 1, (c % 2,), dep, (1, c, m)))
    for c in range(3, -1, -1):
        dep = ("F" + m + "3",) if c == 3 else ("B" + m + str(c + 1),)
        result.append(Job("B" + m + str(c), 1, (c % 2,), dep, (0, -c, m)))
    return result


def _sample_jobs(m: str, duration: int, mode: str, late: int) -> list[Job]:
    result = _backbone_jobs(m, bool(duration), mode == "exposed")
    bb = tuple("B" + m + "0" for m in "ABCD")
    ranks = (0, 1) if mode == "cp" and m == "C" else (0,)
    if mode == "cp" and m == "C":
        duration = 2
    if duration:
        result.append(
            Job("E" + m, duration, ranks, (), (2, m), late if m == "C" else 0)
        )
        dep = bb if mode == "exposed" else ("B" + m + "0",)
        result.append(Job("G" + m, duration, ranks, dep, (3, m)))
    return result


def jobs(mode: str = "overlap", late: int = 0) -> list[Job]:
    """Build the fixed two-rank/four-chunk teaching graph, including update."""
    if mode not in ("exposed", "overlap", "cp") or late < 0:
        raise ValueError("invalid policy")
    result = []
    for m, duration in zip("ABCD", (0, 1, 4, 2), strict=True):
        result.extend(_sample_jobs(m, duration, mode, late))
    bb = tuple("B" + m + "0" for m in "ABCD")
    result.append(Job("R", 1, (0, 1), bb + tuple("G" + m for m in "BCD"), (4,)))
    result.append(Job("U", 1, (0, 1), ("R",), (5,)))
    return result


def bubble_start(
    release: int, deadline: int, duration: int, windows: Sequence[tuple[int, int]]
) -> int | None:
    """Earliest fit in the intersection of one window per required resource."""
    if duration <= 0 or not windows or any(b <= a for a, b in windows):
        raise ValueError("invalid duration or resource window")
    start = max(release, *(a for a, _ in windows))
    end = min(deadline, *(b for _, b in windows))
    return start if start + duration <= end else None


def run(work: Sequence[Job]) -> list[tuple[Job, int, int]]:
    """At each completion, launch ready jobs in priority order if ranks free."""
    pending = {j.name: j for j in work}
    if len(pending) != len(work):
        raise ValueError("duplicate job")
    events: list[tuple[Job, int, int]] = []
    end: dict[str, int] = {}
    now = 0
    while pending:
        occupied = {r for j, a, b in events if a <= now < b for r in j.ranks}
        for j in sorted(list(pending.values()), key=lambda j: j.priority):
            ready = all(d in end and end[d] <= now for d in j.dependencies)
            if ready and j.release <= now and not occupied.intersection(j.ranks):
                events.append((j, now, now + j.duration))
                end[j.name] = now + j.duration
                occupied.update(j.ranks)
                del pending[j.name]
        future = [b for _, _, b in events if b > now]
        future += [j.release for j in pending.values() if j.release > now]
        if pending and not future:
            raise ValueError("dependency deadlock")
        if future:
            now = min(future)
    return events


def check(work: Sequence[Job], events: Sequence[tuple[Job, int, int]]) -> int:
    """Validate every supplied event against the declared workload, not policy."""
    declared = {j.name: j for j in work}
    actual = {j.name: (j, a, b) for j, a, b in events}
    if len(actual) != len(events) or set(actual) != set(declared):
        raise ValueError("missing or duplicate event")
    for name, (j, a, b) in actual.items():
        if j != declared[name] or b - a != j.duration or a < j.release:
            raise ValueError("job identity, duration or readiness mismatch")
        if any(actual[d][2] > a for d in j.dependencies):
            raise ValueError("dependency violation")
    for index, (j, a, b) in enumerate(events):
        for k, c, d in events[index + 1 :]:
            if set(j.ranks).intersection(k.ranks) and max(a, c) < min(b, d):
                raise ValueError("rank conflict")
    return max(b for _, _, b in events)
