"""Declared-order bucket grouping, staged restore and one-replica placement."""

from dataclasses import dataclass


class Buckets:
    """Accumulate fresh keys; input order is explicit, unlike an unordered map."""

    def __init__(self, byte_limit: int, key_limit: int) -> None:
        if min(byte_limit, key_limit) <= 0:
            raise ValueError("positive limits required")
        self.byte_limit = byte_limit
        self.key_limit = key_limit
        self.pending: list[tuple[str, int]] = []

    def heartbeat(
        self, objects: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[str, ...], ...]:
        """Close at count/byte limit or before the next fitting key would overflow.

        Oversized keys are skipped. Already stored keys must be removed by the
        caller. An empty heartbeat leaves a partial bucket waiting.
        """
        ready = []
        for name, size in objects:
            if size <= 0:
                raise ValueError("positive object sizes required")
            if size > self.byte_limit or name in {key for key, _ in self.pending}:
                continue
            if sum(n for _, n in self.pending) + size > self.byte_limit:
                ready.append(tuple(key for key, _ in self.pending))
                self.pending = []
            self.pending.append((name, size))
            if (
                len(self.pending) == self.key_limit
                or sum(n for _, n in self.pending) == self.byte_limit
            ):
                ready.append(tuple(key for key, _ in self.pending))
                self.pending = []
        return tuple(ready)


def restore_timeline(
    request: float = 8,
    buffer_ready: float = 11,
    lookup: float = 1,
    disk: float = 4,
    network: float = 2,
    install: float = 1,
) -> tuple[tuple[str, float, float], ...]:
    """Serial restore after a disk replica is published; each stage is explicit."""
    if min(request, buffer_ready, lookup, disk, network, install) < 0:
        raise ValueError("nonnegative times required")
    query_end = request + lookup
    read_start = max(query_end, buffer_ready)
    read_end = read_start + disk
    copied = read_end + network
    return (
        ("lookup", request, query_end),
        ("wait for staging", query_end, read_start),
        ("disk read", read_start, read_end),
        ("network", read_end, copied),
        ("installation", copied, copied + install),
    )


@dataclass(frozen=True)
class Segment:
    """Memory capacity and possibly unavailable backing-SSD free-ratio estimate."""

    name: str
    memory_free: int
    ssd_free_ratio: float | None


def choose_segment(
    segments: tuple[Segment, ...],
    size: int,
    preferred: tuple[str, ...],
    sampled: tuple[str, ...],
    fallback: tuple[str, ...],
) -> tuple[str, str] | None:
    """Choose one replica; supplied sample/fallback order replaces randomness.

    Missing metrics score 1.0 as in the inspected implementation. Name-based
    ties are a teaching convention; no disk reservation or fresh metric is
    implied. The caller declares the attempted fallback set.
    """
    if size <= 0 or len({s.name for s in segments}) != len(segments):
        raise ValueError("positive size and distinct segments required")
    for segment in segments:
        if segment.memory_free < 0 or (
            segment.ssd_free_ratio is not None and not 0 <= segment.ssd_free_ratio <= 1
        ):
            raise ValueError("invalid capacity or free ratio")
    table = {s.name: s for s in segments}

    def score(name: str) -> float:
        ratio = table[name].ssd_free_ratio
        return 1.0 if ratio is None else ratio

    ranking = sorted(
        (name for name in sampled if name in table), key=lambda n: (-score(n), n)
    )
    for phase, names in (
        ("preferred", preferred),
        ("sampled", tuple(ranking)),
        ("fallback", fallback),
    ):
        for name in names:
            if name in table and table[name].memory_free >= size:
                return name, phase
    return None
