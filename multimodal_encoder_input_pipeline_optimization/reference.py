"""Small scalar attention, packing, and resource accounting models; no timings."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite


@dataclass(frozen=True)
class Sample:
    """One frozen square-media fixture entry; dimensions are synthetic."""

    name: str
    frames: int
    side: int
    text: int

    @property
    def positions(self) -> int:
        return self.frames * (self.side // 14) ** 2

    @property
    def visual(self) -> int:
        return self.positions // 4

    @property
    def length(self) -> int:
        return self.text + self.visual

    @property
    def spatial_cost(self) -> int:
        return self.frames * (self.side // 14) ** 4


SAMPLES = (
    Sample("A", 0, 0, 64),
    Sample("B", 1, 56, 32),
    Sample("C", 1, 112, 32),
    Sample("D", 4, 56, 32),
)


def packing(samples: Sequence[Sample]) -> list[tuple[str, int]]:
    """Valid-position bijection; no padding or cross-segment attention."""
    if len({s.name for s in samples}) != len(samples):
        raise ValueError("duplicate sample identity")
    return [(s.name, p) for s in samples for p in range(s.length)]


def allowed(mapping: Sequence[tuple[str, int]], q: int, k: int) -> bool:
    """Test same-segment causality using original position identities."""
    return mapping[q][0] == mapping[k][0] and mapping[k][1] <= mapping[q][1]


def greedy(
    costs: Mapping[str, float], lanes: int = 2
) -> tuple[list[list[str]], list[float]]:
    """Largest-first assignment, stable sample/rank ties; costs in one unit."""
    if lanes < 1 or any(c < 0 or not isfinite(c) for c in costs.values()):
        raise ValueError("invalid cost or lane count")
    loads: list[float] = [0] * lanes
    groups: list[list[str]] = [[] for _ in loads]
    for name in sorted(costs, key=lambda n: (-costs[n], n)):
        r = min(range(lanes), key=lambda r: (loads[r], r))
        groups[r].append(name)
        loads[r] += costs[name]
    return groups, loads


def attention(
    q: Sequence[float],
    k: Sequence[float],
    v: Sequence[float],
    upstream: Sequence[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """d=1 unmasked attention and analytic q/k/v derivatives."""
    if not k or len(k) != len(v) or len(q) != len(upstream):
        raise ValueError("incompatible attention shapes")
    output, dq = [], []
    dk, dv = [0.0] * len(k), [0.0] * len(v)
    for qi, gi in zip(q, upstream, strict=True):
        scores = [qi * kj for kj in k]
        weights = [exp(x - max(scores)) for x in scores]
        p = [x / sum(weights) for x in weights]
        oi = sum(pi * vi for pi, vi in zip(p, v, strict=True))
        ds = [pi * gi * (vi - oi) for pi, vi in zip(p, v, strict=True)]
        output.append(oi)
        dq.append(sum(si * kj for si, kj in zip(ds, k, strict=True)))
        for j in range(len(k)):
            dk[j] += ds[j] * qi
            dv[j] += p[j] * gi
    return output, dq, dk, dv


def partitioned_attention(
    q: Sequence[float],
    k: Sequence[float],
    v: Sequence[float],
    upstream: Sequence[float],
    shards: Sequence[Sequence[int]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Gather full K/V, partition query rows, and sum replicated adjoints."""
    indices = [i for shard in shards for i in shard]
    if sorted(indices) != list(range(len(q))):
        raise ValueError("query shards must form a disjoint cover")
    out, dq = [0.0] * len(q), [0.0] * len(q)
    dk, dv = [0.0] * len(k), [0.0] * len(v)
    for shard in shards:
        o, gq, gk, gv = attention(
            [q[i] for i in shard], k, v, [upstream[i] for i in shard]
        )
        for i, oi, gi in zip(shard, o, gq, strict=True):
            out[i], dq[i] = oi, gi
        dk = [a + b for a, b in zip(dk, gk, strict=True)]
        dv = [a + b for a, b in zip(dv, gv, strict=True)]
    return out, dq, dk, dv


def peak(intervals: Sequence[tuple[int, int, int]]) -> int:
    """Half-open byte residency: release before allocation at tied times."""
    if any(b <= a or size < 0 for a, b, size in intervals):
        raise ValueError("invalid residency interval")
    events = sorted(
        (t, delta) for a, b, n in intervals for t, delta in ((a, n), (b, -n))
    )
    live = maximum = 0
    for _, delta in events:
        live += delta
        maximum = max(maximum, live)
    return maximum


def admit(
    arrivals: Sequence[tuple[int, int, int]], capacity: int
) -> list[tuple[int, int, int]]:
    """FIFO byte credit. Items are (arrival,size,residence_duration)."""
    active: list[tuple[int, int]] = []
    result: list[tuple[int, int, int]] = []
    now = 0
    for arrival, size, duration in arrivals:
        if not 0 < size <= capacity or arrival < 0 or duration <= 0:
            raise ValueError("oversized item or invalid arrival")
        now = max(now, arrival)
        active = [(b, n) for b, n in active if b > now]
        while sum(n for _, n in active) + size > capacity:
            now = min(b for b, _ in active)
            active = [(b, n) for b, n in active if b > now]
        result.append((now, now + duration, size))
        active.append((now + duration, size))
    return result


def require_version(expected: object, actual: object, outstanding: int = 0) -> None:
    """Reject stale identities or a plan transition with in-flight work."""
    if expected != actual or outstanding:
        raise ValueError("stale identity or nonquiescent plan")


def weighted_loss(losses: Sequence[Sequence[float]], weights: Sequence[float]) -> float:
    """Normalize weighted valid-position sums by their common denominator."""
    if len(losses) != len(weights) or not losses:
        raise ValueError("invalid loss groups")
    z = sum(w * len(group) for w, group in zip(weights, losses, strict=True))
    if z <= 0 or any(w < 0 for w in weights):
        raise ValueError("invalid objective weights")
    return sum(w * sum(g) for w, g in zip(weights, losses, strict=True)) / z
