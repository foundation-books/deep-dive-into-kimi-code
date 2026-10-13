"""Deterministic placement and arithmetic; no engine or network execution."""

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import ceil

WEIGHTS = (90, 60, 40, 20, 10, 10)
SECONDARIES = ("C1", "C2", "C3", "C1", "C2", "C3")
BUDGETS = {"S": (30, 20, 10), "L": (50, 25, 20), "R": (20, 5, 10)}


def ranking(session: str, clusters: tuple[str, ...]) -> tuple[str, ...]:
    """Fixed, equal-weight physical-domain scores; membership is not hashed."""
    if not session or "\0" in session:
        raise ValueError("invalid session identifier")
    if len(set(clusters)) != len(clusters):
        raise ValueError("one candidate per physical domain")
    if any(not item or "\0" in item for item in clusters):
        raise ValueError("invalid cluster identifier")

    def key(cluster: str) -> tuple[int, str]:
        digest = sha256((session + "\0" + cluster).encode("utf-8")).digest()
        return -int.from_bytes(digest, "big"), cluster

    return tuple(sorted(clusters, key=key))


def assignment(session: str, clusters: tuple[str, ...]) -> tuple[str, str]:
    """Return primary and secondary in distinct physical domains."""
    ordered = ranking(session, clusters)
    if len(ordered) < 2:
        raise ValueError("redundancy needs two physical domains")
    return ordered[0], ordered[1]


@dataclass(frozen=True)
class Candidate:
    """An acquired-state proposal whose affordability is an input fact."""

    cluster: str
    context: str = "v7"
    ready: bool = True
    healthy: bool = True
    authorized: bool = True
    acquired_prefix: int = 0
    queue_ms: int = 0
    affordable: bool = True

    def eligible(self, context: str) -> bool:
        return (
            self.context == context and self.ready and self.healthy and self.authorized
        )

    def estimate(self, positions: int = 100) -> Fraction:
        if not 0 <= self.acquired_prefix <= positions or self.queue_ms < 0:
            raise ValueError("invalid acquired boundary or queue")
        return Fraction(self.queue_ms + 1) + Fraction(
            positions - self.acquired_prefix, 10
        )


def placement(
    candidates: tuple[Candidate, ...],
    pair: tuple[str, str],
    slack: int,
    policy: str = "affinity",
    context: str = "v7",
) -> str | None:
    """Descriptions already contain acquired state; return a proposal, not rights."""
    if policy not in ("affinity", "cost"):
        raise ValueError("unknown policy")
    usable = [c for c in candidates if c.eligible(context)]
    if len({c.cluster for c in candidates}) != len(candidates):
        raise ValueError("duplicate destination")
    if policy == "affinity":
        primary = next((c for c in usable if c.cluster == pair[0]), None)
        selected = primary or next((c for c in usable if c.cluster == pair[1]), None)
        usable = [selected] if selected else []
    feasible = [c for c in usable if c.affordable and c.estimate() <= slack]
    return (
        min(feasible, key=lambda c: (c.estimate(), c.cluster)).cluster
        if feasible
        else None
    )


def recovery_route(
    session: str,
    secondary: str,
    candidates: tuple[Candidate, ...],
    *,
    reconstructable: bool,
    fenced: bool,
    attempt: int = 2,
    now: int = 10,
    deadline: int = 190,
) -> str | None:
    """Explicit fallback; unresolved authority returns no dispatch destination."""
    if not reconstructable or not fenced or attempt > 3 or now >= deadline:
        return None
    eligible = tuple(c.cluster for c in candidates if c.eligible("v7") and c.affordable)
    if secondary in eligible:
        return secondary
    ordered = ranking(session, eligible)
    return ordered[0] if ordered else None


def failover_work(
    weights: tuple[int, ...] = WEIGHTS,
    destinations: tuple[str, ...] = SECONDARIES,
) -> dict[str, int]:
    """Sum declared replay weights per assigned survivor."""
    if len(weights) != len(destinations) or any(w < 0 for w in weights):
        raise ValueError("invalid weighted assignment")
    totals: dict[str, int] = {}
    for weight, destination in zip(weights, destinations, strict=True):
        totals[destination] = totals.get(destination, 0) + weight
    return totals


def moments(weights: tuple[int, ...], survivors: int) -> tuple[Fraction, Fraction]:
    """Exact moments under independent uniform secondary choices."""
    if survivors < 1 or any(w < 0 for w in weights):
        raise ValueError("invalid probability model")
    probability = Fraction(1, survivors)
    return sum(weights) * probability, sum(w * w for w in weights) * probability * (
        1 - probability
    )


def replay_bound(
    work: tuple[int, ...], allowance: tuple[int, ...], slotted: bool = True
) -> Fraction:
    """Lower bound for ten-ms end-credit or continuous-rate service."""
    if not work or len(work) != len(allowance):
        raise ValueError("matching nonempty vectors required")
    if any(w < 0 for w in work) or any(b <= 0 for b in allowance):
        raise ValueError("nonnegative work and positive allowance required")
    ratios = [Fraction(w, b) for w, b in zip(work, allowance, strict=True)]
    return Fraction(10 * max(ceil(x) for x in ratios)) if slotted else 10 * max(ratios)


def service_bound(ahead: int, work: int, rate: int, latency: int) -> Fraction:
    """Conditional FIFO completion bound for a ready admitted item."""
    if min(ahead, work, latency) < 0 or rate <= 0:
        raise ValueError("invalid service premise")
    return latency + Fraction(ahead + work, rate)


def usable_capacity(time_ms: int, ready_ms: int = 50) -> int:
    """Readiness-gated C4 prefill units per ten-ms interval."""
    return 100 if time_ms >= ready_ms else 0


def drain_ready(
    new_assignments_stopped: bool, obligations: int, replacement_verified: bool
) -> bool:
    """Check stopped arrivals, verified replacements and zero obligations."""
    if obligations < 0:
        raise ValueError("negative obligations")
    return new_assignments_stopped and obligations == 0 and replacement_verified
