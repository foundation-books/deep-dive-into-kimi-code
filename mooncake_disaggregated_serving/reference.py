"""Finite Chapter 42 accounting and safety oracles; no hardware simulation."""

from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
import json
from math import ceil


@dataclass(frozen=True)
class Context:
    """Exact computation context included before hashing token blocks."""

    namespace: str
    version: str
    layout: str
    canonical_context: str


def prefix_keys(
    tokens: tuple[str, ...], context: Context, block_size: int = 4
) -> tuple[str, ...]:
    """Hash full blocks with predecessor identity; discard a partial tail."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    fields = (
        context.namespace,
        context.version,
        context.layout,
        context.canonical_context,
    )
    previous = sha256(json.dumps(fields).encode()).hexdigest()
    result = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        encoded = json.dumps((previous, tokens[start : start + block_size])).encode()
        previous = sha256(encoded).hexdigest()
        result.append(previous)
    return tuple(result)


def reusable_boundary(
    required: tuple[str, ...], protected_complete: set[str], block_size: int = 4
) -> int:
    """Inputs represent already validated complete per-block groups."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    count = 0
    for key in required:
        if key not in protected_complete:
            break
        count += block_size
    return count


def allocated_bytes(tokens: int, block_size: int = 4) -> int:
    """Dense teaching KV payload under full-page allocation, in bytes."""
    if tokens < 0 or block_size <= 0:
        raise ValueError("invalid token or block count")
    return ceil(tokens / block_size) * block_size * 256


def occupancy(allocations: dict[str, int]) -> int:
    """Keys identify unique physical allocations, not logical aliases."""
    if any(size < 0 for size in allocations.values()):
        raise ValueError("negative allocation")
    return sum(allocations.values())


def layer_schedule(
    extra_reader_until: float = 0,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Four records; two alternating slots; independent serial engines."""
    if extra_reader_until < 0:
        raise ValueError("negative reader time")
    compute_end = transfer_end = 0.0
    releases = [0.0, 0.0]
    timeline = []
    for layer in range(4):
        slot = layer % 2
        start = max(compute_end, releases[slot])
        compute_end = start + 3
        transfer_start = max(compute_end, transfer_end)
        transfer_end = transfer_start + 2
        releases[slot] = max(transfer_end, extra_reader_until if layer == 0 else 0)
        timeline.append(((start, compute_end), (transfer_start, transfer_end)))
    return tuple(timeline)


def chunk_schedule(chunks: int = 3, stages: int = 2, duration: float = 4) -> float:
    """Makespan for equal-service stages with causal chunk dependencies."""
    if chunks <= 0 or stages <= 0 or duration <= 0:
        raise ValueError("positive pipeline dimensions required")
    finishes = [[0.0] * (stages + 1) for _ in range(chunks + 1)]
    for i in range(1, chunks + 1):
        for j in range(1, stages + 1):
            finishes[i][j] = max(finishes[i - 1][j], finishes[i][j - 1]) + duration
    return finishes[-1][-1]


def route_costs(bandwidth: float) -> dict[str, float]:
    """TTFT in assigned ms; each route starts from the original placement."""
    if bandwidth <= 0:
        raise ValueError("positive bandwidth required")
    return {
        "P1": 9 + 4 + 5,
        "P2-local": 1 + 10 + 5,
        "P2-remote": 1 + (1 + 1 / bandwidth) + 4 + 5,
    }


def transfer_bound(
    payload: float, setup: float, links: tuple[float, ...], shared_capacity: float
) -> float:
    """Serial setup followed by data limited by links and shared source."""
    if payload < 0 or setup < 0 or not links or min(links) <= 0:
        raise ValueError("invalid transfer parameters")
    if shared_capacity <= 0:
        raise ValueError("shared capacity must be positive")
    return setup + payload / min(sum(links), shared_capacity)


def choose_route(
    costs: dict[str, float], available: set[str], limit: float = 15
) -> str | None:
    """Choose a predicted eligible route with a stable identifier tie break."""
    eligible = [
        (cost, name)
        for name, cost in costs.items()
        if name in available and 0 <= cost <= limit
    ]
    return min(eligible)[1] if eligible else None


def retention_value(
    reuses: int, recompute: float, fetch: float, write: float, manage: float
) -> float:
    """Additive forecast benefit in the common units of the supplied costs."""
    if min(reuses, recompute, fetch, write, manage) < 0:
        raise ValueError("negative cost or count")
    return reuses * (recompute - fetch) - write - manage


def forecast(
    current: int, events: tuple[tuple[float, int], ...], horizon: float
) -> int:
    """Events in (0,horizon]; equal-time order is the given order."""
    if current < 0 or horizon < 0:
        raise ValueError("negative state or horizon")
    state = current
    for time, delta in sorted(events, key=lambda event: event[0]):
        if 0 < time <= horizon:
            state += delta
            if state < 0:
                raise ValueError("impossible completion ledger")
    return state


def admit(
    forecast_count: int,
    service_limit: int,
    capacity: int,
    reserved: int,
    margin: int,
    request_bound: int,
) -> bool:
    """Evaluate service and byte guards under serialized reservation grants."""
    if (
        min(forecast_count, service_limit, capacity, reserved, margin, request_bound)
        < 0
    ):
        raise ValueError("negative admission parameter")
    return (
        forecast_count + 1 <= service_limit
        and reserved + margin + request_bound <= capacity
    )


@dataclass(frozen=True)
class ReceivedObject:
    """Observed state for one expected object at the publication gate."""

    identity: str
    expected_generation: int
    current_generation: int
    complete: bool
    compatible: bool
    protected_until: float
    quiescent: bool


def can_publish(
    expected: set[str], received: tuple[ReceivedObject, ...], now: float
) -> bool:
    """Evaluate under the single owner's lock; caller retains protection."""
    if now < 0 or len({item.identity for item in received}) != len(received):
        return False
    by_id = {item.identity: item for item in received}
    if not expected.issubset(by_id):
        return False
    return all(
        item.complete
        and item.compatible
        and item.quiescent
        and item.expected_generation == item.current_generation
        and item.protected_until > now
        for item in (by_id[key] for key in expected)
    )


def can_retire(outstanding_writes: int, readers: int, recovery_needed: bool) -> bool:
    """Permit reuse only after physical writes, readers and recovery release."""
    if outstanding_writes < 0 or readers < 0:
        raise ValueError("negative lifetime count")
    return outstanding_writes == 0 and readers == 0 and not recovery_needed


def goodput(
    emissions: tuple[tuple[float, ...], ...],
    completed: tuple[bool, ...],
    observation_ms: float,
) -> float:
    """Arrival zero; successful completion at final emission; no later drain."""
    if observation_ms <= 0 or len(emissions) != len(completed):
        raise ValueError("invalid observation population")
    count = 0
    for times, done in zip(emissions, completed, strict=True):
        if any(time < 0 for time in times) or tuple(sorted(times)) != times:
            raise ValueError("invalid emission order")
        if done and times and times[-1] <= observation_ms and times[0] <= 15:
            count += all(b - a <= 5 for a, b in pairwise(times))
    return count / (observation_ms / 1000)
