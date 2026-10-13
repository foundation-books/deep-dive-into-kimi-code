"""Chapter 41's deterministic contracts, not a production runtime."""

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class Record:
    """A quiescent transcript/environment boundary with explicit provenance."""

    generation: int = 4
    tokens: tuple[str, ...] = ("p", "p", "call", "call", "obs", "obs", "r", "r")
    mask: tuple[int, ...] = (0, 0, 1, 1, 0, 0, 1, 1)
    versions: tuple[int | None, ...] = (None, None, 7, 7, None, None, 7, 7)
    log_probs: tuple[float | None, ...] = (
        None,
        None,
        -1.0,
        -2.0,
        None,
        None,
        -1.5,
        -0.5,
    )
    tool: str = "u1"
    observation_end: int = 6
    environment: str = "E8"
    environment_tool: str = "u1"
    sampler: str = "saved-stream/config-v1"
    in_flight: bool = False


def _valid_provenance(record: Record) -> bool:
    for mask, version, probability in zip(
        record.mask, record.versions, record.log_probs, strict=True
    ):
        if mask == 0:
            if version is not None or probability is not None:
                return False
        elif mask == 1:
            if version is None or version < 0 or probability is None:
                return False
            if not isfinite(probability) or probability > 0:
                return False
        else:
            return False
    return True


def valid_record(record: Record) -> bool:
    """Validate the finite fixture; checkpoint identity comes from trusted ancestry."""
    n = len(record.tokens)
    if not (n == len(record.mask) == len(record.versions) == len(record.log_probs)):
        return False
    if record.in_flight or not record.sampler or record.generation < 0:
        return False
    if (record.environment, record.environment_tool) != ("E8", record.tool):
        return False
    if record.tool != "u1" or record.observation_end != 6 or n < 8:
        return False
    if record.tokens[4:6] != ("obs", "obs"):
        return False
    if record.mask[:8] != Record().mask or record.versions[:8] != Record().versions:
        return False
    return _valid_provenance(record)


def continue_record(record: Record) -> Record:
    """Add four v8 generated tokens without replacing old behavior metadata."""
    if not valid_record(record):
        raise ValueError("invalid resume boundary")
    return replace(
        record,
        tokens=record.tokens + ("new",) * 4,
        mask=record.mask + (1,) * 4,
        versions=record.versions + (8,) * 4,
        log_probs=record.log_probs + (-0.75,) * 4,
    )


def ready_groups(groups: dict[str, tuple[bool, ...]], width: int = 4) -> set[str]:
    """Completeness only; provenance remains a separate learning gate."""
    return {
        name
        for name, attempts in groups.items()
        if len(attempts) == width and all(attempts)
    }


def cache_bytes(tokens: int) -> int:
    """Return one paged MLA prefix and one recurrent boundary in toy bytes."""
    if tokens <= 0:
        raise ValueError("positive prefix required")
    return 64 * ((tokens + 3) // 4) + 32


def scalar_state(coefficient: int, inputs: tuple[int, ...]) -> int:
    """Evaluate the scalar cross-version counterexample."""
    state = 0
    for value in inputs:
        state = coefficient * state + value
    return state


@dataclass(frozen=True)
class Cache:
    """A complete toy cache descriptor with trusted state verification."""

    version: int
    prefix: tuple[str, ...]
    boundary: int
    layout: str = "fixture-v1"
    groups: frozenset[str] = frozenset({"KDA", "MLA"})
    verified: bool = True


def compatible(cache: Cache, version: int, prefix: tuple[str, ...]) -> bool:
    """Check the baseline same-version compatibility predicate."""
    return (
        cache.version == version
        and cache.prefix == prefix
        and cache.boundary == len(prefix)
        and cache.layout == "fixture-v1"
        and cache.groups == frozenset({"KDA", "MLA"})
        and cache.verified
    )


def physical_bytes(views: tuple[tuple[str, int], ...]) -> int:
    """Allocation identities are tier-local; conflicting aliases are invalid."""
    allocations: dict[str, int] = {}
    for identity, size in views:
        if size < 0 or (identity in allocations and allocations[identity] != size):
            raise ValueError("inconsistent allocation")
        allocations[identity] = size
    return sum(allocations.values())


def can_admit(
    live: tuple[int, ...], queued: tuple[int, ...], capacity: int, margin: int
) -> bool:
    """Check total bounded reservations, including explicit safety margin."""
    if min((*live, *queued, capacity, margin)) < 0:
        raise ValueError("negative budget")
    return sum(live) + sum(queued) + margin <= capacity


@dataclass(frozen=True)
class Access:
    """A read or write lifetime on one physical slot."""

    slot: int
    start: int
    end: int
    write: bool


def safe_accesses(accesses: tuple[Access, ...]) -> bool:
    """Half-open device-event lifetimes; read/read overlap is permitted."""
    for index, left in enumerate(accesses):
        if left.start < 0 or left.end <= left.start:
            return False
        for right in accesses[index + 1 :]:
            conflict = left.write or right.write
            overlap = max(left.start, right.start) < min(left.end, right.end)
            if left.slot == right.slot and conflict and overlap:
                return False
    return True


def slot_schedule(last_reader: int = 5) -> tuple[Access, ...]:
    """Construct the three-chunk schedule, waiting for every slot reader."""
    third_copy = max(5, last_reader)
    third_forward = max(8, third_copy + 2)
    return (
        Access(0, 0, 2, True),
        Access(1, 2, 4, True),
        Access(0, 2, last_reader, False),
        Access(1, 5, 8, False),
        Access(0, third_copy, third_copy + 2, True),
        Access(0, third_forward, third_forward + 3, False),
    )


@dataclass(frozen=True)
class Delta:
    """An immutable parent identity and changed page payload."""

    identity: str
    parent: str
    pages: tuple[tuple[int, str], ...]


def restore(base: dict[int, str], deltas: tuple[Delta, ...]) -> dict[int, str]:
    """Apply a complete ordered delta chain without mutating its base."""
    pages = dict(base)
    parent = "base"
    seen = {parent}
    for delta in deltas:
        if delta.parent != parent or delta.identity in seen:
            raise ValueError("broken snapshot ancestry")
        if len(dict(delta.pages)) != len(delta.pages):
            raise ValueError("duplicate page in delta")
        pages.update(delta.pages)
        parent = delta.identity
        seen.add(parent)
    return pages
