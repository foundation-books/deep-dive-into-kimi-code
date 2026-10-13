"""Slotted finite fixture: service outcomes, not hardware timing predictions."""

from dataclasses import dataclass

from .ledger import Contract, GrantKey, Owner
from .reference import SECONDARIES, WEIGHTS


@dataclass
class Request:
    """One declared replay request and its two service endpoints."""

    name: str
    destination: str
    replay: int
    restored: int | None = None
    finished: int | None = None


def _require(accepted: bool) -> None:
    """Fixture transitions must execute even when Python assertions are disabled."""
    if not accepted:
        raise RuntimeError("fixture transition rejected")


def _key(request: str) -> GrantKey:
    return request + ".r1", 1 if request.startswith("S") else 2, 6, "memory"


def _admit(owner: Owner, name: str, contract: Contract) -> None:
    _require(owner.prepare(_key(name), 1, contract=contract))
    _require(owner.commit(_key(name)))
    _require(owner.allocate(_key(name), name, 1))


def _finish(
    owner: Owner, name: str, slot: int, time: int, cls: str, work: tuple[int, int]
) -> None:
    _require(owner.dispatch(_key(name), "suffix", slot, cls, 0, work[0]))
    _require(owner.dispatch(_key(name), "decode", slot, cls, 1, work[1]))
    _require(owner.publish(_key(name), 0, "It", now=time))
    owner.deliver(name + ".r1", 0)
    owner.retire(_key(name))
    _require(owner.release(_key(name), quiescent=True))


def _replay(
    requests: list[Request], dest: str, owner: Owner, slot: int, time: int
) -> list[tuple[str, int, str]]:
    remaining = 20
    result = []
    for request in requests:
        if request.destination != dest or not request.replay:
            continue
        served = 0
        while request.replay and remaining:
            amount = min(10, request.replay, remaining)
            quantum = f"replay-{slot}-{served}"
            _require(owner.dispatch(_key(request.name), quantum, slot, "R", 0, amount))
            request.replay -= amount
            remaining -= amount
            served += amount
        if served:
            if request.replay == 0:
                request.restored = time
            result.append(
                (
                    request.name,
                    served,
                    "restored" if request.replay == 0 else "replaying",
                )
            )
    return result


def _record(
    events: list[dict[str, object]],
    owners: dict[str, Owner],
    time: int,
    request: str,
    dest: str,
    cls: str,
    work: tuple[int, int, int],
    status: str,
) -> None:
    live, reserved = owners[dest].occupancy()
    attempt = 1 if request.startswith("S") else 2
    is_short = request.startswith("S")
    arrival = 10 if is_short else -10
    prefix = 0 if is_short else WEIGHTS["ABCDEF".index(request)]
    events.append(
        dict(
            time=time,
            request=request + ".r1",
            attempt=attempt,
            epoch=6,
            membership=1,
            context="v7",
            destination=dest,
            cls=cls,
            work=work,
            live=live,
            reserved=reserved,
            status=status,
            request_age=time - arrival,
            admission_wait=0 if is_short else 20,
            saved_prefix=prefix,
            grant=_key(request),
            output_cap=1 if is_short else 4,
            allocation=request,
        )
    )


def fleet_trace() -> tuple[list[dict[str, object]], dict[str, Owner]]:
    """Failure at 0, authority ready at 10; completion credits at slot ends."""
    owners = {cluster: Owner(epoch=6) for cluster in ("C1", "C2", "C3")}
    requests = [
        Request(name, dest, weight)
        for name, dest, weight in zip("ABCDEF", SECONDARIES, WEIGHTS, strict=True)
    ]
    events: list[dict[str, object]] = []

    for request in requests:
        owner = owners[request.destination]
        _admit(
            owner,
            request.name,
            Contract((request.replay + 10, 4, 0), ("R", "L"), deadline=190),
        )
        _record(
            events,
            owners,
            10,
            request.name,
            request.destination,
            "R",
            (0, 0, 0),
            "admitted",
        )
    for dest, owner in owners.items():
        name = "S" + dest
        _admit(owner, name, Contract((2, 1, 0), ("S",), 1, 210))

    for time in range(20, 201, 10):
        slot = time // 10 - 1
        for dest, owner in owners.items():
            if time == 20:
                name = "S" + dest
                _finish(owner, name, slot, time, "S", (2, 1))
                _record(events, owners, time, name, dest, "S", (2, 1, 0), "complete")
            for request in requests:
                if request.destination != dest or request.finished is not None:
                    continue
                if request.restored is not None and request.restored < time:
                    _finish(owner, request.name, slot, time, "L", (10, 4))
                    request.finished = time
                    _record(
                        events,
                        owners,
                        time,
                        request.name,
                        dest,
                        "L",
                        (10, 4, 0),
                        "complete",
                    )
            for name, served, status in _replay(requests, dest, owner, slot, time):
                _record(events, owners, time, name, dest, "R", (served, 0, 0), status)
        if all(request.finished is not None for request in requests):
            return events, owners
    raise RuntimeError("finite fixture did not complete")


def fifo_slots(work: tuple[int, ...], allowance: int) -> tuple[int, ...]:
    """Completion slot indices with partial final quanta and protected service."""
    if allowance <= 0 or any(item <= 0 for item in work):
        raise ValueError("positive work and service required")
    remaining = list(work)
    completions = [0] * len(work)
    slot = 0
    while any(remaining):
        slot += 1
        budget = allowance
        for index, item in enumerate(remaining):
            if item == 0:
                continue
            amount = min(item, budget)
            remaining[index] -= amount
            budget -= amount
            if remaining[index] == 0:
                completions[index] = slot
    return tuple(completions)
