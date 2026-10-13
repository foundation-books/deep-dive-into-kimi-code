"""Constructed Chapter 39 physical manifest; units are assigned, not bytes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


DOMAINS = ("P0", "P1", "H0", "H1")
FIXED = (18, 18, 5, 5)
CAPACITIES = (32, 32, 16, 16)
CORE = {(0, 0): "A", (1, 0): "B", (2, 0): "C", (0, 1): "D"}
INPUTS = (Fraction(1, 3), Fraction(-7, 16), Fraction(9, 8), Fraction(-1, 16))


@dataclass(frozen=True)
class Event:
    """One atomic operation; zero deltas still carry dependency witnesses."""

    number: int
    object: str
    action: str
    domain: str
    units: int
    prerequisite: str
    resource: str = "--"
    slack: int | None = None
    justification: str = "--"


@dataclass(frozen=True)
class Trace:
    """All post-operation occupancies, including the initial reservations."""

    events: tuple[Event, ...]
    totals: tuple[tuple[int, ...], ...]
    peaks: tuple[int, ...]
    activation_traffic: int
    replay_work: int


class Manifest:
    """Build individual allocations, handoffs, readiness, and final uses."""

    def __init__(self, compressed: bool) -> None:
        self.compressed = compressed
        self.events: list[Event] = []
        self.live: dict[tuple[str, str], int] = {}
        self.done: set[str] = {"fixed"}
        self.traffic = 0
        self.forward_ready = "fixed"

    def emit(
        self,
        obj: str,
        action: str,
        domain: str = "P0",
        units: int = 0,
        prerequisite: str = "fixed",
        resource: str = "--",
        slack: int | None = None,
        justification: str = "--",
    ) -> str:
        if prerequisite == "fixed" and self.events:
            prerequisite = f"e{len(self.events)}"
        token = f"e{len(self.events) + 1}"
        self.events.append(
            Event(
                len(self.events) + 1,
                obj,
                action,
                domain,
                units,
                prerequisite,
                resource,
                slack,
                justification,
            )
        )
        self.done.add(token)
        return token

    def allocate(self, obj: str, domain: str, units: int, why: str) -> str:
        self.live[(obj, domain)] = units
        return self.emit(obj, "allocate", domain, units, why)

    def release(self, obj: str, domain: str, why: str) -> str:
        units = self.live.pop((obj, domain))
        return self.emit(obj, "release", domain, -units, why, justification=why)

    def handoff(self, obj: str, source: str, target: str, units: int) -> str:
        """Direct receive and reserved one-unit staging; one serial link."""
        launch = self.allocate(obj, target, units, f"{obj}@{source}")
        ready = self.emit(
            obj, "copy ready", target, prerequisite=launch, resource="L", slack=1
        )
        self.release(obj, source, ready)
        self.traffic += units
        return ready

    def core_forward(self, name: str, p: int) -> None:
        domain = f"P{p}"
        seed = self.allocate(f"{name}.s", domain, 2, self.forward_ready)
        z = self.emit(
            f"{name}.z",
            "produce 3 in arena; use; release",
            domain,
            prerequisite=seed,
            resource="arena:3+1",
        )
        raw = self.emit(
            f"{name}.x",
            "produce original 4 in arena",
            domain,
            prerequisite=z,
            resource="arena:4",
        )
        size = 2 if self.compressed else 4
        x = self.allocate(f"{name}.x", domain, size, raw)
        self.emit(
            f"{name}.x",
            "encode/copy; release arena original",
            domain,
            prerequisite=x,
            resource="metadata:1 reserved",
        )
        if not self.compressed and name in ("C", "D"):
            self.handoff(f"{name}.x", domain, f"H{p}", 4)
        weighted = self.emit(
            f"{name}.w",
            "produce 2 in arena; use; eliminate",
            domain,
            prerequisite=x,
            resource="arena:2",
        )
        self.allocate(f"{name}.b", domain, 2, weighted)
        if name in ("A", "B"):
            self.handoff(f"{name}.x", "P0", "P1", size)

    def region_forward(self, name: str, p: int) -> None:
        domain = f"P{p}"
        produced = self.emit(
            name,
            "forward region in arena; seed view 1",
            domain,
            prerequisite=self.forward_ready,
            resource="arena:3+1",
        )
        allocated = self.allocate(f"{name}.s", f"H{p}", 1, produced)
        self.emit(
            name,
            "host seed ready; release arena view",
            domain,
            prerequisite=allocated,
            resource="L",
            slack=1,
        )
        self.traffic += 1

    def region_backward(self, name: str, p: int) -> None:
        domain = f"P{p}"
        restored = self.emit(
            name,
            "restore seed into arena view 1",
            domain,
            prerequisite=f"{name}.s@H{p}",
            resource="L",
            slack=1,
        )
        replayed = self.emit(
            name,
            "replay output 3 + work 1; consume gradient",
            domain,
            prerequisite=restored,
            resource="arena:3+1",
        )
        self.emit(
            name,
            "release arena; accumulate into reserved gradient",
            domain,
            prerequisite=replayed,
            resource="gradient:4 reserved",
        )
        self.release(f"{name}.s", f"H{p}", replayed)
        self.traffic += 1

    def block_last_uses(self, name: str, p: int) -> None:
        domain = f"P{p}"
        if p == 0:
            launch = self.allocate(f"{name}.b.copy", "P1", 2, f"{name}.b@P0")
            ready = self.emit(
                f"{name}.b.copy",
                "send complete; close send ref",
                "P1",
                prerequisite=launch,
                resource="L",
                slack=0,
            )
            last = self.emit(
                f"{name}.b.copy", "two downstream last uses", "P1", prerequisite=ready
            )
            self.release(f"{name}.b.copy", "P1", last)
        local = self.emit(
            f"{name}.b",
            "local AttnRes last use",
            domain,
            prerequisite=f"{name}.b@{domain}",
        )
        self.release(f"{name}.b", domain, local)

    def core_backward(self, name: str, p: int) -> None:
        domain = f"P{p}"
        # This consumer depends on the retained block, not on the missing x.
        self.block_last_uses(name, p)
        size = 2 if self.compressed else 4
        if name in ("A", "B"):
            self.handoff(f"{name}.x", "P1", "P0", size)
        elif not self.compressed:
            self.handoff(f"{name}.x", f"H{p}", domain, 4)
        ready = self.emit(
            f"{name}.x",
            "validate; decode if compressed; MoE consumers",
            domain,
            prerequisite=f"{name}.x@{domain}",
            resource="arena:4" if self.compressed else "arena:work",
        )
        self.release(f"{name}.x", domain, ready)
        replayed = self.emit(
            f"{name}.z",
            "replay 3 + work 1; elementwise backward",
            domain,
            prerequisite=f"{name}.s@{domain}",
            resource="arena:3+1",
        )
        self.emit(
            name,
            "release replay; accumulate reserved gradient",
            domain,
            prerequisite=replayed,
            resource="gradient:4 reserved",
        )
        self.release(f"{name}.s", domain, replayed)

    def optimizer(self) -> None:
        """Per-pipeline slice; identical DP profiles, distinct DP identities."""
        for p in (0, 1):
            domain, host = f"P{p}", f"H{p}"
            for q in (0, 1):
                input_ready = self.emit(
                    f"G{p}.{q}",
                    "8-coordinate RS input view (2 units)",
                    domain,
                    resource="gradient:4 reserved",
                )
                self.allocate(f"O{p}.{q}", domain, 1, input_ready)
                self.emit(
                    f"G{p}.{q}",
                    "reduce-scatter scratch; input slice reusable",
                    domain,
                    resource="arena:1; DP link",
                )
            # Both outputs coexist; epoch 0 drains while epoch 1 is produced.
            for q in (0, 1):
                ready = self.handoff(f"O{p}.{q}", domain, host, 1)
                self.emit(
                    f"O{p}.{q}", "once-only CPU receipt", host, prerequisite=ready
                )
            self.emit(
                f"G{p}",
                "all 8 owner coordinates ready",
                host,
                resource="CPU gradient:2",
            )
        for matrix in (0, 1):
            for p in (0, 1):
                domain = f"P{p}"
                candidate_ready = self.emit(
                    f"M{matrix}",
                    "candidate 8 local + 8 received in arena 4",
                    domain,
                    resource=f"DP owner D{matrix}; L",
                )
                gram = self.allocate(f"M{matrix}.gram", domain, 4, candidate_ready)
                cubic = self.allocate(f"M{matrix}.cubic", domain, 4, gram)
                self.release(f"M{matrix}.gram", domain, cubic)
                transformed = self.emit(
                    f"M{matrix}",
                    "finite map overwrites arena candidate",
                    domain,
                    prerequisite=cubic,
                    resource="arena:4",
                )
                self.release(f"M{matrix}.cubic", domain, transformed)
                self.emit(
                    f"M{matrix}",
                    "deliver nonlocal direction 8; update both row owners",
                    domain,
                    prerequisite=transformed,
                    resource="L; receive aliases retired gradient",
                )
                publication_ready = self.emit(
                    f"M{matrix}",
                    "publish updated rows 8 each way; release assembly",
                    domain,
                    resource="L; parameters:8 reserved",
                )
        for p in (0, 1):
            for q in (0, 1):
                self.release(f"O{p}.{q}", f"H{p}", publication_ready)


def finite_manifest(compressed: bool = False) -> tuple[Event, ...]:
    """All sixteen region records in forward and reverse dependency order."""
    builder = Manifest(compressed)
    for microbatch in range(4):
        for layer in range(4):
            name = CORE.get((microbatch, layer), f"R{microbatch}{layer}")
            p = layer % 2
            if name in "ABCD":
                builder.core_forward(name, p)
            else:
                builder.region_forward(name, p)
            if layer < 3:
                builder.forward_ready = builder.emit(
                    name,
                    "next-stage output handoff; ready before next region",
                    f"P{p}",
                    resource="L; arena output/input aliases",
                    slack=1,
                )
    builder.emit("ABCD", "common frontier", resource="all links idle")
    for microbatch in reversed(range(4)):
        for layer in reversed(range(4)):
            name = CORE.get((microbatch, layer), f"R{microbatch}{layer}")
            if name in "ABCD":
                builder.core_backward(name, layer % 2)
            else:
                builder.region_backward(name, layer % 2)
    builder.optimizer()
    if builder.live:
        raise ValueError(f"unreleased allocations: {builder.live}")
    return tuple(builder.events)


def sweep_manifest(events: tuple[Event, ...], traffic: int = 0) -> Trace:
    """Reject a lost input, circular release, wrong ready event, or excess."""
    live: dict[tuple[str, str], int] = {}
    ready: set[str] = {"fixed"}
    values = list(FIXED)
    peaks = list(FIXED)
    totals: list[tuple[int, ...]] = []
    for e in events:
        requirement = e.prerequisite
        if requirement not in ready:
            raise ValueError(f"unready prerequisite at e{e.number}: {requirement}")
        key = (e.object, e.domain)
        if e.units > 0:
            if key in live:
                raise ValueError("duplicate allocation")
            live[key] = e.units
            ready.add(f"{e.object}@{e.domain}")
        elif e.units < 0:
            if live.pop(key, None) != -e.units:
                raise ValueError(f"missing object release at e{e.number}")
            ready.discard(f"{e.object}@{e.domain}")
        index = DOMAINS.index(e.domain)
        values[index] += e.units
        peaks = [max(a, b) for a, b in zip(peaks, values, strict=True)]
        if any(a > b for a, b in zip(values, CAPACITIES, strict=True)):
            raise ValueError(f"capacity at e{e.number}: {values}")
        if e.slack is not None and e.slack < 0:
            raise ValueError("negative slack")
        ready.add(f"e{e.number}")
        totals.append(tuple(values))
    if live:
        raise ValueError(f"missing final consumers: {live}")
    replay_work = sum(3 for e in events if e.action.startswith("replay"))
    return Trace(events, tuple(totals), tuple(peaks), traffic, replay_work)


def complete_trace(compressed: bool = False) -> Trace:
    """Policy traffic excludes invariant pipeline and summary transfers."""
    events = finite_manifest(compressed)
    traffic = sum(
        e.units
        for e in events
        if e.units > 0
        and (
            (e.object.endswith(".x") and "@" in e.prerequisite)
            or (e.object.startswith("R") and e.object.endswith(".s"))
        )
    ) + sum(1 for e in events if e.action == "restore seed into arena view 1")
    return sweep_manifest(events, traffic)
