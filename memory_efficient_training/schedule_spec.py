"""Independent expansion of the printed compact schedule's physical events.

This is an assigned-unit accounting oracle, not a trainer or a timing model.
The original finite_fixture manifest remains the identity/readiness verifier.
Comparing every event projection checks that shortening changed representation,
not the fixture's allocation, release, alias, or invocation order.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    """An object/domain delta; zero denotes work inside a charged reservation."""

    object: str
    domain: str
    units: int = 0


@dataclass(frozen=True)
class Block:
    """One printed family invocation with its ordered physical prefix."""

    name: str
    operations: tuple[Operation, ...]

    def accounting(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return net deltas and maximum prefix increments in P0/P1/H0/H1."""
        domains = ("P0", "P1", "H0", "H1")
        net, peak = [0] * 4, [0] * 4
        for operation in self.operations:
            g = domains.index(operation.domain)
            net[g] += operation.units
            peak[g] = max(peak[g], net[g])
        return tuple(net), tuple(peak)


class _Expansion:
    """Build physical deltas from the printed family rules."""

    def __init__(self, compressed: bool) -> None:
        self.compressed = compressed
        self.q = 2 if compressed else 4
        self.blocks: list[Block] = []
        self.ops: list[Operation] = []
        self.retained: dict[tuple[str, str], int] = {}

    def work(self, obj: str, domain: str) -> None:
        self.ops.append(Operation(obj, domain))

    def allocate(self, obj: str, domain: str, size: int) -> None:
        if (obj, domain) in self.retained:
            raise ValueError("duplicate physical allocation")
        self.retained[obj, domain] = size
        self.ops.append(Operation(obj, domain, size))

    def release(self, obj: str, domain: str) -> None:
        self.ops.append(Operation(obj, domain, -self.retained.pop((obj, domain))))

    def transfer(self, obj: str, source: str, target: str, size: int) -> None:
        self.allocate(obj, target, size)
        self.work(obj, target)
        self.release(obj, source)

    def close(self, name: str) -> None:
        self.blocks.append(Block(name, tuple(self.ops)))
        self.ops.clear()

    def forward(self, name: str, k: int) -> None:
        device, host = f"P{k % 2}", f"H{k % 2}"
        if name in ("A", "B", "C", "D"):
            self.allocate(f"{name}.s", device, 2)
            self.work(f"{name}.z", device)
            self.work(f"{name}.x", device)
            self.allocate(f"{name}.x", device, self.q)
            self.work(f"{name}.x", device)
            if not self.compressed and name in ("C", "D"):
                self.transfer(f"{name}.x", device, host, self.q)
            self.work(f"{name}.w", device)
            self.allocate(f"{name}.b", device, 2)
            if name in ("A", "B"):
                self.transfer(f"{name}.x", "P0", "P1", self.q)
        else:
            self.work(name, device)
            self.allocate(f"{name}.s", host, 1)
            self.work(name, device)
        if k < 3:
            self.work(name, device)
        self.close(f"F({name})")

    def backward(self, name: str, k: int) -> None:
        device, host = f"P{k % 2}", f"H{k % 2}"
        if name in ("A", "B", "C", "D"):
            if k % 2 == 0:
                self.allocate(f"{name}.b.copy", "P1", 2)
                self.work(f"{name}.b.copy", "P1")
                self.work(f"{name}.b.copy", "P1")
                self.release(f"{name}.b.copy", "P1")
            self.work(f"{name}.b", device)
            self.release(f"{name}.b", device)
            if name in ("A", "B"):
                self.transfer(f"{name}.x", "P1", "P0", self.q)
            elif not self.compressed:
                self.transfer(f"{name}.x", host, device, self.q)
            self.work(f"{name}.x", device)
            self.release(f"{name}.x", device)
            self.work(f"{name}.z", device)
            self.work(name, device)
            self.release(f"{name}.s", device)
        else:
            for _ in range(3):
                self.work(name, device)
            self.release(f"{name}.s", host)
        self.close(f"B({name})")

    def gradient(self, p: int) -> None:
        device, host = f"P{p}", f"H{p}"
        for j in (0, 1):
            self.work(f"G{p}.{j}", device)
            self.allocate(f"O{p}.{j}", device, 1)
            self.work(f"G{p}.{j}", device)
        for j in (0, 1):
            self.transfer(f"O{p}.{j}", device, host, 1)
            self.work(f"O{p}.{j}", host)
        self.work(f"G{p}", host)
        self.close(f"G{p}")

    def muon(self, i: int, p: int) -> None:
        device, name = f"P{p}", f"M{i}"
        self.work(name, device)
        self.allocate(f"{name}.gram", device, 4)
        self.allocate(f"{name}.cubic", device, 4)
        self.release(f"{name}.gram", device)
        self.work(name, device)
        self.release(f"{name}.cubic", device)
        self.work(name, device)
        self.work(name, device)
        self.close(f"M{i}(P{p},D{i})")


def schedule_blocks(compressed: bool = False) -> tuple[Block, ...]:
    """Expand the manuscript's F, U, B, reduction, and Muon families."""
    expansion = _Expansion(compressed)
    highlighted = {(0, 0): "A", (0, 1): "D", (1, 0): "B", (2, 0): "C"}
    for m in range(4):
        for k in range(4):
            expansion.forward(highlighted.get((m, k), f"R{m}{k}"), k)
    expansion.work("ABCD", "P0")
    expansion.close("frontier")
    for m in reversed(range(4)):
        for k in reversed(range(4)):
            expansion.backward(highlighted.get((m, k), f"R{m}{k}"), k)
    for p in (0, 1):
        expansion.gradient(p)
    for i in (0, 1):
        for p in (0, 1):
            expansion.muon(i, p)
    for p in (0, 1):
        for j in (0, 1):
            expansion.release(f"O{p}.{j}", f"H{p}")
    expansion.close("final host release")
    if expansion.retained:
        raise ValueError(f"unreleased objects: {expansion.retained}")
    return tuple(expansion.blocks)


def resource_sweep(compressed: bool = False) -> tuple[int, ...]:
    """Compose family prefix bounds including fixed reservations."""
    occupancy, peaks = [18, 18, 5, 5], [18, 18, 5, 5]
    for block in schedule_blocks(compressed):
        net, increment = block.accounting()
        for g in range(4):
            peaks[g] = max(peaks[g], occupancy[g] + increment[g])
            occupancy[g] += net[g]
    if occupancy != [18, 18, 5, 5]:
        raise ValueError("final reservation mismatch")
    return tuple(peaks)
