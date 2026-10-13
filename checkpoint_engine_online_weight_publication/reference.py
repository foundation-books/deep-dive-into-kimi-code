"""Exact coordinates and a synchronous teaching publication state machine.

No transport, GPU, persistent store, or production controller is implemented.
Epoch authorization and device completion are explicit inputs to this model.
"""

from dataclasses import dataclass, field

Coordinate = tuple[str, int, int]
Values = dict[Coordinate, int]


def snapshot(version: int) -> Values:
    """Create the 28 exact scalar values from the manuscript."""
    values = {
        ("W", i, j): 100 * version + 4 * i + j for i in range(4) for j in range(4)
    }
    values.update({("b", i, 0): 100 * version + 20 + i for i in range(4)})
    for expert in range(2):
        values.update(
            {
                (f"E{expert}", i, j): 100 * version + 30 + 4 * expert + 2 * i + j
                for i in range(2)
                for j in range(2)
            }
        )
    return values


def source_owner(coord: Coordinate) -> int:
    """Return the fixture's authoritative trainer rank."""
    name, row, _ = coord
    return row // 2 if name in ("W", "b") else int(name[1])


def destination_owns(worker: int, coord: Coordinate) -> bool:
    """Identify destination columns, replicated bias, and expert ownership."""
    name, _, col = coord
    return name == "b" or (name == "W" and col // 2 == worker) or name == f"E{worker}"


def reconstruct(sources: list[Values], worker: int, expected: Values) -> Values:
    """Reject conflicting duplicates, omissions, and wrong values."""
    result: Values = {}
    for source in sources:
        for coord, value in source.items():
            if coord not in expected:
                raise ValueError("unknown coordinate")
            if destination_owns(worker, coord):
                if coord in result and result[coord] != value:
                    raise ValueError("conflicting duplicate")
                result[coord] = value
    target = {c: v for c, v in expected.items() if destination_owns(worker, c)}
    if result != target:
        raise ValueError("incomplete or incorrect destination")
    return result


def pipeline_time(chunks: int, transfer: int, install: int) -> int:
    """Simulate independent serial stages with sufficient buffering."""
    if min(chunks, transfer, install) <= 0:
        raise ValueError("positive chunk count and durations required")
    complete = 0
    for k in range(1, chunks + 1):
        complete = max(k * transfer, complete) + install
    return complete


def peak_bytes(allocations: dict[str, tuple[int, int, int]]) -> int:
    """Count unique physical allocations at half-open live intervals."""
    for size, start, end in allocations.values():
        if size < 0 or end <= start:
            raise ValueError("invalid allocation")
    events = {time for _, start, end in allocations.values() for time in (start, end)}
    return max(
        (
            sum(
                size for size, start, end in allocations.values() if start <= time < end
            )
            for time in events
        ),
        default=0,
    )


@dataclass
class Replica:
    """Whole-replica model; installation covers both workers atomically here."""

    epoch: int = 3
    active: int = 7
    handles: dict[int, Values] = field(default_factory=lambda: {7: snapshot(7)})
    readers: dict[int, int] = field(default_factory=dict)
    attempt: tuple[int, int, int] | None = None
    staged: Values = field(default_factory=dict)
    installed: bool = False
    admitted: bool = True
    decision: tuple[int, int, int] | None = None

    def begin(self, epoch: int, version: int, attempt: int) -> None:
        if epoch != self.epoch or version <= self.active:
            raise ValueError("stale epoch or nonforward version")
        token = (epoch, version, attempt)
        if self.attempt == token:
            return
        if self.attempt and attempt <= self.attempt[2]:
            raise ValueError("superseded attempt")
        self.attempt, self.staged, self.installed = token, {}, False

    def chunk(self, token: tuple[int, int, int], values: Values) -> None:
        if token != self.attempt or token[0] != self.epoch:
            raise ValueError("unauthorized attempt")
        if self.installed:
            if any(self.staged.get(c) != v for c, v in values.items()):
                raise ValueError("mutation after installation")
            return
        expected = snapshot(token[1])
        for coord, value in values.items():
            if coord not in expected or expected[coord] != value:
                raise ValueError("invalid content")
        self.staged.update(values)

    def install(self, token: tuple[int, int, int], device_complete: bool) -> None:
        if token != self.attempt or token[0] != self.epoch:
            raise ValueError("unauthorized installation")
        if not device_complete or self.staged != snapshot(token[1]):
            raise ValueError("incomplete installation")
        self.handles[token[1]] = dict(self.staged)
        self.installed = True

    def activate(self, token: tuple[int, int, int]) -> int:
        if token[0] != self.epoch:
            raise ValueError("fenced epoch")
        if token == self.decision and token[1] == self.active and self.admitted:
            return self.active
        if token != self.attempt or not self.installed:
            raise ValueError("not installed")
        self.decision, self.active, self.admitted = token, token[1], True
        return self.active

    def acquire(self) -> int:
        if not self.admitted or self.active not in self.handles:
            raise ValueError("admission closed")
        self.readers[self.active] = self.readers.get(self.active, 0) + 1
        return self.active

    def read(self, handle: int, coord: Coordinate) -> int:
        if self.readers.get(handle, 0) <= 0:
            raise ValueError("request has no reference")
        return self.handles[handle][coord]

    def release(self, handle: int) -> None:
        if self.readers.get(handle, 0) <= 0:
            raise ValueError("unowned handle")
        self.readers[handle] -= 1

    def retire(self, version: int, device_complete: bool) -> None:
        if (
            version == self.active
            or self.readers.get(version, 0)
            or not device_complete
        ):
            raise ValueError("live allocation")
        self.handles.pop(version, None)

    def restart(self, new_epoch: int) -> None:
        if new_epoch <= self.epoch:
            raise ValueError("epoch must advance")
        self.epoch, self.admitted = new_epoch, False
        self.handles, self.readers, self.staged = {}, {}, {}
        self.attempt, self.installed = None, False
        # Active is reset below the target so begin can rebuild the logged version.
        self.active = -1
