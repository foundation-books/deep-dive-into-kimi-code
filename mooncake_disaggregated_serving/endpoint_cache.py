"""Single-thread SIEVE trace with distinct lookup eviction and reclamation."""

from dataclasses import dataclass


@dataclass
class Endpoint:
    """A resource identity distinct from its reusable peer name."""

    name: str
    serial: int
    visited: bool = False


class Sieve:
    """Oldest-to-newest scanning equivalent to the pinned list's reverse walk.

    This model has no locks, QPs or DMA. Runtime hits do use synchronization.
    Reclamation requires explicitly supplied safe resource identities.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("positive capacity required")
        self.capacity = capacity
        self.entries: list[Endpoint] = []
        self.pending: list[Endpoint] = []
        self.hand: int | None = None
        self.next_serial = 0
        self.last_scans = 0

    def get(self, name: str) -> Endpoint | None:
        """Set the hit bit without moving the entry."""
        for entry in self.entries:
            if entry.name == name:
                entry.visited = True
                return entry
        return None

    def insert(self, name: str) -> Endpoint:
        """Reuse an existing endpoint or create a new unvisited entry."""
        present = self.get(name)
        if present is not None:
            return present
        if len(self.entries) == self.capacity:
            self.evict()
        entry = Endpoint(name, self.next_serial)
        self.next_serial += 1
        self.entries.append(entry)
        return entry

    def evict(self) -> Endpoint | None:
        """Clear visited bits while walking; remove the first unvisited entry."""
        self.last_scans = 0
        if not self.entries:
            return None
        index = next(
            (i for i, e in enumerate(self.entries) if e.serial == self.hand), 0
        )
        while self.entries[index].visited:
            self.last_scans += 1
            self.entries[index].visited = False
            index = (index + 1) % len(self.entries)
        self.last_scans += 1
        self.hand = (
            self.entries[(index + 1) % len(self.entries)].serial
            if len(self.entries) > 1
            else None
        )
        victim = self.entries.pop(index)
        self.pending.append(victim)
        return victim

    def reclaim(self, safe_serials: set[int]) -> tuple[int, ...]:
        """Destroy only already-evicted resources whose work is declared drained."""
        freed = tuple(e.serial for e in self.pending if e.serial in safe_serials)
        self.pending = [e for e in self.pending if e.serial not in safe_serials]
        return freed
