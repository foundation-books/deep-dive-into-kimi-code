"""Explicit single-owner bundle events, with device completion as an input."""

from __future__ import annotations

from dataclasses import dataclass, field

from .reference import Pool


@dataclass(frozen=True)
class Bundle:
    """Immutable descriptor for one complete, context-qualified boundary."""

    ident: str
    context: tuple[str, ...]
    prefix: tuple[str, ...]
    boundary: int
    generation: int
    groups: tuple[str, ...]
    handles: tuple[str, ...]


@dataclass
class BundleOwner:
    """Serialized metadata events and all-component leases."""

    pool: Pool
    expected_groups: tuple[str, ...]
    bundles: dict[str, Bundle] = field(default_factory=dict)
    visible: set[str] = field(default_factory=set)
    invalid: set[str] = field(default_factory=set)
    leases: dict[int, tuple[str, ...]] = field(default_factory=dict)
    next_lease: int = 0

    def __post_init__(self) -> None:
        if not self.expected_groups or len(set(self.expected_groups)) != len(
            self.expected_groups
        ):
            raise ValueError("a nonempty, unique model group set is required")

    def prepare(self, bundle: Bundle) -> None:
        if (
            bundle.ident in self.bundles
            or bundle.ident in self.invalid
            or not self._valid(bundle)
        ):
            raise ValueError("invalid descriptor")
        if set(bundle.handles) != self.pool.owned.get(bundle.ident):
            raise ValueError("bundle must own exactly its payload allocations")
        if any(h in self.pool.writers or self.pool.pins[h] for h in bundle.handles):
            raise ValueError("payload still owned")
        existing = {h for d in self.bundles.values() for h in d.handles}
        if existing.intersection(bundle.handles):
            raise ValueError("new writes require fresh slots")
        self.bundles[bundle.ident] = bundle
        self.pool.retained.add(bundle.ident)
        self.pool.writers.update(bundle.handles)

    def _valid(self, bundle: Bundle) -> bool:
        if bundle.boundary <= 0 or len(bundle.prefix) != bundle.boundary:
            return False
        if sorted(bundle.groups) != sorted(self.expected_groups):
            return False
        if len(set(bundle.handles)) != len(bundle.handles):
            return False
        if any(h not in self.pool.pages for h in bundle.handles):
            return False
        pages = [self.pool.pages[h] for h in bundle.handles]
        states = [p for p in pages if p.kind == "KDA"]
        if sorted(p.group for p in states) != sorted(bundle.groups):
            return False
        if any(p.boundary != bundle.boundary for p in states):
            return False
        ranges = sorted(p.initialized for p in pages if p.kind == "MLA")
        end = 0
        for lo, hi in ranges:
            if lo > end:
                return False
            end = max(end, hi)
        return end >= bundle.boundary

    def complete(self, ident: str, handle: str) -> None:
        if handle not in self.bundles[ident].handles:
            raise ValueError("foreign completion")
        self.pool.writers.discard(handle)

    def publish(self, ident: str) -> bool:
        bundle = self.bundles[ident]
        if ident in self.invalid or any(h in self.pool.writers for h in bundle.handles):
            return False
        if not self._valid(bundle):
            return False
        self.visible.add(ident)
        self.pool.retained.add(ident)
        return True

    def acquire(
        self,
        ident: str,
        generation: int,
        context: tuple[str, ...],
        prefix: tuple[str, ...],
    ) -> int | None:
        bundle = self.bundles.get(ident)
        if bundle is None or ident not in self.visible:
            return None
        if (bundle.generation, bundle.context, bundle.prefix) != (
            generation,
            context,
            prefix,
        ):
            return None
        if not self._valid(bundle):
            return None
        # No eviction or allocation may interleave with this owner operation.
        for handle in bundle.handles:
            self.pool.pins[handle] += 1
        lease = self.next_lease
        self.next_lease += 1
        self.leases[lease] = bundle.handles
        return lease

    def invalidate(self, ident: str) -> None:
        self.invalid.add(ident)
        self.visible.discard(ident)
        self.pool.retained.discard(ident)

    def release_lease(self, lease: int) -> None:
        for handle in self.leases.pop(lease, ()):
            self.pool.pins[handle] -= 1

    def retire(self, ident: str) -> bool:
        bundle = self.bundles[ident]
        if ident not in self.invalid:
            return False
        if any(self.pool.pins[h] or h in self.pool.writers for h in bundle.handles):
            return False
        if not self.pool.release(ident):
            return False
        del self.bundles[ident]
        return True
