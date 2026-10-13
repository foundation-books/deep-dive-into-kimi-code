"""Capacity and registration accounting for an explicitly provisioned arena."""


class StagingPool:
    """Ideal unfragmented arena with bounded external leases and optional overflow.

    This is independent of dummy/real client topology. A failed acquisition
    returns None; the caller decides whether to wait or reject. No OS memory,
    registration, exported Python views or concurrent requests are implemented.
    """

    def __init__(self, capacity: int, max_bytes: int, internal: int = 0) -> None:
        if capacity <= 0 or max_bytes < 0 or not 0 <= internal <= capacity:
            raise ValueError("invalid arena or pressure bound")
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.internal = internal
        self.leases: dict[int, tuple[int, bool]] = {}
        self.registrations = 1
        self.unregistrations = 0
        self.next_id = 0
        self.peak_registered = capacity

    @property
    def leased(self) -> int:
        """All live external bytes, including overflow."""
        return sum(size for size, _ in self.leases.values())

    @property
    def arena_used(self) -> int:
        """Internal and external suballocations that occupy the shared arena."""
        return self.internal + sum(
            size for size, overflow in self.leases.values() if not overflow
        )

    @property
    def registered(self) -> int:
        """Reserved arena plus live separately registered overflow allocations."""
        return self.capacity + sum(
            size for size, overflow in self.leases.values() if overflow
        )

    def acquire(self, size: int, allow_overflow: bool = True) -> int | None:
        """Charge the full allocation to the arena or to a temporary overflow."""
        if size <= 0:
            raise ValueError("positive lease size required")
        if self.leased + size > self.max_bytes:
            return None
        overflow = self.arena_used + size > self.capacity
        if overflow and not allow_overflow:
            return None
        key = self.next_id
        self.next_id += 1
        self.leases[key] = (size, overflow)
        self.registrations += int(overflow)
        self.peak_registered = max(self.peak_registered, self.registered)
        return key

    def release(self, key: int) -> None:
        """Release a lease and immediately unregister its overflow, if any."""
        _, overflow = self.leases.pop(key)
        self.unregistrations += int(overflow)


def registration_costs(
    operations: int, overflow: int, pair_cost: float, suballoc_cost: float
) -> tuple[float, float]:
    """Full-lifecycle naive versus pooled cost, charging setup/teardown once."""
    if not 0 <= overflow <= operations or min(pair_cost, suballoc_cost) < 0:
        raise ValueError("invalid operation counts or costs")
    return operations * pair_cost, (
        1 + overflow
    ) * pair_cost + operations * suballoc_cost
