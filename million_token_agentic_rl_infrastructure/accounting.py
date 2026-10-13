"""Physical allocation events for the chapter's phase-boundary ledger."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationEvent:
    """Create a unique tier-local allocation, or retire it with size None."""

    tier: str
    identity: str
    size: int | None


def occupancy_trace(events: tuple[AllocationEvent, ...]) -> tuple[dict[str, int], ...]:
    """Count live unique bytes after each event; reject premature/double bookkeeping."""
    live: dict[tuple[str, str], int] = {}
    tiers: set[str] = set()
    snapshots = []
    for event in events:
        key = (event.tier, event.identity)
        tiers.add(event.tier)
        if event.size is None:
            if key not in live:
                raise ValueError("retiring an allocation that is not live")
            del live[key]
        else:
            if event.size < 0 or key in live:
                raise ValueError("negative size or duplicate physical identity")
            live[key] = event.size
        snapshots.append(
            {
                tier: sum(size for (owner, _), size in live.items() if owner == tier)
                for tier in sorted(tiers)
            }
        )
    return tuple(snapshots)
