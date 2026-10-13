"""Layer and depth schedules for the integrated K3 oracle."""

from __future__ import annotations

from typing import Sequence


def terminal_global_schedule(groups: int) -> tuple[str, ...]:
    """Return `(K,K,K,G)^groups,G`."""

    if groups <= 0:
        raise ValueError("at least one complete hybrid group is required")
    return ("K", "K", "K", "G") * groups + ("G",)


def validate_schedule(schedule: Sequence[str]) -> tuple[str, ...]:
    """Validate terminal-global order, not merely K/G counts."""

    typed = tuple(schedule)
    if len(typed) < 5 or (len(typed) - 1) % 4:
        raise ValueError("schedule must contain complete four-layer groups and a tail")
    if typed[-1] != "G":
        raise ValueError("schedule must end in an additional global layer")
    for start in range(0, len(typed) - 1, 4):
        if typed[start : start + 4] != ("K", "K", "K", "G"):
            raise ValueError("hybrid groups must appear in K,K,K,G order")
    return typed


def channel_schedule(layers: int) -> tuple[str, ...]:
    """Return one dense channel layer followed by routed layers."""

    if layers <= 0:
        raise ValueError("layer count must be positive")
    return ("D",) + ("M",) * (layers - 1)


def depth_partition(layers: int, block_size: int) -> tuple[int, ...]:
    """Partition decoder layers, retaining a nonempty partial tail."""

    if layers <= 0 or block_size <= 0:
        raise ValueError("layer and block sizes must be positive")
    complete, tail = divmod(layers, block_size)
    return (block_size,) * complete + ((tail,) if tail else ())


def full_scale_schedule_summary() -> dict[str, object]:
    """Return exact structural deductions for the pinned K3 geometry."""

    schedule = validate_schedule(terminal_global_schedule(23))
    partition = depth_partition(len(schedule), 12)
    return {
        "layers": len(schedule),
        "kda_layers": schedule.count("K"),
        "global_layers": schedule.count("G"),
        "attention_schedule": schedule,
        "channel_schedule": channel_schedule(len(schedule)),
        "depth_partition": partition,
        "transformation_points": 2 * len(schedule),
        "terminal_sources": 1 + len(partition),
    }


__all__ = [
    "channel_schedule",
    "depth_partition",
    "full_scale_schedule_summary",
    "terminal_global_schedule",
    "validate_schedule",
]
