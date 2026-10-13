"""Named negative mutants used by Chapter 38 checks."""

from __future__ import annotations

from dataclasses import replace

from .objects import ObjectKey
from .specialized import Matrix


def wrong_microbatch(key: ObjectKey) -> ObjectKey:
    """Return a key whose microbatch version is intentionally stale."""
    return replace(key, microbatch=key.microbatch + 1)


def wrong_scale(scale: str) -> str:
    """Return an intentionally stale representation-scale marker."""
    return f"stale:{scale}"


def wrong_route_order(routes: tuple[int, ...]) -> tuple[int, ...]:
    """Reverse route order to model plan-identity corruption."""
    return tuple(reversed(routes))


def duplicate_epoch(values: tuple[int, ...]) -> tuple[int, ...]:
    """Double one epoch's values to model duplicate accumulation."""
    return tuple(value * 2 for value in values)


def wrong_shard_order(shards: tuple[Matrix, ...]) -> tuple[Matrix, ...]:
    """Reverse matrix shards to violate their canonical row order."""
    return tuple(reversed(shards))
