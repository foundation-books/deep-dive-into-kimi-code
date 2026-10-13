"""Named negative mutants for Chapter 37's deterministic oracles."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .schedule import CollectiveKey
from .semantics import Route
from .topology import ProcessGroup


def duplicate_group_member(group: ProcessGroup) -> ProcessGroup:
    """Return a group mutant whose final member appears twice."""
    return replace(group, members=group.members + (group.members[-1],))


def dropped_route(routes: Sequence[Route]) -> tuple[Route, ...]:
    """Return a routing mutant with the final semantic route removed."""
    return tuple(routes[:-1])


def reversed_collectives(trace: Sequence[CollectiveKey]) -> tuple[CollectiveKey, ...]:
    """Return a collective-trace mutant with reversed launch order."""
    return tuple(reversed(trace))


def stale_epoch(trace: Sequence[CollectiveKey]) -> tuple[CollectiveKey, ...]:
    """Return a collective-trace mutant whose first epoch is stale."""
    if not trace:
        raise ValueError("trace must be nonempty")
    return (replace(trace[0], epoch=trace[0].epoch - 1), *trace[1:])


def wrong_shape(trace: Sequence[CollectiveKey]) -> tuple[CollectiveKey, ...]:
    """Return a collective-trace mutant whose first payload shape is wrong."""
    if not trace:
        raise ValueError("trace must be nonempty")
    return (replace(trace[0], shape=(1, 8, 8)), *trace[1:])
