"""Declared-resource conflict oracle for Chapter 28."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Access:
    """Declare one call's resource access sets and provider order."""

    call_id: str
    reads: frozenset[str]
    writes: frozenset[str]
    provider_index: int


def conflicts(left: Access, right: Access) -> bool:
    """Return whether two accesses have a read/write or write/write hazard."""
    return bool(
        left.writes & (right.reads | right.writes)
        or right.writes & (left.reads | left.writes)
    )


def conflict_edges(accesses: Iterable[Access]) -> tuple[tuple[str, str], ...]:
    """Return sorted call pairs whose declared resource accesses conflict."""
    values = tuple(accesses)
    edges: list[tuple[str, str]] = []
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if conflicts(left, right):
                first, second = sorted((left.call_id, right.call_id))
                edges.append((first, second))
    return tuple(sorted(edges))


def schedule_batches(accesses: Iterable[Access]) -> tuple[tuple[str, ...], ...]:
    """Return a minimum-batch schedule for the chapter's small finite graphs."""

    values = tuple(accesses)
    by_id = {access.call_id: access for access in values}
    if len(by_id) != len(values):
        raise ValueError("duplicate call ID")
    ordered = sorted(values, key=lambda item: item.provider_index)
    best: list[list[str]] | None = None

    def assign(position: int, batches: list[list[str]]) -> None:
        nonlocal best
        if best is not None and len(batches) >= len(best):
            return
        if position == len(ordered):
            best = [list(batch) for batch in batches]
            return
        access = ordered[position]
        for batch in batches:
            if all(not conflicts(access, by_id[other]) for other in batch):
                batch.append(access.call_id)
                assign(position + 1, batches)
                batch.pop()
        batches.append([access.call_id])
        assign(position + 1, batches)
        batches.pop()

    assign(0, [])
    assert best is not None
    return tuple(tuple(batch) for batch in best)


def provider_order(
    completed_call_ids: Iterable[str], accesses: Iterable[Access]
) -> tuple[str, ...]:
    """Restore provider submission order for a completed call set."""
    index = {access.call_id: access.provider_index for access in accesses}
    completed = tuple(completed_call_ids)
    if set(completed) != set(index):
        raise ValueError("completion set does not match submitted calls")
    return tuple(sorted(completed, key=index.__getitem__))
