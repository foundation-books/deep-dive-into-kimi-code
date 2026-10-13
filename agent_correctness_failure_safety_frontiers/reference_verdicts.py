"""World and execution-verdict primitives for Chapter 32."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .reference_types import Event


def verdict(worlds: Sequence[int], predicate: Callable[[int], bool]) -> str:
    """Classify a predicate without assuming it when constructing the worlds."""

    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in worlds):
        raise ValueError("allocation worlds must be nonnegative integers")
    if not worlds:
        return "inconsistent"
    values = {predicate(n) for n in worlds}
    if values == {True}:
        return "supported"
    if values == {False}:
        return "refuted"
    return "unknown"


def fold(events: Sequence[Event]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Apply the strict exchange fold; a timeout retains an open exchange."""

    calls: dict[str, Event] = {}
    closed: set[str] = set()
    for event in events:
        if event.kind not in {"call", "result", "error", "timeout"}:
            raise ValueError("unknown event grammar")
        if not all((event.call, event.tool, event.payload, event.request)):
            raise ValueError("empty binding")
        if event.kind == "call":
            if event.call in calls:
                raise ValueError("duplicate call")
            calls[event.call] = event
            continue
        if event.call not in calls or event.call in closed:
            raise ValueError("result/observation has no open call")
        prior = calls[event.call]
        if (event.tool, event.payload, event.request) != (
            prior.tool,
            prior.payload,
            prior.request,
        ):
            raise ValueError("mismatched binding")
        if event.kind in {"result", "error"}:
            closed.add(event.call)
    return tuple(sorted(calls.keys() - closed)), tuple(sorted(closed))


def classify_run(
    counts: Sequence[int],
    authorities: Sequence[bool],
    goal: bool,
    events: Sequence[Event],
    disposed: bool,
) -> dict[str, bool]:
    """Build the actual-state fixture vector used by evidence verdicts."""

    if len(counts) != len(authorities) + 1 or not counts:
        raise ValueError("one initial state and one successor per covered effect")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
        raise ValueError("invalid allocation count")
    syntax = all(
        event.kind in {"call", "result", "error", "timeout"}
        and all((event.call, event.tool, event.payload, event.request))
        for event in events
    )
    try:
        fold(events)
        protocol = True
    except ValueError:
        protocol = False
    return {
        "syntax": syntax,
        "protocol": protocol,
        "authority": all(authorities),
        "goal": goal,
        "invariant": all(n <= 1 for n in counts),
        "progress": disposed,
    }


__all__ = ["classify_run", "fold", "verdict"]
