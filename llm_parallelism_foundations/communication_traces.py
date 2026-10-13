"""Finite semantic trace checks; no GPU visibility or runtime progress proof."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from .collectives import group_members


@dataclass(frozen=True)
class CollectiveCall:
    """An application event label plus the declared collective signature."""

    event: str
    group: tuple[int, ...]
    operation: str
    input_count: int
    dtype: str = "rational"


def validate_collective_order(
    traces: Mapping[int, Sequence[CollectiveCall]], group: Sequence[int]
) -> None:
    """Every participant must present the same ordered finite event sequence."""
    group_members(group, traces)
    expected = tuple(traces[group[0]])
    for rank in group:
        for call in traces[rank]:
            if call.group != tuple(group) or call.input_count < 0:
                raise ValueError("incompatible participant group or count")
        if tuple(traces[rank]) != expected:
            raise ValueError("collective event order or signature mismatch")


def buffer_trace() -> dict[str, int]:
    """Ordered event slots, with no duration or performance interpretation."""
    return {
        "produced": 0,
        "submitted": 1,
        "ready": 3,
        "consumer_start": 4,
        "consumer_end": 6,
        "reuse": 7,
    }


def validate_buffer_trace(events: Mapping[str, int]) -> None:
    """This fixture communicates in place, then protects one consumer's read."""
    names = tuple(buffer_trace())
    if set(events) != set(names) or any(events[n] < 0 for n in names):
        raise ValueError("missing or invalid lifetime event")
    for producer, consumer in pairwise(names):
        if events[producer] >= events[consumer]:
            raise ValueError(f"{consumer} precedes required completion of {producer}")
