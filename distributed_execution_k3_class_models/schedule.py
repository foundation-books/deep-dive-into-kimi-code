"""Versioned schedules, collective traces, and resource bounds for Chapter 37."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
from typing import Mapping, Sequence

from .semantics import f
from .reader_interfaces import placement_witness


@dataclass(frozen=True, order=True)
class EventId:
    """Stable identity of one direction, logical chunk, and microbatch event."""

    direction: str
    logical_chunk: int
    microbatch: int


@dataclass(frozen=True)
class Event:
    """One versioned operator event in the pipeline DAG."""

    event_id: EventId
    physical_stage: int
    virtual_chunk: int
    dependencies: tuple[EventId, ...]


def pipeline_events(microbatches: int = 4) -> dict[EventId, Event]:
    """Build the P=2, V=2 forward/backward semantic DAG."""
    if microbatches <= 0:
        raise ValueError("at least one microbatch is required")
    events: dict[EventId, Event] = {}
    for microbatch in range(microbatches):
        for chunk in range(4):
            event_id = EventId("F", chunk, microbatch)
            dependencies: tuple[EventId, ...] = (
                () if chunk == 0 else (EventId("F", chunk - 1, microbatch),)
            )
            events[event_id] = Event(event_id, chunk % 2, chunk // 2, dependencies)
        for chunk in reversed(range(4)):
            event_id = EventId("B", chunk, microbatch)
            dependencies = (EventId("F", chunk, microbatch),)
            if chunk < 3:
                dependencies += (EventId("B", chunk + 1, microbatch),)
            events[event_id] = Event(event_id, chunk % 2, chunk // 2, dependencies)
    return events


def topological_order(events: Mapping[EventId, Event]) -> tuple[EventId, ...]:
    """Return a deterministic topological order or reject a cycle."""
    remaining = set(events)
    completed: set[EventId] = set()
    ordered: list[EventId] = []
    while remaining:
        ready = sorted(
            event_id
            for event_id in remaining
            if set(events[event_id].dependencies) <= completed
        )
        if not ready:
            raise ValueError(
                "event dependencies contain a cycle or missing predecessor"
            )
        chosen = ready[0]
        remaining.remove(chosen)
        completed.add(chosen)
        ordered.append(chosen)
    return tuple(ordered)


def validate_pipeline(events: Mapping[EventId, Event], microbatches: int = 4) -> None:
    """Check complete versions and forward-before-backward order."""
    expected = {
        EventId(direction, chunk, microbatch)
        for direction in ("F", "B")
        for chunk in range(4)
        for microbatch in range(microbatches)
    }
    if set(events) != expected:
        raise ValueError("pipeline event set is incomplete or duplicated")
    order = topological_order(events)
    positions = {event_id: index for index, event_id in enumerate(order)}
    for event in events.values():
        if any(
            positions[dependency] >= positions[event.event_id]
            for dependency in event.dependencies
        ):
            raise ValueError("dependency is not ordered before its consumer")


def pipeline_resource_schedule() -> tuple[tuple[EventId | None, ...], ...]:
    """Return the manuscript's complete equal-slot schedule for ``P=2, V=2, M=4``."""

    def forward(chunk: int, microbatch: int) -> EventId:
        return EventId("F", chunk, microbatch)

    def backward(chunk: int, microbatch: int) -> EventId:
        return EventId("B", chunk, microbatch)

    return (
        (
            forward(0, 0),
            forward(0, 1),
            forward(2, 0),
            forward(2, 1),
            forward(0, 2),
            backward(2, 0),
            forward(0, 3),
            backward(2, 1),
            backward(0, 0),
            backward(0, 1),
            forward(2, 2),
            forward(2, 3),
            None,
            backward(2, 2),
            None,
            backward(2, 3),
            backward(0, 2),
            backward(0, 3),
        ),
        (
            None,
            forward(1, 0),
            forward(1, 1),
            forward(3, 0),
            backward(3, 0),
            forward(3, 1),
            backward(3, 1),
            backward(1, 0),
            backward(1, 1),
            forward(1, 2),
            forward(1, 3),
            forward(3, 2),
            backward(3, 2),
            forward(3, 3),
            backward(3, 3),
            backward(1, 2),
            backward(1, 3),
            None,
        ),
    )


def validate_resource_schedule(
    schedule: Sequence[Sequence[EventId | None]],
    events: Mapping[EventId, Event],
) -> None:
    """Validate stage placement, uniqueness, completeness, and dependency order."""
    if len(schedule) != 2 or not schedule or len({len(row) for row in schedule}) != 1:
        raise ValueError("the teaching schedule needs two equal-length stage rows")
    positions: dict[EventId, int] = {}
    for physical_stage, row in enumerate(schedule):
        for slot, event_id in enumerate(row):
            if event_id is None:
                continue
            if event_id in positions:
                raise ValueError("a pipeline event appears more than once")
            if event_id not in events:
                raise ValueError("the schedule contains an unknown pipeline event")
            if events[event_id].physical_stage != physical_stage:
                raise ValueError(
                    "a pipeline event is assigned to the wrong physical stage"
                )
            positions[event_id] = slot
    if set(positions) != set(events):
        raise ValueError("the resource schedule is incomplete")
    for event_id, event in events.items():
        if any(
            positions[dependency] >= positions[event_id]
            for dependency in event.dependencies
        ):
            raise ValueError("a scheduled dependency does not precede its consumer")


def ideal_interleaved_bubble_fraction(microbatches: int = 4) -> Fraction:
    """Return ``(P-1)/(M*V+P-1)`` for equal chunk times."""
    if microbatches <= 0:
        raise ValueError("microbatch count must be positive")
    return Fraction(1, microbatches * 2 + 1)


def fixed_stage_order_events() -> dict[EventId, Event]:
    """Add only the displayed stage order to the complete semantic DAG."""
    events = pipeline_events()
    for row in pipeline_resource_schedule():
        previous = None
        for event_id in row:
            if event_id is None:
                continue
            event = events[event_id]
            if previous is not None and previous not in event.dependencies:
                events[event_id] = replace(
                    event, dependencies=event.dependencies + (previous,)
                )
            previous = event_id
    return events


def earliest_finishes(
    events: Mapping[EventId, Event], durations: Mapping[EventId, Fraction]
) -> dict[EventId, Fraction]:
    """Return the full weighted-DAG finish ledger, not a timing observation."""
    finish: dict[EventId, Fraction] = {}
    for event_id in topological_order(events):
        if event_id not in durations or durations[event_id] <= 0:
            raise ValueError("every event needs a positive duration")
        finish[event_id] = durations[event_id] + max(
            (finish[predecessor] for predecessor in events[event_id].dependencies),
            default=f(0),
        )
    return finish


def longest_dependency_path(
    events: Mapping[EventId, Event], durations: Mapping[EventId, Fraction]
) -> Fraction:
    """Compute the exact weighted longest path through the semantic DAG."""
    order = topological_order(events)
    finish: dict[EventId, Fraction] = {}
    for event_id in order:
        if event_id not in durations or durations[event_id] <= 0:
            raise ValueError("every event needs a positive duration")
        start = max(
            (finish[dependency] for dependency in events[event_id].dependencies),
            default=f(0),
        )
        finish[event_id] = start + durations[event_id]
    return max(finish.values(), default=f(0))


@dataclass(frozen=True)
class CollectiveKey:
    """Collective key plus separately validated group/revision metadata.

    The historical ``layer`` field carries logical chunk k; decoder_layer is
    the distinct one-based layer. ``epoch`` is the execution epoch only.
    """

    group_id: str
    participants: tuple[int, ...]
    operation: str
    object_id: str
    layer: int
    microbatch: int
    direction: str
    sequence: int
    shape: tuple[int, ...]
    epoch: int
    dtype: str = "rat"
    membership_epoch: int = 0
    parameter_revision: int = 7
    decoder_layer: int = 5


def canonical_collective_trace(epoch: int = 7) -> tuple[CollectiveKey, ...]:
    """Return the two-event context-parallel trace used by mismatch checks."""
    participants = (19, 23)
    return (
        CollectiveKey(
            "CP[d1,p0,e1,t1]",
            participants,
            "all_gather",
            "kda_fragment[ell5,u1,rev7]",
            2,
            2,
            "F",
            0,
            (2, 8, 8),
            epoch,
        ),
        CollectiveKey(
            "CP[d1,p0,e1,t1]",
            participants,
            "reduce_scatter",
            "kda_fragment_grad[ell5,u1,rev7]",
            2,
            1,
            "B",
            1,
            (2, 8, 8),
            epoch,
        ),
    )


def _key_mismatch_kind(expected: CollectiveKey, actual: CollectiveKey) -> str:
    """Classify unequal keys in the manuscript's diagnostic priority order."""
    fields = (
        (
            "group",
            (expected.group_id, expected.participants),
            (actual.group_id, actual.participants),
        ),
        ("epoch", expected.epoch, actual.epoch),
        ("membership_epoch", expected.membership_epoch, actual.membership_epoch),
        ("revision", expected.parameter_revision, actual.parameter_revision),
        ("shape", expected.shape, actual.shape),
        ("type", expected.dtype, actual.dtype),
        (
            "order",
            (expected.operation, expected.sequence),
            (actual.operation, actual.sequence),
        ),
        ("object", expected.object_id, actual.object_id),
    )
    return next((kind for kind, left, right in fields if left != right), "key")


def first_collective_mismatch(
    traces: Mapping[int, Sequence[CollectiveKey]],
) -> tuple[int, int, str] | None:
    """Return rank, sequence position, and mismatch class for the first divergence."""
    if not traces:
        raise ValueError("at least one participant trace is required")
    reference_rank = min(traces)
    reference = tuple(traces[reference_rank])
    for rank in sorted(traces):
        candidate = tuple(traces[rank])
        if len(candidate) != len(reference):
            return rank, min(len(candidate), len(reference)), "length"
        for index, (expected, actual) in enumerate(
            zip(reference, candidate, strict=True)
        ):
            if expected == actual:
                continue
            return rank, index, _key_mismatch_kind(expected, actual)
    return None


@dataclass(frozen=True)
class ResourceVector:
    """Typed analytical coordinates, with measured time deliberately absent."""

    useful_work: Fraction
    collective_payload: tuple[tuple[str, Fraction], ...]
    physical_link_load: tuple[tuple[str, Fraction], ...]
    resident_storage: Fraction
    launches: int
    synchronizations: int
    duration_input: Fraction
    link_load_bound: Fraction
    observed_time: None = None
    critical_path_bound: None = None
    provenance: tuple[str, ...] = (
        "work, multi-axis payload, storage, backend counts, and 17 ticks: assigned cost card",
        "physical inter-node load: derived isolated M1/M2 placement witness",
        "different scopes; no common step-time bound is asserted",
    )


def resource_fixture(alternative: bool = False) -> ResourceVector:
    """Retain assigned cost inputs alongside a separately scoped placement witness.

    The duration input 17 is supplied; the cost card has no critical-path DAG.
    Use earliest_finishes for the printed 18/26 resource-order DAG.
    """
    payload = (("TP", f(8)), ("CP", f(6)), ("EP", f(10)), ("PP", f(4)), ("DP", f(3)))
    _, directed_loads = placement_witness(alternative)
    links = tuple(
        (f"isolated:N{source}->N{target}", f(load))
        for (source, target), load in directed_loads.items()
    )
    maximum = max(value for _, value in links)
    return ResourceVector(f(128), payload, links, f(96), 14, 9, f(17), maximum)


def legal_overlap(shared_link_budget: Fraction, demands: Sequence[Fraction]) -> bool:
    """Require positive demands; return whether simultaneous demand fits the budget."""
    if shared_link_budget <= 0 or any(demand < 0 for demand in demands):
        raise ValueError("budget must be positive and demands nonnegative")
    return sum(demands, f(0)) <= shared_link_budget


def trace_digest(events: Mapping[EventId, Event]) -> str:
    """Hash the complete deterministic pipeline event set and dependencies."""
    payload = "|".join(
        f"{event_id.direction}{event_id.logical_chunk}m{event_id.microbatch}:"
        + ",".join(
            f"{dependency.direction}{dependency.logical_chunk}m{dependency.microbatch}"
            for dependency in events[event_id].dependencies
        )
        for event_id in sorted(events)
    )
    return sha256(payload.encode("utf-8")).hexdigest()
