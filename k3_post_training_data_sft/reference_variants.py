"""Frozen corruptions, recovery records, and Chapter 17 record fixture."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Sequence

from .reference_types import (
    BLUE_BOX,
    RED_BOX,
    VALIDATOR_VERSION,
    Event,
    center_y,
    recovery_events,
    valid_events,
)
from .reference_validation import certificate_for, content_hash, validate_record


def _renumber(events: Sequence[Event]) -> list[Event]:
    """Return events with contiguous ordinals after an insertion."""

    return [
        Event(index, event.kind, event.actor, event.payload, event.correlation)
        for index, event in enumerate(events)
    ]


def _terminal_success(events: Sequence[Event]) -> bool:
    """Evaluate the frozen studio terminal predicate independently of validity."""

    terminals = [event for event in events if event.kind == "terminal_answer"]
    if len(terminals) != 1 or not isinstance(terminals[0].payload, dict):
        return False
    return terminals[0].payload == {
        "relation": "below",
        "center_delta": 60,
        "evidence_calls": ["c1", "c2"],
    }


def _permissive_simulator_accepts(events: Sequence[Event]) -> bool:
    """Model the declared simulator-only C7 schema shortcut."""

    compare_calls = [
        event
        for event in events
        if event.kind == "tool_call"
        and isinstance(event.payload, dict)
        and event.payload.get("name") == "compare_vertical"
    ]
    if len(compare_calls) != 1 or not isinstance(compare_calls[0].payload, dict):
        return False
    arguments = compare_calls[0].payload.get("arguments")
    return (
        isinstance(arguments, dict)
        and set(arguments) == {"red_box", "blue_box", "bbox_center"}
        and _terminal_success(events)
    )


def _variant_c1(base: list[Event]) -> dict[str, object] | None:
    base[2] = Event(
        2,
        "tool_result",
        "environment",
        {"red_box": list(RED_BOX), "blue_box": [20, 70, 40, 40]},
        "c1",
    )
    return None


def _variant_c2(base: list[Event]) -> dict[str, object] | None:
    base[5] = Event(
        5,
        "terminal_answer",
        "assistant",
        {
            "relation": "above",
            "center_delta": -60,
            "evidence_calls": ["c1", "c2"],
        },
    )
    return None


def _variant_c3(base: list[Event]) -> dict[str, object] | None:
    event = base[4]
    base[4] = Event(event.ordinal, event.kind, event.actor, event.payload, "c1")
    return None


def _variant_c4(base: list[Event]) -> dict[str, object] | None:
    base.insert(
        3,
        Event(
            3,
            "state_transform",
            "environment",
            {"name": "crop", "offset_x": 10, "offset_y": 10},
        ),
    )
    base[:] = _renumber(base)
    return None


def _variant_c5(base: list[Event]) -> dict[str, object] | None:
    event = base[1]
    assert isinstance(event.payload, dict)
    payload = dict(event.payload)
    payload["depends_on"] = ["c1"]
    base[1] = Event(event.ordinal, event.kind, event.actor, payload, event.correlation)
    return None


def _variant_c6(base: list[Event]) -> dict[str, object] | None:
    parent = list(valid_events())
    parent[5] = Event(
        5,
        "terminal_answer",
        "assistant",
        {
            "relation": "above",
            "center_delta": -60,
            "evidence_calls": ["c1", "c2"],
        },
    )
    return certificate_for(parent)


def _variant_c7(base: list[Event]) -> dict[str, object] | None:
    event = base[3]
    assert isinstance(event.payload, dict)
    arguments = dict(event.payload["arguments"])
    arguments["bbox_center"] = [20, 100]
    base[3] = Event(
        event.ordinal,
        event.kind,
        event.actor,
        {"name": "compare_vertical", "arguments": arguments},
        event.correlation,
    )
    return None


def _variant_c8(base: list[Event]) -> dict[str, object] | None:
    base.insert(
        5,
        Event(
            5,
            "tool_call",
            "assistant",
            {
                "name": "locate_shapes",
                "arguments": {"image_id": "studio-layout-001"},
            },
            "c3",
        ),
    )
    base.insert(
        6,
        Event(
            6,
            "tool_result",
            "environment",
            {"red_box": list(RED_BOX), "blue_box": list(BLUE_BOX)},
            "c3",
        ),
    )
    base[:] = _renumber(base)
    return None


VariantBuilder = Callable[[list[Event]], dict[str, object] | None]
VARIANT_BUILDERS: dict[str, VariantBuilder] = {
    "C1": _variant_c1,
    "C2": _variant_c2,
    "C3": _variant_c3,
    "C4": _variant_c4,
    "C5": _variant_c5,
    "C6": _variant_c6,
    "C7": _variant_c7,
    "C8": _variant_c8,
}


def corrupted_variant(identifier: str) -> tuple[tuple[Event, ...], dict[str, object]]:
    """Return one of the eight frozen studio variants and its certificate."""

    try:
        builder = VARIANT_BUILDERS[identifier]
    except KeyError as error:
        raise ValueError(f"unknown corruption {identifier}") from error
    base = list(valid_events())
    candidate_certificate = builder(base)
    candidate = tuple(base)
    if candidate_certificate is None:
        candidate_certificate = certificate_for(candidate)
    return candidate, candidate_certificate


EXPECTED_VARIANTS: dict[str, dict[str, object]] = {
    "C1": {
        "name": "right answer, forged result",
        "accepted": False,
        "code": "TRANSITION_FIDELITY",
        "terminal_success": True,
        "trajectory_validity": False,
    },
    "C2": {
        "name": "valid calls, wrong answer",
        "accepted": False,
        "code": "TERMINAL_CONSISTENCY",
        "terminal_success": False,
        "trajectory_validity": False,
    },
    "C3": {
        "name": "call--result mismatch",
        "accepted": False,
        "code": "ORPHAN_RESULT",
        "terminal_success": True,
        "trajectory_validity": False,
    },
    "C4": {
        "name": "stale-state trace",
        "accepted": False,
        "code": "STALE_STATE_ARGUMENTS",
        "terminal_success": True,
        "trajectory_validity": False,
    },
    "C5": {
        "name": "causality violation",
        "accepted": False,
        "code": "CAUSAL_PREFIX",
        "terminal_success": True,
        "trajectory_validity": False,
    },
    "C6": {
        "name": "unverified repair",
        "accepted": False,
        "code": "STALE_CERTIFICATE",
        "terminal_success": True,
        "trajectory_validity": True,
    },
    "C7": {
        "name": "simulator-only success",
        "accepted": False,
        "code": "CALL_SCHEMA",
        "terminal_success": True,
        "trajectory_validity": False,
    },
    "C8": {
        "name": "valid but inefficient trace",
        "accepted": True,
        "code": "ACCEPT",
        "terminal_success": True,
        "trajectory_validity": True,
    },
}


EXPECTED_REJECTIONS = {
    identifier: str(expected["code"])
    for identifier, expected in EXPECTED_VARIANTS.items()
    if expected["accepted"] is False
}


REPAIR_DEPENDENCY_GRAPH = {
    "task_or_environment": [
        "actions",
        "observations",
        "terminal",
        "serialization",
        "mask",
        "certificate",
    ],
    "action_or_observation": [
        "downstream_actions",
        "downstream_observations",
        "terminal",
        "serialization",
        "mask",
        "certificate",
    ],
    "terminal": ["serialization", "mask", "certificate"],
}


def _variant_fixture(identifier: str, expected: dict[str, object]) -> dict[str, object]:
    candidate, candidate_certificate = corrupted_variant(identifier)
    result = validate_record(candidate, candidate_certificate)
    content_result = validate_record(candidate)
    call_count = sum(event.kind == "tool_call" for event in candidate)
    return {
        "name": expected["name"],
        "accepted": result.accepted,
        "code": result.code,
        "expected_accepted": expected["accepted"],
        "expected_code": expected["code"],
        "event_ordinal": result.event_ordinal,
        "terminal_success": _terminal_success(candidate),
        "expected_terminal_success": expected["terminal_success"],
        "trajectory_validity": content_result.accepted,
        "simulator_accepted": (
            _permissive_simulator_accepts(candidate) if identifier == "C7" else None
        ),
        "call_count": call_count,
        "quality_flag": "LOW_EFFICIENCY"
        if result.accepted and call_count > 2
        else None,
    }


def _recovery_fixture(kind: str) -> dict[str, object]:
    recovery = recovery_events(kind)
    result = validate_record(recovery, certificate_for(recovery))
    return {
        "accepted": result.accepted,
        "code": result.code,
        "event_count": len(recovery),
        "call_count": sum(event.kind == "tool_call" for event in recovery),
        "content_hash": content_hash(recovery),
    }


def record_fixture() -> dict[str, object]:
    """Return the valid record and all eight frozen variant outcomes."""

    events = valid_events()
    certificate = certificate_for(events)
    valid = validate_record(events, certificate)
    variants = {
        identifier: _variant_fixture(identifier, expected)
        for identifier, expected in EXPECTED_VARIANTS.items()
    }
    recoveries = {
        kind: _recovery_fixture(kind) for kind in ("executor_error", "state_refresh")
    }
    return {
        "validator_version": VALIDATOR_VERSION,
        "valid_record": asdict(valid),
        "content_hash": certificate["content_hash"],
        "red_center_y": center_y(RED_BOX),
        "blue_center_y": center_y(BLUE_BOX),
        "center_delta": center_y(BLUE_BOX) - center_y(RED_BOX),
        "call_count": sum(event.kind == "tool_call" for event in events),
        "repair_dependency_graph": REPAIR_DEPENDENCY_GRAPH,
        "recovery_records": recoveries,
        "variants": variants,
    }


__all__ = [
    "EXPECTED_REJECTIONS",
    "EXPECTED_VARIANTS",
    "REPAIR_DEPENDENCY_GRAPH",
    "corrupted_variant",
    "record_fixture",
]
