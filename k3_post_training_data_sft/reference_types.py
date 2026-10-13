"""Typed records and frozen Chapter 17 studio interactions."""

from __future__ import annotations

from dataclasses import dataclass


Box = tuple[int, int, int, int]
RED_BOX: Box = (20, 20, 40, 40)
BLUE_BOX: Box = (20, 80, 40, 40)
CROPPED_RED_BOX: Box = (10, 10, 40, 40)
CROPPED_BLUE_BOX: Box = (10, 70, 40, 40)
VALIDATOR_VERSION = "chapter16-validator-v2"
SERIALIZER_VERSION = "chapter16-symbolic-v1"


@dataclass(frozen=True)
class Event:
    """One typed event in the Chapter 17 record."""

    ordinal: int
    kind: str
    actor: str
    payload: object
    correlation: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """Named validator result rather than an uninformative Boolean."""

    accepted: bool
    code: str
    event_ordinal: int | None


def task_contract() -> dict[str, object]:
    """Return the frozen studio task contract."""

    return {
        "instruction": (
            "Use the available tools to determine whether the blue square is "
            "below the red circle, and report the evidence."
        ),
        "actions": {
            "locate_shapes": {
                "required_arguments": ["image_id"],
                "optional_arguments": [],
            },
            "compare_vertical": {
                "required_arguments": ["red_box", "blue_box"],
                "optional_arguments": [],
            },
        },
        "observations": ["shape_boxes", "vertical_comparison", "tool_error"],
        "terminal_predicate": "below with center delta 60 and cited evidence",
        "maximum_calls": 3,
    }


def environment_contract() -> dict[str, object]:
    """Return the deterministic studio environment fixture."""

    return {
        "state_id": "studio-canvas-v1",
        "red_box": list(RED_BOX),
        "blue_box": list(BLUE_BOX),
        "red_center_y": center_y(RED_BOX),
        "blue_center_y": center_y(BLUE_BOX),
    }


def center_y(box: Box) -> int:
    """Return the integral vertical center of a studio box."""

    _, y_min, _, height = box
    return y_min + height // 2


def valid_events() -> tuple[Event, ...]:
    """Return the six-event valid studio interaction."""

    return (
        Event(0, "instruction", "user", task_contract()["instruction"]),
        Event(
            1,
            "tool_call",
            "assistant",
            {
                "name": "locate_shapes",
                "arguments": {"image_id": "studio-layout-001"},
            },
            "c1",
        ),
        Event(
            2,
            "tool_result",
            "environment",
            {"red_box": list(RED_BOX), "blue_box": list(BLUE_BOX)},
            "c1",
        ),
        Event(
            3,
            "tool_call",
            "assistant",
            {
                "name": "compare_vertical",
                "arguments": {"red_box": list(RED_BOX), "blue_box": list(BLUE_BOX)},
            },
            "c2",
        ),
        Event(
            4,
            "tool_result",
            "environment",
            {"below": True, "center_delta": 60},
            "c2",
        ),
        Event(
            5,
            "terminal_answer",
            "assistant",
            {
                "relation": "below",
                "center_delta": 60,
                "evidence_calls": ["c1", "c2"],
            },
        ),
    )


def _locate_payload() -> dict[str, object]:
    return {
        "name": "locate_shapes",
        "arguments": {"image_id": "studio-layout-001"},
    }


def _executor_error_events() -> tuple[Event, ...]:
    instruction = Event(0, "instruction", "user", task_contract()["instruction"])
    locate_payload = _locate_payload()
    return (
        instruction,
        Event(1, "tool_call", "assistant", locate_payload, "c1"),
        Event(
            2,
            "tool_error",
            "environment",
            {"code": "TRANSIENT_EXECUTOR_FAILURE"},
            "c1",
        ),
        Event(3, "tool_call", "assistant", locate_payload, "c2"),
        Event(
            4,
            "tool_result",
            "environment",
            {"red_box": list(RED_BOX), "blue_box": list(BLUE_BOX)},
            "c2",
        ),
        Event(
            5,
            "tool_call",
            "assistant",
            {
                "name": "compare_vertical",
                "arguments": {
                    "red_box": list(RED_BOX),
                    "blue_box": list(BLUE_BOX),
                },
            },
            "c3",
        ),
        Event(
            6,
            "tool_result",
            "environment",
            {"below": True, "center_delta": 60},
            "c3",
        ),
        Event(
            7,
            "terminal_answer",
            "assistant",
            {
                "relation": "below",
                "center_delta": 60,
                "evidence_calls": ["c2", "c3"],
            },
        ),
    )


def _state_refresh_events() -> tuple[Event, ...]:
    instruction = Event(0, "instruction", "user", task_contract()["instruction"])
    locate_payload = _locate_payload()
    return (
        instruction,
        Event(1, "tool_call", "assistant", locate_payload, "c1"),
        Event(
            2,
            "tool_result",
            "environment",
            {"red_box": list(RED_BOX), "blue_box": list(BLUE_BOX)},
            "c1",
        ),
        Event(
            3,
            "state_transform",
            "environment",
            {"name": "crop", "offset_x": 10, "offset_y": 10},
        ),
        Event(4, "tool_call", "assistant", locate_payload, "c2"),
        Event(
            5,
            "tool_result",
            "environment",
            {
                "red_box": list(CROPPED_RED_BOX),
                "blue_box": list(CROPPED_BLUE_BOX),
            },
            "c2",
        ),
        Event(
            6,
            "tool_call",
            "assistant",
            {
                "name": "compare_vertical",
                "arguments": {
                    "red_box": list(CROPPED_RED_BOX),
                    "blue_box": list(CROPPED_BLUE_BOX),
                },
            },
            "c3",
        ),
        Event(
            7,
            "tool_result",
            "environment",
            {"below": True, "center_delta": 60},
            "c3",
        ),
        Event(
            8,
            "terminal_answer",
            "assistant",
            {
                "relation": "below",
                "center_delta": 60,
                "evidence_calls": ["c2", "c3"],
            },
        ),
    )


def recovery_events(kind: str) -> tuple[Event, ...]:
    """Return one of two frozen, dependency-closed recovery trajectories."""

    if kind == "executor_error":
        return _executor_error_events()
    if kind == "state_refresh":
        return _state_refresh_events()
    raise ValueError(f"unknown recovery kind {kind}")


__all__ = [
    "BLUE_BOX",
    "Box",
    "CROPPED_BLUE_BOX",
    "CROPPED_RED_BOX",
    "Event",
    "RED_BOX",
    "SERIALIZER_VERSION",
    "VALIDATOR_VERSION",
    "ValidationResult",
    "center_y",
    "environment_contract",
    "recovery_events",
    "task_contract",
    "valid_events",
]
