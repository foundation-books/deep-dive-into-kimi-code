"""Typed causal validation for the Chapter 17 studio record."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Mapping, Sequence

from .reference_types import (
    BLUE_BOX,
    RED_BOX,
    VALIDATOR_VERSION,
    Box,
    Event,
    ValidationResult,
    center_y,
    task_contract,
)


@dataclass
class _ValidationState:
    """Mutable bookkeeping for one record-validation pass."""

    unmatched: dict[str, tuple[Event, dict[str, object]]] = field(default_factory=dict)
    seen_call_keys: set[str] = field(default_factory=set)
    completed_results: set[str] = field(default_factory=set)
    current_red_box: Box = RED_BOX
    current_blue_box: Box = BLUE_BOX
    calls: int = 0
    unresolved_failures: set[str] = field(default_factory=set)
    successful_call_types: dict[str, str] = field(default_factory=dict)
    terminal: Event | None = None


def canonical_event_bytes(events: Sequence[Event]) -> bytes:
    """Return stable bytes used by the chapter certificate fixture."""

    payload = [asdict(event) for event in events]
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def content_hash(events: Sequence[Event]) -> str:
    """Hash the accepted typed content."""

    return hashlib.sha256(canonical_event_bytes(events)).hexdigest()


def certificate_for(events: Sequence[Event]) -> dict[str, object]:
    """Issue the deterministic certificate used by the fixture."""

    return {
        "validator_version": VALIDATOR_VERSION,
        "content_hash": content_hash(events),
        "required_predicates": [
            "schema",
            "causal_prefix",
            "call_result",
            "transition_fidelity",
            "terminal_consistency",
        ],
    }


def _reject(code: str, event: Event | None = None) -> ValidationResult:
    return ValidationResult(False, code, None if event is None else event.ordinal)


def _is_box(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) for item in value)
    )


def _validate_call_schema(event: Event) -> str | None:
    if event.actor != "assistant" or not isinstance(event.payload, dict):
        return "TYPE_COMPLETENESS"
    name = event.payload.get("name")
    arguments = event.payload.get("arguments")
    if not isinstance(arguments, dict):
        return "TYPE_COMPLETENESS"
    if name == "locate_shapes":
        return None if arguments == {"image_id": "studio-layout-001"} else "CALL_SCHEMA"
    if name == "compare_vertical":
        if set(arguments) != {"red_box", "blue_box"}:
            return "CALL_SCHEMA"
        if not _is_box(arguments["red_box"]) or not _is_box(arguments["blue_box"]):
            return "CALL_SCHEMA"
        return None
    return "CALL_SCHEMA"


def _execute_call(
    event: Event, red_box: Box = RED_BOX, blue_box: Box = BLUE_BOX
) -> dict[str, object]:
    """Execute the deterministic chapter subset after schema validation."""

    assert isinstance(event.payload, dict)
    name = event.payload["name"]
    arguments = event.payload["arguments"]
    assert isinstance(arguments, dict)
    if name == "locate_shapes":
        return {"red_box": list(red_box), "blue_box": list(blue_box)}
    red = tuple(arguments["red_box"])
    blue = tuple(arguments["blue_box"])
    assert len(red) == 4 and len(blue) == 4
    delta = center_y(blue) - center_y(red)  # type: ignore[arg-type]
    return {"below": delta > 0, "center_delta": delta}


def _dependency_keys(event: Event) -> tuple[str, ...] | None:
    """Return declared observation dependencies, or ``None`` if malformed."""

    if not isinstance(event.payload, dict):
        return None
    raw = event.payload.get("depends_on", [])
    if not isinstance(raw, list) or any(not isinstance(key, str) for key in raw):
        return None
    return tuple(raw)


def _apply_state_transform(
    event: Event, red_box: Box, blue_box: Box
) -> tuple[Box, Box] | None:
    """Apply the one declared crop transform used by the stale-state fixture."""

    if event.actor != "environment" or event.payload != {
        "name": "crop",
        "offset_x": 10,
        "offset_y": 10,
    }:
        return None

    def translate(box: Box) -> Box:
        x_min, y_min, width, height = box
        return (x_min - 10, y_min - 10, width, height)

    return translate(red_box), translate(blue_box)


def _validate_instruction(event: Event) -> ValidationResult | None:
    if event.actor != "user" or not isinstance(event.payload, str):
        return _reject("TYPE_COMPLETENESS", event)
    return None


def _validate_tool_call(
    event: Event, state: _ValidationState
) -> ValidationResult | None:
    schema_error = _validate_call_schema(event)
    if schema_error:
        return _reject(schema_error, event)
    dependencies = _dependency_keys(event)
    if dependencies is None:
        return _reject("TYPE_COMPLETENESS", event)
    if any(key not in state.completed_results for key in dependencies):
        return _reject("CAUSAL_PREFIX", event)
    correlation = event.correlation
    if not correlation or correlation in state.seen_call_keys:
        return _reject("CALL_RESULT_CONSERVATION", event)
    state.calls += 1
    maximum_calls = task_contract()["maximum_calls"]
    assert isinstance(maximum_calls, int)
    if state.calls > maximum_calls:
        return _reject("CALL_BUDGET", event)
    assert isinstance(event.payload, dict)
    if event.payload.get("name") == "compare_vertical":
        arguments = event.payload["arguments"]
        assert isinstance(arguments, dict)
        if (
            tuple(arguments["red_box"]) != state.current_red_box
            or tuple(arguments["blue_box"]) != state.current_blue_box
        ):
            return _reject("STALE_STATE_ARGUMENTS", event)
    state.seen_call_keys.add(correlation)
    state.unmatched[correlation] = (
        event,
        _execute_call(event, state.current_red_box, state.current_blue_box),
    )
    return None


def _validate_tool_result(
    event: Event, state: _ValidationState
) -> ValidationResult | None:
    if event.actor != "environment" or not isinstance(event.payload, dict):
        return _reject("TYPE_COMPLETENESS", event)
    correlation = event.correlation
    if not correlation or correlation not in state.unmatched:
        return _reject("ORPHAN_RESULT", event)
    call_event, expected = state.unmatched.pop(correlation)
    if event.payload != expected:
        return _reject("TRANSITION_FIDELITY", event)
    assert isinstance(call_event.payload, dict)
    call_name = call_event.payload["name"]
    assert isinstance(call_name, str)
    state.successful_call_types[correlation] = call_name
    state.unresolved_failures.discard(call_name)
    state.completed_results.add(correlation)
    return None


def _validate_tool_error(
    event: Event, state: _ValidationState
) -> ValidationResult | None:
    if event.actor != "environment" or not isinstance(event.payload, dict):
        return _reject("TYPE_COMPLETENESS", event)
    correlation = event.correlation
    if not correlation or correlation not in state.unmatched:
        return _reject("ORPHAN_RESULT", event)
    call_event, _ = state.unmatched.pop(correlation)
    assert isinstance(call_event.payload, dict)
    call_name = call_event.payload["name"]
    assert isinstance(call_name, str)
    state.unresolved_failures.add(call_name)
    state.completed_results.add(correlation)
    return None


def _validate_state_transform(
    event: Event, state: _ValidationState
) -> ValidationResult | None:
    transformed = _apply_state_transform(
        event, state.current_red_box, state.current_blue_box
    )
    if transformed is None:
        return _reject("TRANSITION_FIDELITY", event)
    if state.unmatched:
        return _reject("CAUSAL_PREFIX", event)
    state.current_red_box, state.current_blue_box = transformed
    return None


def _validate_terminal(
    event: Event, state: _ValidationState
) -> ValidationResult | None:
    if event.actor != "assistant" or not isinstance(event.payload, dict):
        return _reject("TYPE_COMPLETENESS", event)
    if state.terminal is not None:
        return _reject("MULTIPLE_TERMINALS", event)
    state.terminal = event
    return None


def _validate_event(event: Event, state: _ValidationState) -> ValidationResult | None:
    if event.kind == "instruction":
        return _validate_instruction(event)
    if event.kind == "tool_call":
        return _validate_tool_call(event, state)
    if event.kind == "tool_result":
        return _validate_tool_result(event, state)
    if event.kind == "tool_error":
        return _validate_tool_error(event, state)
    if event.kind == "state_transform":
        return _validate_state_transform(event, state)
    if event.kind == "terminal_answer":
        return _validate_terminal(event, state)
    return _reject("UNKNOWN_EVENT_TYPE", event)


def _validate_completed_record(state: _ValidationState) -> ValidationResult | None:
    if state.unmatched:
        return _reject("UNMATCHED_CALL")
    terminal = state.terminal
    if terminal is None:
        return _reject("MISSING_TERMINAL")
    assert isinstance(terminal.payload, dict)
    if state.unresolved_failures and terminal.payload.get("relation") == "below":
        return _reject("TERMINAL_AFTER_ERROR", terminal)
    evidence_calls = terminal.payload.get("evidence_calls")
    terminal_is_supported = (
        terminal.payload.get("relation") == "below"
        and terminal.payload.get("center_delta") == 60
        and isinstance(evidence_calls, list)
        and len(evidence_calls) == 2
        and all(isinstance(key, str) for key in evidence_calls)
        and state.successful_call_types.get(evidence_calls[0]) == "locate_shapes"
        and state.successful_call_types.get(evidence_calls[1]) == "compare_vertical"
    )
    if not terminal_is_supported:
        return _reject("TERMINAL_CONSISTENCY", terminal)
    return None


def _validate_certificate(
    events: Sequence[Event], certificate: Mapping[str, object]
) -> ValidationResult | None:
    if certificate.get("validator_version") != VALIDATOR_VERSION:
        return _reject("CERTIFICATE_VERSION")
    if certificate.get("content_hash") != content_hash(events):
        return _reject("STALE_CERTIFICATE")
    return None


def validate_record(
    events: Sequence[Event], certificate: Mapping[str, object] | None = None
) -> ValidationResult:
    """Validate the typed studio trajectory in causal dependency order."""

    if not events:
        return _reject("EMPTY_RECORD")
    if [event.ordinal for event in events] != list(range(len(events))):
        return _reject("ORDINAL_SEQUENCE")

    state = _ValidationState()
    for event in events:
        rejection = _validate_event(event, state)
        if rejection is not None:
            return rejection

    rejection = _validate_completed_record(state)
    if rejection is not None:
        return rejection
    if certificate is not None:
        rejection = _validate_certificate(events, certificate)
        if rejection is not None:
            return rejection
    return ValidationResult(True, "ACCEPT", None)


__all__ = ["certificate_for", "content_hash", "validate_record"]
