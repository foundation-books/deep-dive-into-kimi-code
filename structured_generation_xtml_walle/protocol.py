"""XTML-oriented events, options, and result matching for Chapter 27."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
from typing import Any, Mapping, Sequence


JSON = Any


@dataclass(frozen=True)
class Event:
    """Represent one typed open, close, or text protocol event."""

    kind: str
    name: str = ""
    attributes: tuple[tuple[str, str], ...] = ()
    text: str = ""


@dataclass(frozen=True)
class Element:
    """Represent a named element with attributes and nested children."""

    name: str
    attributes: tuple[tuple[str, str], ...] = ()
    children: tuple[Any, ...] = ()


def element_events(element: Element) -> tuple[Event, ...]:
    """Serialize to typed events; ordinary text never becomes structure."""

    events: list[Event] = [
        Event("open", element.name, tuple(sorted(element.attributes)))
    ]
    for child in element.children:
        if isinstance(child, Element):
            events.extend(element_events(child))
        elif isinstance(child, str):
            events.append(Event("text", text=child))
        else:
            raise TypeError("element children must be Element or str")
    events.append(Event("close", element.name))
    return tuple(events)


def parse_element_events(
    events: Sequence[Event], start: int = 0
) -> tuple[Element, int]:
    """Parse one element from the protocol event stream."""
    if start >= len(events) or events[start].kind != "open":
        raise ValueError("expected open event")
    opening = events[start]
    children: list[Any] = []
    index = start + 1
    while index < len(events):
        event = events[index]
        if event.kind == "text":
            children.append(event.text)
            index += 1
        elif event.kind == "open":
            child, index = parse_element_events(events, index)
            children.append(child)
        elif event.kind == "close":
            if event.name != opening.name:
                raise ValueError(f"misnested close: {event.name} for {opening.name}")
            return Element(opening.name, opening.attributes, tuple(children)), index + 1
        else:
            raise ValueError(f"unexpected event: {event.kind}")
    raise ValueError(f"unclosed element: {opening.name}")


def round_trip_element(element: Element) -> Element:
    """Round-trip an element through its event representation."""
    events = element_events(element)
    decoded, end = parse_element_events(events)
    if end != len(events):
        raise ValueError("trailing structural events")
    return decoded


@dataclass(frozen=True)
class Option:
    """Bind an option value to a lifetime scope and optional turn."""

    key: str
    value: JSON
    scope: str
    turn: int | None = None


def effective_options(options: Sequence[Option], turn: int) -> dict[str, JSON]:
    """Resolve global, one-shot, and dynamic option lifetimes."""

    effective: dict[str, JSON] = {}
    for option in options:
        if option.scope == "global":
            effective[option.key] = deepcopy(option.value)
        elif (
            option.scope == "dynamic"
            and option.turn is not None
            and option.turn <= turn
        ):
            effective[option.key] = deepcopy(option.value)
        elif option.scope == "one-shot" and option.turn == turn:
            effective[option.key] = deepcopy(option.value)
        elif option.scope not in {"global", "dynamic", "one-shot"}:
            raise ValueError(f"unknown option scope: {option.scope}")
    return effective


def _is_number(value: JSON) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def xtml_scalar_type(value: JSON) -> str:
    """Return the xTML scalar type name for a JSON value."""
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if _is_number(value):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _assert_json_value(value: JSON, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string JSON object key at {path}")
            _assert_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def xtml_argument_text(value: JSON) -> str:
    """Serialize a JSON value as canonical xTML argument text."""
    _assert_json_value(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class Call:
    """Capture one ordered tool call and its optional external identity."""

    external_id: str | None
    tool: str
    arguments: Mapping[str, JSON]
    index: int


@dataclass(frozen=True)
class Result:
    """Capture a tool result and optional call identity or order metadata."""

    external_id: str | None
    tool: str | None
    payload: JSON
    index: int | None = None


@dataclass(frozen=True)
class NormalizationOutcome:
    """Report normalized results or the diagnostic preventing normalization."""

    status: str
    results: tuple[Result, ...]
    diagnostic: str


@dataclass(frozen=True)
class ProtocolTrace:
    """Finite protocol trace used by the chapter's executable oracle."""

    zones: tuple[str, ...]
    channels: tuple[str, ...]
    calls: tuple[Call, ...]
    results: tuple[Result, ...]
    options: tuple[Option, ...] = ()
    turn: int = 1
    required_tool_call: bool = True
    structurally_closed: bool = True


@dataclass(frozen=True)
class ProductState:
    """Product state (XTML, payload grammar, options, pending calls)."""

    xtml_state: str
    grammar_prefix: bytes
    options: tuple[Option, ...]
    pending_calls: tuple[Call, ...]


def _identity_map(calls: Sequence[Call]) -> tuple[dict[str, Call], str | None]:
    identity: dict[str, Call] = {}
    for expected, call in enumerate(calls, start=1):
        if call.index != expected:
            return {}, "nonconsecutive-call-index"
        if call.external_id is None:
            continue
        if call.external_id in identity:
            return {}, "ambiguous-call-id"
        identity[call.external_id] = call
    return identity, None


def normalize_results(
    calls: Sequence[Call], results: Sequence[Result]
) -> NormalizationOutcome:
    """Resolve a whole result run by opaque ID before reordering it."""

    calls_copy = deepcopy(tuple(calls))
    results_copy = deepcopy(tuple(results))
    identity, error = _identity_map(calls_copy)
    if error:
        return NormalizationOutcome("error", results_copy, error)
    resolved: dict[int, Result] = {}
    for result in results_copy:
        if result.external_id is None or result.external_id not in identity:
            return NormalizationOutcome(
                "unresolved", results_copy, "unresolved-result-run"
            )
        call = identity[result.external_id]
        if result.tool is not None and result.tool != call.tool:
            return NormalizationOutcome("error", results_copy, "result-tool-mismatch")
        if call.index in resolved:
            return NormalizationOutcome("error", results_copy, "duplicate-result")
        resolved[call.index] = replace(result, tool=call.tool, index=call.index)
    ordered = tuple(resolved[index] for index in sorted(resolved))
    if len(ordered) != len(calls_copy):
        return NormalizationOutcome("open", ordered, "missing-result")
    return NormalizationOutcome("closed", ordered, "resolved-in-call-order")


def zone_sequence_errors(zones: Sequence[str]) -> tuple[str, ...]:
    """Check the chapter's global/history/one-shot/assistant zone order."""

    rank = {
        "global-options": 0,
        "history": 1,
        "dynamic-options": 1,
        "one-shot-options": 2,
        "assistant": 3,
    }
    errors: list[str] = []
    previous = -1
    history_seen = False
    for zone in zones:
        if zone not in rank:
            errors.append(f"unknown-zone:{zone}")
            continue
        if rank[zone] < previous:
            errors.append(f"zone-order:{zone}")
        if zone == "dynamic-options" and not history_seen:
            errors.append("dynamic-options-before-history")
        if zone == "one-shot-options" and not history_seen:
            errors.append("one-shot-options-before-history")
        history_seen = history_seen or zone == "history"
        previous = max(previous, rank[zone])
    if not zones or zones[-1] != "assistant":
        errors.append("assistant-zone-not-final")
    return tuple(errors)


def _channel_errors(trace: ProtocolTrace) -> list[str]:
    channel_rank = {"thinking": 0, "tools": 1, "tool-results": 2, "response": 3}
    errors: list[str] = []
    previous = -1
    for channel in trace.channels:
        if channel not in channel_rank:
            errors.append(f"unknown-channel:{channel}")
            continue
        if channel_rank[channel] < previous:
            errors.append(f"channel-order:{channel}")
        previous = max(previous, channel_rank[channel])
    if trace.channels.count("response") != 1 or trace.channels[-1:] != ("response",):
        errors.append("response-channel-not-unique-final")
    if trace.required_tool_call and "tools" not in trace.channels:
        errors.append("required-tool-call-missing")
    if trace.calls and "tools" not in trace.channels:
        errors.append("calls-without-tools-channel")
    if trace.results and "tool-results" not in trace.channels:
        errors.append("results-without-tool-results-channel")
    return errors


def _option_errors(trace: ProtocolTrace) -> list[str]:
    errors: list[str] = []
    for option in trace.options:
        if option.scope == "one-shot" and option.turn != trace.turn:
            errors.append(f"one-shot-option-out-of-turn:{option.key}")
        if option.scope not in {"global", "dynamic", "one-shot"}:
            errors.append(f"unknown-option-scope:{option.scope}")
    return errors


def protocol_errors(trace: ProtocolTrace) -> tuple[str, ...]:
    """Validate structural closure, channel order, scopes, and call identity."""

    errors = list(zone_sequence_errors(trace.zones))
    if not trace.structurally_closed:
        errors.append("unclosed-xtml")
    errors.extend(_channel_errors(trace))
    errors.extend(_option_errors(trace))

    outcome = normalize_results(trace.calls, trace.results)
    if outcome.status != "closed":
        errors.append(f"result-run:{outcome.diagnostic}")
    return tuple(errors)
