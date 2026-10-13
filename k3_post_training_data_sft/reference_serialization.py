"""Canonical symbolic serialization, masks, and loss for Chapter 17."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Mapping, Sequence

from .reference_types import SERIALIZER_VERSION, Event, valid_events
from .reference_validation import validate_record


def serialize_events(events: Sequence[Event]) -> str:
    """Canonical symbolic serialization; explicitly not XTML."""

    envelope = {
        "events": [asdict(event) for event in events],
        "serializer": SERIALIZER_VERSION,
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def parse_events(serialized: str) -> tuple[Event, ...]:
    """Parse the canonical symbolic serialization and reject noncanonical input."""

    value = json.loads(serialized)
    if not isinstance(value, dict) or value.get("serializer") != SERIALIZER_VERSION:
        raise ValueError("unknown serializer")
    raw_events = value.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("events must be a list")
    events: list[Event] = []
    for raw in raw_events:
        if not isinstance(raw, dict) or set(raw) != {
            "ordinal",
            "kind",
            "actor",
            "payload",
            "correlation",
        }:
            raise ValueError("event fields are not canonical")
        events.append(Event(**raw))
    parsed = tuple(events)
    if serialize_events(parsed) != serialized:
        raise ValueError("serialization is not canonical")
    return parsed


def structural_tokens_and_mask(
    events: Sequence[Event],
) -> tuple[list[str], list[int], list[dict[str, object]]]:
    """Map each event to symbolic tokens, a target mask, and structural spans."""

    tokens: list[str] = []
    mask: list[int] = []
    spans: list[dict[str, object]] = []
    for event in events:
        start = len(tokens)
        event_tokens = [
            f"<{event.kind}>",
            f"actor={event.actor}",
            f"correlation={event.correlation or '-'}",
            json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
            f"</{event.kind}>",
        ]
        supervised = int(event.actor == "assistant" and event.kind != "instruction")
        tokens.extend(event_tokens)
        mask.extend([supervised] * len(event_tokens))
        spans.append(
            {
                "ordinal": event.ordinal,
                "kind": event.kind,
                "actor": event.actor,
                "start": start,
                "end_exclusive": len(tokens),
                "supervised": bool(supervised),
            }
        )
    return tokens, mask, spans


def validate_mask(
    tokens: Sequence[str], mask: Sequence[int], spans: Sequence[Mapping[str, object]]
) -> bool:
    """Check exact length and assistant-owned supervision."""

    if len(tokens) != len(mask) or any(value not in (0, 1) for value in mask):
        return False
    covered = [False] * len(tokens)
    for span in spans:
        start = span.get("start")
        end = span.get("end_exclusive")
        supervised = span.get("supervised")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not (0 <= start < end <= len(tokens))
        ):
            return False
        expected = int(supervised is True)
        if any(
            covered[index] or mask[index] != expected for index in range(start, end)
        ):
            return False
        for index in range(start, end):
            covered[index] = True
    return all(covered)


def serialization_fixture() -> dict[str, object]:
    """Return round-trip and mask-alignment evidence."""

    events = valid_events()
    serialized = serialize_events(events)
    parsed = parse_events(serialized)
    tokens, mask, spans = structural_tokens_and_mask(parsed)
    mutated_mask = list(mask)
    mutated_mask[0] = 1
    duplicate_index = list(events)
    call = duplicate_index[3]
    result = duplicate_index[4]
    duplicate_index[3] = Event(call.ordinal, call.kind, call.actor, call.payload, "c1")
    duplicate_index[4] = Event(
        result.ordinal, result.kind, result.actor, result.payload, "c1"
    )
    duplicate_result = validate_record(tuple(duplicate_index))

    unmatched_result = list(events)
    result = unmatched_result[4]
    unmatched_result[4] = Event(
        result.ordinal, result.kind, result.actor, result.payload, "c9"
    )
    unmatched_result_check = validate_record(tuple(unmatched_result))

    missing_boundary_rejected = False
    try:
        parse_events(serialized[:-1])
    except (ValueError, json.JSONDecodeError):
        missing_boundary_rejected = True
    return {
        "serializer_version": SERIALIZER_VERSION,
        "round_trip_equal": parsed == events,
        "canonical_stable": serialize_events(parsed) == serialized,
        "serialized_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "event_count": len(events),
        "token_count": len(tokens),
        "supervised_token_count": sum(mask),
        "span_count": len(spans),
        "mask_valid": validate_mask(tokens, mask, spans),
        "user_target_mutation_rejected": not validate_mask(tokens, mutated_mask, spans),
        "malformed_cases": {
            "duplicate_call_index": duplicate_result.code,
            "unmatched_tool_result": unmatched_result_check.code,
            "missing_message_boundary_rejected": missing_boundary_rejected,
        },
        "tokens": tokens,
        "mask": mask,
        "spans": spans,
    }


def weighted_masked_loss(
    records: Sequence[tuple[Sequence[float], Sequence[int], float]],
) -> float:
    """Compute the globally normalized weighted masked negative log likelihood."""

    numerator = 0.0
    denominator = 0.0
    for probabilities, mask, weight in records:
        if len(probabilities) != len(mask) or weight < 0:
            raise ValueError("invalid probability, mask, or weight shape")
        if any(value <= 0 or value > 1 for value in probabilities):
            raise ValueError("probabilities must lie in (0, 1]")
        if any(value not in (0, 1) for value in mask):
            raise ValueError("mask values must be binary")
        numerator -= weight * sum(
            active * math.log(probability)
            for probability, active in zip(probabilities, mask, strict=True)
        )
        denominator += weight * sum(mask)
    if denominator <= 0:
        raise ValueError("weighted target denominator must be positive")
    return numerator / denominator


def masked_loss_fixture() -> dict[str, object]:
    """Reproduce the two-record weighted-loss example with explicit masks."""

    probabilities_a = (0.8, 0.5, 0.25, 0.9, 0.4, 0.2)
    mask_a = (0, 1, 1, 0, 1, 1)
    probabilities_b = (0.6, 0.3)
    mask_b = (1, 1)
    active_a = tuple(
        probability
        for probability, active in zip(probabilities_a, mask_a, strict=True)
        if active
    )
    active_b = tuple(
        probability
        for probability, active in zip(probabilities_b, mask_b, strict=True)
        if active
    )
    sum_a = -sum(math.log(value) for value in active_a)
    sum_b = -sum(math.log(value) for value in active_b)
    record_a = sum_a / len(active_a)
    record_b = sum_b / len(active_b)
    global_weighted = weighted_masked_loss(
        ((probabilities_a, mask_a, 2), (probabilities_b, mask_b, 1))
    )
    record_mean = (record_a + record_b) / 2
    zero_denominator_rejected = False
    try:
        weighted_masked_loss((((0.8, 0.7), (0, 0), 1),))
    except ValueError:
        zero_denominator_rejected = True
    return {
        "record_a_probabilities": list(probabilities_a),
        "record_a_mask": list(mask_a),
        "record_b_probabilities": list(probabilities_b),
        "record_b_mask": list(mask_b),
        "record_a_loss_sum": sum_a,
        "record_b_loss_sum": sum_b,
        "record_a_normalized": record_a,
        "record_b_normalized": record_b,
        "global_weighted_token_loss": global_weighted,
        "mean_record_normalized_loss": record_mean,
        "weighted_target_units": {"direct_a": 8, "agent_b": 2},
        "zero_denominator_rejected": zero_denominator_rejected,
    }


__all__ = [
    "masked_loss_fixture",
    "parse_events",
    "serialization_fixture",
    "serialize_events",
    "structural_tokens_and_mask",
    "validate_mask",
    "weighted_masked_loss",
]
