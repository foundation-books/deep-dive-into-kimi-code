"""Deterministic XTML and product checks CH26-D06 through CH26-D10."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

from .fixture import (
    CORRUPTION_FIRST_FAILURE,
    STUDIO_RECORD,
    json_bytes,
    require,
    response_recognizer,
    studio_calls,
    studio_options,
    studio_protocol,
    studio_results,
    studio_validity,
)
from .protocol import (
    Element,
    Event,
    Option,
    ProductState,
    effective_options,
    element_events,
    normalize_results,
    parse_element_events,
    protocol_errors,
    round_trip_element,
    xtml_argument_text,
    xtml_scalar_type,
    zone_sequence_errors,
)
from .schema import (
    STUDIO_RESPONSE,
    STUDIO_RESPONSE_SCHEMA,
    instance_conforms,
    instance_errors,
)


JSON = Any


def xtml_round_trip_check() -> dict[str, JSON]:
    """Return diagnostics for the xTML round trip invariant."""
    tree = Element(
        "assistant",
        children=(
            Element("response", children=("literal <|open|> remains ordinary text",)),
        ),
    )
    events = element_events(tree)
    failures: list[str] = []
    for malformed in (
        (Event("open", "response"),),
        (Event("open", "response"), Event("close", "tools")),
    ):
        try:
            parse_element_events(malformed)
        except ValueError as error:
            failures.append(str(error))
    round_trip = round_trip_element(tree) == tree
    ordinary_special_text = events[2].kind == "text"
    payload = {
        "round_trip": round_trip,
        "event_kinds": [event.kind for event in events],
        "ordinary_special_text": ordinary_special_text,
        "malformed_diagnostics": failures,
    }
    require(round_trip, "XTML-oriented event round trip failed")
    require(ordinary_special_text, "content became structure")
    require(len(failures) == 2, "malformed XTML cases were not rejected")
    return payload


def option_scope_check() -> dict[str, JSON]:
    """Return diagnostics for the option scope invariant."""
    options = (
        Option("tools", ["A", "B"], "global"),
        Option("thinking_effort", "high", "global"),
        Option("tools", ["A", "B", "C"], "dynamic", 2),
        Option("response_schema", "S2", "one-shot", 2),
        Option("response_schema", "S3", "one-shot", 3),
    )
    effective = {str(turn): effective_options(options, turn) for turn in (1, 2, 3)}
    valid_errors = protocol_errors(studio_protocol())
    bad_zone_errors = zone_sequence_errors(
        ("global-options", "one-shot-options", "history", "assistant")
    )
    no_tool_errors = protocol_errors(
        replace(
            studio_protocol(),
            channels=("response",),
            calls=(),
            results=(),
        )
    )
    require(effective["1"]["tools"] == ["A", "B"], "global option drifted")
    require(effective["2"]["response_schema"] == "S2", "one-shot S2 missing")
    require(effective["3"]["response_schema"] == "S3", "one-shot S3 missing")
    require(not valid_errors, f"canonical protocol failed: {valid_errors}")
    require(bool(bad_zone_errors), "illegal zone order passed")
    require(
        "required-tool-call-missing" in no_tool_errors, "required call not enforced"
    )
    return {
        "effective": effective,
        "canonical_protocol_errors": list(valid_errors),
        "illegal_zone_errors": list(bad_zone_errors),
        "missing_required_tool_errors": list(no_tool_errors),
    }


def typed_argument_check() -> dict[str, JSON]:
    """Return diagnostics for the typed argument invariant."""
    values = ("30", 30, True, None, [1, 2], {"x": 1})
    quoted = {**STUDIO_RESPONSE, "evidence_count": "11"}
    malformed: list[str] = []
    for value in ({1: "not-json"}, math.nan, (1, 2)):
        try:
            xtml_argument_text(value)
        except (TypeError, ValueError) as error:
            malformed.append(str(error))
    argument_types = [xtml_scalar_type(value) for value in values]
    studio_response_conforms = instance_conforms(
        STUDIO_RESPONSE, STUDIO_RESPONSE_SCHEMA
    )
    payload = {
        "types": argument_types,
        "texts": [xtml_argument_text(value) for value in values],
        "studio_response_conforms": studio_response_conforms,
        "quoted_count_errors": list(instance_errors(quoted, STUDIO_RESPONSE_SCHEMA)),
        "malformed_json_diagnostics": malformed,
    }
    require(
        argument_types == ["string", "number", "boolean", "null", "array", "object"],
        "typed argument labels drifted",
    )
    require(studio_response_conforms, "canonical response failed schema")
    require(len(malformed) == 3, "malformed JSON argument was serialized")
    return payload


def result_matching_check() -> dict[str, JSON]:
    """Return diagnostics for the result matching invariant."""
    calls, results = studio_calls(), studio_results()
    original = deepcopy(results)
    normalized = normalize_results(calls, results)
    repeated = normalize_results(calls, normalized.results)
    cases = {
        "duplicate": normalize_results(calls, results + (results[0],)).diagnostic,
        "unresolved": normalize_results(
            calls, (replace(results[0], external_id="stale"),)
        ).diagnostic,
        "missing": normalize_results(calls, results[:1]).diagnostic,
        "ambiguous_call_id": normalize_results(
            (calls[0], replace(calls[1], external_id="call-a")), results
        ).diagnostic,
        "nonconsecutive_index": normalize_results(
            (calls[0], replace(calls[1], index=3)), results
        ).diagnostic,
        "tool_mismatch": normalize_results(
            calls, (replace(results[0], tool="wrong_tool"), results[1])
        ).diagnostic,
    }
    expected = {
        "duplicate": "duplicate-result",
        "unresolved": "unresolved-result-run",
        "missing": "missing-result",
        "ambiguous_call_id": "ambiguous-call-id",
        "nonconsecutive_index": "nonconsecutive-call-index",
        "tool_mismatch": "result-tool-mismatch",
    }
    require(normalized.status == "closed", "canonical results did not close")
    require([result.index for result in normalized.results] == [1, 2], "bad order")
    require(repeated == normalized, "normalization is not idempotent")
    require(results == original, "normalization mutated arrival records")
    require(cases == expected, f"matching diagnostics drifted: {cases}")
    return {
        "normalization_status": normalized.status,
        "indices": [result.index for result in normalized.results],
        "tools": [result.tool for result in normalized.results],
        "idempotent": repeated == normalized,
        "input_unmodified": results == original,
        "failure_diagnostics": cases,
    }


def _first_failed(vector: dict[str, bool | None]) -> str | None:
    return next(
        (f"V{index}" for index in range(8) if vector[f"V{index}"] is False), None
    )


def _state_payload(state: ProductState) -> dict[str, JSON]:
    return {
        "xtml_state": state.xtml_state,
        "grammar_prefix": state.grammar_prefix.decode("utf-8"),
        "effective_option_keys": sorted(effective_options(state.options, 1)),
        "pending_ids": [call.external_id for call in state.pending_calls],
    }


def product_trace_check() -> dict[str, JSON]:
    """Return diagnostics for the product trace invariant."""
    canonical = studio_validity()
    expected_canonical = {**{f"V{index}": True for index in range(7)}, "V7": None}
    require(canonical == expected_canonical, f"canonical vector drifted: {canonical}")

    corruptions: dict[str, JSON] = {}
    for index, expected in CORRUPTION_FIRST_FAILURE.items():
        vector = studio_validity(index)
        actual = _first_failed(vector)
        require(
            actual == expected, f"corruption {index}: expected {expected}, got {actual}"
        )
        corruptions[str(index)] = {
            "expected_first_failure": expected,
            "actual_first_failure": actual,
            "vector": vector,
        }

    calls = studio_calls()
    normalized = normalize_results(calls, studio_results())
    require(normalized.status == "closed", "product trace has an open result run")
    options = studio_options()
    response_bytes = json_bytes(STUDIO_RECORD["response"])
    states = (
        ProductState("assistant-ready", b"", options, ()),
        ProductState("calls-open", b"", options, calls),
        ProductState("results-matched", b"", options, ()),
        ProductState("response-closed", response_bytes, options, ()),
    )
    require(states[1].pending_calls == calls, "calls were not registered as pending")
    require(not states[2].pending_calls, "matched calls remained pending")
    require(
        response_recognizer().end_is_legal(states[3].grammar_prefix),
        "closed product state has an incomplete payload",
    )
    return {
        "record_id": STUDIO_RECORD["record_id"],
        "source_version": STUDIO_RECORD["source_version"],
        "canonical": canonical,
        "corruptions": corruptions,
        "states": [_state_payload(state) for state in states],
        "handoff": {
            "semantic_blocker": "D3",
            "evidence_count": len(STUDIO_RECORD["evidence"]["present"]),
            "runtime_observation": None,
            "next_checker": "runtime-execution-layer",
        },
    }
