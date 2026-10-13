"""Canonical running fixture and evaluated validity vectors for Chapter 27."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from typing import Any, Callable

from .decoder import FiniteLanguageRecognizer, evaluate_token_trace
from .protocol import (
    Call,
    Option,
    ProtocolTrace,
    Result,
    protocol_errors,
)
from .schema import (
    STUDIO_RESPONSE,
    STUDIO_RESPONSE_SCHEMA,
    CanonicalizationTrace,
    Diagnostic,
    canonicalize_to_fixed_point,
    instance_conforms,
    validate_schema_profile,
)


JSON = Any

# RE26.01 is stored once as an aggregate. The schema and response objects are
# shared with schema.py so focused validator checks and the trace cannot drift.
STUDIO_RECORD: dict[str, JSON] = {
    "record_id": "studio-contract-001",
    "source_version": "studio@v7",
    "evidence": {
        "present": [
            "R1",
            "R2",
            "R3",
            "C1",
            "C2",
            "C3",
            "V1",
            "V2",
            "V3",
            "D1",
            "D2",
        ],
        "missing": ["D3"],
    },
    "tools": [
        {
            "name": "fetch_evidence",
            "arguments": {"atom_id": "string", "source_version": "string"},
        },
        {
            "name": "run_release_check",
            "arguments": {
                "check_id": "schema | tests | dependencies",
                "timeout_s": "integer",
            },
        },
    ],
    "response_schema": STUDIO_RESPONSE_SCHEMA,
    "options": [
        {
            "key": "tools",
            "value": ["fetch_evidence", "run_release_check"],
            "scope": "global",
            "turn": None,
        },
        {
            "key": "thinking_effort",
            "value": "high",
            "scope": "global",
            "turn": None,
        },
        {
            "key": "tool_choice",
            "value": "required",
            "scope": "one-shot",
            "turn": 1,
        },
        {
            "key": "response_schema",
            "value": STUDIO_RESPONSE_SCHEMA,
            "scope": "one-shot",
            "turn": 1,
        },
    ],
    "zones": ["global-options", "history", "one-shot-options", "assistant"],
    "channels": ["tools", "tool-results", "response"],
    "calls": [
        {
            "external_id": "call-a",
            "tool": "fetch_evidence",
            "arguments": {"atom_id": "D3", "source_version": "studio@v7"},
            "index": 1,
        },
        {
            "external_id": "call-b",
            "tool": "run_release_check",
            "arguments": {"check_id": "dependencies", "timeout_s": 30},
            "index": 2,
        },
    ],
    "result_arrivals": [
        {
            "external_id": "call-b",
            "tool": None,
            "payload": {"status": "incomplete", "blocking_ids": ["D3"]},
        },
        {
            "external_id": "call-a",
            "tool": None,
            "payload": {"present": False},
        },
    ],
    "response": STUDIO_RESPONSE,
}


CORRUPTION_FIRST_FAILURE = {
    1: "V1",
    2: "V2",
    3: "V3",
    4: "V3",
    5: "V3",
    6: "V5",
    7: "V5",
    8: "V5",
    9: "V5",
    10: "V6",
}


def require(condition: bool, diagnostic: str) -> None:
    """Raise an assertion when a fixture condition is false."""
    if not condition:
        raise AssertionError(diagnostic)


def json_bytes(value: JSON) -> bytes:
    """Encode a JSON value using the canonical fixture spelling."""
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def studio_calls() -> tuple[Call, ...]:
    """Build the deterministic calls teaching fixture."""
    return tuple(Call(**deepcopy(item)) for item in STUDIO_RECORD["calls"])


def studio_results() -> tuple[Result, ...]:
    """Build the deterministic results teaching fixture."""
    return tuple(Result(**deepcopy(item)) for item in STUDIO_RECORD["result_arrivals"])


def studio_options() -> tuple[Option, ...]:
    """Build the deterministic options teaching fixture."""
    return tuple(Option(**deepcopy(item)) for item in STUDIO_RECORD["options"])


def studio_protocol(
    *,
    calls: tuple[Call, ...] | None = None,
    results: tuple[Result, ...] | None = None,
    channels: tuple[str, ...] | None = None,
    structurally_closed: bool = True,
) -> ProtocolTrace:
    """Build the deterministic protocol teaching fixture."""
    return ProtocolTrace(
        zones=tuple(STUDIO_RECORD["zones"]),
        channels=(tuple(STUDIO_RECORD["channels"]) if channels is None else channels),
        calls=(studio_calls() if calls is None else calls),
        results=(studio_results() if results is None else results),
        options=studio_options(),
        turn=1,
        required_tool_call=True,
        structurally_closed=structurally_closed,
    )


def ready_response() -> dict[str, JSON]:
    """Return the accepted studio response for complete evidence."""
    return {
        "verdict": "ready",
        "blocking_ids": [],
        "evidence_count": 11,
        "summary": "All release evidence is present at studio@v7.",
    }


def response_recognizer() -> FiniteLanguageRecognizer:
    """Build the finite recognizer for the two legal studio responses."""
    return FiniteLanguageRecognizer(
        (json_bytes(STUDIO_RECORD["response"]), json_bytes(ready_response()))
    )


def semantic_correct(response: dict[str, JSON]) -> bool:
    """Return whether a response matches the studio evidence state."""
    evidence = STUDIO_RECORD["evidence"]
    missing = list(evidence["missing"])
    present_count = len(evidence["present"])
    if missing:
        return (
            response.get("verdict") == "not_ready"
            and response.get("blocking_ids") == missing
            and response.get("evidence_count") == present_count
            and all(atom in response.get("summary", "") for atom in missing)
            and STUDIO_RECORD["source_version"] in response.get("summary", "")
        )
    return (
        response.get("verdict") == "ready"
        and response.get("blocking_ids") == []
        and response.get("evidence_count") == present_count
    )


@dataclass
class _StudioCase:
    schema: dict[str, JSON]
    response: dict[str, JSON]
    calls: tuple[Call, ...]
    results: tuple[Result, ...]
    channels: tuple[str, ...]
    structurally_closed: bool = True
    reject_canonical: bool = False


def _unsupported_schema(case: _StudioCase) -> None:
    case.schema["oneOf"] = [{"type": "object"}]


def _rejected_canonical(case: _StudioCase) -> None:
    case.reject_canonical = True


def _quoted_count(case: _StudioCase) -> None:
    case.response["evidence_count"] = "11"


def _missing_summary(case: _StudioCase) -> None:
    case.response.pop("summary")


def _extra_property(case: _StudioCase) -> None:
    case.response["confidence"] = 0.9


def _unclosed_xtml(case: _StudioCase) -> None:
    case.structurally_closed = False


def _reused_index(case: _StudioCase) -> None:
    case.calls = (case.calls[0], replace(case.calls[1], index=1))


def _stale_result(case: _StudioCase) -> None:
    case.results = (
        replace(case.results[0], external_id="stale-call-3"),
        case.results[1],
    )


def _early_response(case: _StudioCase) -> None:
    case.channels = ("response", "tools", "tool-results")


def _wrong_semantics(case: _StudioCase) -> None:
    case.response = ready_response()


_CORRUPTORS: dict[int, Callable[[_StudioCase], None]] = {
    1: _unsupported_schema,
    2: _rejected_canonical,
    3: _quoted_count,
    4: _missing_summary,
    5: _extra_property,
    6: _unclosed_xtml,
    7: _reused_index,
    8: _stale_result,
    9: _early_response,
    10: _wrong_semantics,
}


def _studio_case(corruption: int | None) -> _StudioCase:
    if corruption is not None and corruption not in _CORRUPTORS:
        raise ValueError("corruption must be 1 through 10")
    case = _StudioCase(
        deepcopy(STUDIO_RECORD["response_schema"]),
        deepcopy(STUDIO_RECORD["response"]),
        studio_calls(),
        studio_results(),
        tuple(STUDIO_RECORD["channels"]),
    )
    if corruption is not None:
        _CORRUPTORS[corruption](case)
    return case


def _canonical_trace(case: _StudioCase) -> CanonicalizationTrace:
    if case.reject_canonical:
        return CanonicalizationTrace(
            deepcopy(case.schema),
            Diagnostic("error", ("injected-caller-must-reject",)),
            1,
            "rejected",
        )
    return canonicalize_to_fixed_point(case.schema)


def studio_validity(corruption: int | None = None) -> dict[str, bool | None]:
    """Evaluate the fixture instead of assigning an expected failure label."""

    case = _studio_case(corruption)
    vector: dict[str, bool | None] = {f"V{index}": None for index in range(8)}
    try:
        serialized = json_bytes(case.response)
        json.loads(serialized)
        vector["V0"] = True
    except (TypeError, ValueError, json.JSONDecodeError):
        vector["V0"] = False
        return vector

    vector["V1"] = not validate_schema_profile(case.schema, "ultra")
    if not vector["V1"]:
        return vector
    canonical = _canonical_trace(case)
    vector["V2"] = canonical.status == "stable" and canonical.diagnostic.acceptable
    if not vector["V2"]:
        return vector
    vector["V3"] = instance_conforms(case.response, canonical.candidate)
    if not vector["V3"]:
        return vector
    decoded = evaluate_token_trace(
        response_recognizer(), (serialized,), end_requested=True
    )
    vector["V4"] = decoded.status == "accepted"
    if not vector["V4"]:
        return vector
    trace = studio_protocol(
        calls=case.calls,
        results=case.results,
        channels=case.channels,
        structurally_closed=case.structurally_closed,
    )
    vector["V5"] = not protocol_errors(trace)
    if not vector["V5"]:
        return vector
    vector["V6"] = semantic_correct(case.response)
    vector["V7"] = None
    return vector
