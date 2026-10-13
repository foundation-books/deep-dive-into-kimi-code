"""Canonical `studio-execution-001` fixture for Chapter 28."""

from __future__ import annotations

from dataclasses import asdict

from .model import CanonicalCall, Event, Observation, RegistryEntry, digest
from .projection import project_result


FIXTURE_VERSION = "ch27-reference-v1"

ENVIRONMENT = {
    "environment_id": "studio-prod-read",
    "evidence_source": "studio@v7",
    "authorization_ref": "authz-studio-read-001",
    "runtime_config": "runtime-config-v1",
}
ENVIRONMENT_DIGEST = digest(ENVIRONMENT)

REGISTRY_ENTRIES = (
    RegistryEntry(
        name="fetch_evidence",
        interface_version="1",
        executable="evidence-reader@7",
        schema_digest=digest({"atom_id": "string", "source_version": "string"}),
        adapter_version="adapter@2",
        capabilities=("read:evidence",),
        reads=("ledger:studio@v7",),
        writes=(),
        effect_class="pure",
    ),
    RegistryEntry(
        name="run_release_check",
        interface_version="1",
        executable="dependency-checker@4",
        schema_digest=digest(
            {
                "check_id": ["schema", "tests", "dependencies"],
                "timeout_s": "integer",
            }
        ),
        adapter_version="adapter@2",
        capabilities=("read:evidence", "run:release-check"),
        reads=("ledger:studio@v7",),
        writes=(),
        effect_class="pure",
    ),
)
REGISTRY_MAP = {entry.name: entry for entry in REGISTRY_ENTRIES}
REGISTRY_DIGEST = digest(
    {key: REGISTRY_MAP[key].__dict__ for key in sorted(REGISTRY_MAP)}
)

CALL_A = CanonicalCall(
    call_id="call-a",
    name="fetch_evidence",
    interface_version="1",
    arguments={"atom_id": "D3", "source_version": "studio@v7"},
    argument_digest=digest({"atom_id": "D3", "source_version": "studio@v7"}),
    registry_digest=REGISTRY_DIGEST,
    environment_digest=ENVIRONMENT_DIGEST,
)
CALL_B = CanonicalCall(
    call_id="call-b",
    name="run_release_check",
    interface_version="1",
    arguments={"check_id": "dependencies", "timeout_s": 30},
    argument_digest=digest({"check_id": "dependencies", "timeout_s": 30}),
    registry_digest=REGISTRY_DIGEST,
    environment_digest=ENVIRONMENT_DIGEST,
)

RAW_A = {
    "executor_status": "completed",
    "domain_status": "absent",
    "atom_id": "D3",
    "source_version": "studio@v7",
    "present": False,
    "diagnostics": ["dependency sign-off D3 is absent"],
}
PROJECTION_A = project_result(
    RAW_A,
    budget=2048,
    raw_reference="artifact://studio/observation-a-1",
    policy_version="projection-v1",
)
OBSERVATION_A = Observation(
    observation_id="observation-a-1",
    call_id="call-a",
    attempt_id="attempt-a-1",
    executable="evidence-reader@7",
    environment_digest=ENVIRONMENT_DIGEST,
    executor_status="completed",
    domain_status="absent",
    raw=RAW_A,
    raw_digest=digest(RAW_A),
    projection=PROJECTION_A,
    projection_digest=digest(PROJECTION_A),
)

RAW_B = {
    "executor_status": "completed",
    "domain_status": "incomplete",
    "blocking_ids": ["D3"],
    "receipt": "receipt-b-1",
    "diagnostics": ["dependency evidence D3 is absent at studio@v7"],
}
PROJECTION_B = project_result(
    RAW_B,
    budget=2048,
    raw_reference="artifact://studio/observation-b-1",
    policy_version="projection-v1",
)
OBSERVATION_B = Observation(
    observation_id="observation-b-1",
    call_id="call-b",
    attempt_id="attempt-b-1",
    executable="dependency-checker@4",
    environment_digest=ENVIRONMENT_DIGEST,
    executor_status="completed",
    domain_status="incomplete",
    raw=RAW_B,
    raw_digest=digest(RAW_B),
    projection=PROJECTION_B,
    projection_digest=digest(PROJECTION_B),
    recovered=True,
)

ASSISTANT_MESSAGE = {
    "role": "assistant",
    "reasoning_content": (
        "D3 is missing, so I will inspect the versioned evidence and run the "
        "dependency check before preserving a not-ready verdict."
    ),
    "content": "",
    "tool_calls": [
        {
            "id": "call-a",
            "name": "fetch_evidence",
            "arguments": CALL_A.arguments,
        },
        {
            "id": "call-b",
            "name": "run_release_check",
            "arguments": CALL_B.arguments,
        },
    ],
}

FINAL_MESSAGE = {
    "role": "assistant",
    "tool_results": [
        {
            "tool_call_id": "call-a",
            "executor_status": "completed",
            "domain_status": "absent",
            "projection": PROJECTION_A,
        },
        {
            "tool_call_id": "call-b",
            "executor_status": "completed",
            "domain_status": "incomplete",
            "projection": PROJECTION_B,
        },
    ],
    "content": {
        "verdict": "not_ready",
        "blocking_ids": ["D3"],
        "evidence_count": 11,
    },
}

REGISTRY_PAYLOAD = {
    "entries": [asdict(entry) for entry in REGISTRY_ENTRIES],
    "registry_digest": REGISTRY_DIGEST,
    "environment": ENVIRONMENT,
}

STUDIO_EVENTS = (
    Event("e01", "history.append", "assistant-turn", {"message": ASSISTANT_MESSAGE}),
    Event("e02", "registry.freeze", "registry", REGISTRY_PAYLOAD),
    Event(
        "e03",
        "call.create",
        "call-a",
        {"call": asdict(CALL_A)},
        ("e01", "e02"),
    ),
    Event(
        "e04",
        "call.create",
        "call-b",
        {"call": asdict(CALL_B)},
        ("e01", "e02"),
    ),
    Event(
        "e05",
        "attempt.admit",
        "attempt-a-1",
        {
            "attempt_id": "attempt-a-1",
            "call_id": "call-a",
            "number": 1,
            "executor": "evidence-reader@7",
            "checks": {
                "binding": True,
                "arguments": True,
                "authority": True,
                "environment": True,
                "resources": True,
            },
        },
        ("e03",),
    ),
    Event(
        "e06",
        "attempt.admit",
        "attempt-b-1",
        {
            "attempt_id": "attempt-b-1",
            "call_id": "call-b",
            "number": 1,
            "executor": "dependency-checker@4",
            "checks": {
                "binding": True,
                "arguments": True,
                "authority": True,
                "environment": True,
                "resources": True,
                "timeout": True,
            },
        },
        ("e04",),
    ),
    Event(
        "e07",
        "attempt.dispatch",
        "attempt-a-1",
        {"attempt_id": "attempt-a-1"},
        ("e05",),
    ),
    Event(
        "e08",
        "attempt.dispatch",
        "attempt-b-1",
        {"attempt_id": "attempt-b-1"},
        ("e06",),
    ),
    Event(
        "e09",
        "effect.receipt",
        "attempt-b-1",
        {"attempt_id": "attempt-b-1", "receipt": "receipt-b-1"},
        ("e08",),
        observed_at=9,
    ),
    Event(
        "e10",
        "attempt.terminal",
        "attempt-a-1",
        {"attempt_id": "attempt-a-1", "status": "succeeded"},
        ("e07",),
        observed_at=10,
    ),
    Event(
        "e11",
        "observation.record",
        "observation-a-1",
        {"observation": asdict(OBSERVATION_A)},
        ("e10",),
        observed_at=11,
    ),
    Event(
        "e12",
        "recovery.unknown",
        "attempt-b-1",
        {
            "attempt_id": "attempt-b-1",
            "diagnostic": {
                "code": "RESULT_NOT_DURABLE",
                "layer": "observation",
                "effect_knowledge": "unknown",
                "admissible_next_actions": ["reconcile", "stop"],
            },
        },
        ("e09",),
    ),
    Event(
        "e13",
        "observation.record",
        "observation-b-1",
        {"observation": asdict(OBSERVATION_B)},
        ("e12",),
    ),
    Event(
        "e14",
        "handoff.record",
        "studio-execution-001",
        {
            "unknown_outcomes": [],
            "pending": ["D3@new-source-version"],
            "message": FINAL_MESSAGE,
        },
        ("e11", "e13"),
    ),
)

STUDIO_CRASH_PREFIX = STUDIO_EVENTS[:11]

STUDIO_A_FIRST_ORDER = STUDIO_EVENTS
STUDIO_B_FIRST_ORDER = (
    STUDIO_EVENTS[0],
    STUDIO_EVENTS[1],
    STUDIO_EVENTS[3],
    STUDIO_EVENTS[5],
    STUDIO_EVENTS[7],
    STUDIO_EVENTS[8],
    STUDIO_EVENTS[11],
    STUDIO_EVENTS[12],
    STUDIO_EVENTS[2],
    STUDIO_EVENTS[4],
    STUDIO_EVENTS[6],
    STUDIO_EVENTS[9],
    STUDIO_EVENTS[10],
    STUDIO_EVENTS[13],
)

STUDIO_CHECKPOINTS = {
    "K0": STUDIO_EVENTS[:1],
    "K1": STUDIO_EVENTS[:2],
    "K2": STUDIO_EVENTS[:8],
    "K3": STUDIO_EVENTS[:11],
    "K4": STUDIO_EVENTS[:12],
    "K5": STUDIO_EVENTS,
}
