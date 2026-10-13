"""Ten deterministic checks corresponding to Chapter 28's evidence table."""

from __future__ import annotations

from dataclasses import asdict, replace

from .effects import idempotency_witness, recovery_action
from .fixture import (
    CALL_A,
    FIXTURE_VERSION,
    OBSERVATION_A,
    STUDIO_A_FIRST_ORDER,
    STUDIO_B_FIRST_ORDER,
    STUDIO_CHECKPOINTS,
    STUDIO_CRASH_PREFIX,
    STUDIO_EVENTS,
)
from .handoff import (
    INVARIANT_COVERAGE,
    checkpoint_summaries,
    expected_checkpoint_summaries,
    unresolved_attempts,
)
from .model import Event, canonical_json, digest
from .projection import project_result
from .runtime import (
    ExecutionContractError,
    all_topological_orders,
    fold_events,
    preserved_history_errors,
    replace_event,
    topological_order,
)
from .scheduler import (
    Access,
    conflict_edges,
    conflicts,
    provider_order,
    schedule_batches,
)


def _result(check_id: str, **details: object) -> dict[str, object]:
    return {
        "check_id": check_id,
        "fixture_version": FIXTURE_VERSION,
        "status": "pass",
        **details,
    }


def _raises(events: tuple[Event, ...], fragment: str) -> str:
    try:
        fold_events(events)
    except ExecutionContractError as error:
        message = str(error)
        if fragment not in message:
            raise AssertionError(f"expected {fragment!r} in {message!r}") from error
        return message
    raise AssertionError(f"expected ExecutionContractError containing {fragment!r}")


def check_event_graph() -> dict[str, object]:
    """Return diagnostics for the event graph invariant."""
    ordered = topological_order(reversed(STUDIO_EVENTS))
    assert len(ordered) == 14
    duplicate = STUDIO_EVENTS + (STUDIO_EVENTS[0],)
    duplicate_error = _raises(duplicate, "duplicate event ID")
    unknown_parent = replace_event(STUDIO_EVENTS[0], parents=("absent",))
    parent_error = _raises((unknown_parent,), "unknown parents")
    return _result(
        "CH27-D01",
        event_count=len(ordered),
        first_event=ordered[0].event_id,
        last_event=ordered[-1].event_id,
        duplicate_error=duplicate_error,
        parent_error=parent_error,
    )


def check_fold_replay() -> dict[str, object]:
    """Return diagnostics for the fold replay invariant."""
    orders = all_topological_orders(STUDIO_EVENTS)
    digests = {fold_events(order, ordered=True).state_digest() for order in orders}
    left = fold_events(STUDIO_A_FIRST_ORDER, ordered=True)
    right = fold_events(STUDIO_B_FIRST_ORDER, ordered=True)
    assert len(orders) == 924
    assert len(digests) == 1
    assert left.canonical_value() == right.canonical_value()
    assert left.state_digest() == right.state_digest()
    return _result(
        "CH27-D02",
        admissible_orders=len(orders),
        distinct_state_digests=len(digests),
        canonical_state_digest=left.state_digest(),
        model_calls=0,
        tool_calls=0,
        appended_events=0,
    )


def check_reasoning_history() -> dict[str, object]:
    """Return diagnostics for the reasoning history invariant."""
    state = fold_events(STUDIO_EVENTS)
    assert preserved_history_errors(state) == ()
    broken_message = dict(STUDIO_EVENTS[0].payload["message"])
    broken_message.pop("reasoning_content")
    broken_event = replace_event(STUDIO_EVENTS[0], payload={"message": broken_message})
    broken = fold_events((broken_event,) + STUDIO_EVENTS[1:])
    errors = preserved_history_errors(broken)
    assert errors == ("assistant-missing:reasoning_content",)
    return _result(
        "CH27-D03",
        complete_fields=["role", "reasoning_content", "content", "tool_calls"],
        closed_call_ids=["call-a", "call-b"],
        corruption_errors=list(errors),
    )


def check_binding_admission() -> dict[str, object]:
    """Return diagnostics for the binding admission invariant."""
    state = fold_events(STUDIO_EVENTS[:8])
    assert len(state.registry) == 2
    assert set(state.calls) == {"call-a", "call-b"}
    assert {attempt.status for attempt in state.attempts.values()} == {"dispatched"}
    corrupt_call = replace(
        CALL_A, environment_digest=digest({"environment_id": "wrong"})
    )
    corrupt_event = replace_event(
        STUDIO_EVENTS[2], payload={"call": asdict(corrupt_call)}
    )
    environment_error = _raises(
        STUDIO_EVENTS[:2] + (corrupt_event,), "environment digest mismatch"
    )
    denied = replace_event(
        STUDIO_EVENTS[4],
        payload={**STUDIO_EVENTS[4].payload, "checks": {"authority": False}},
    )
    admission_error = _raises(STUDIO_EVENTS[:4] + (denied,), "failed precondition")
    return _result(
        "CH27-D04",
        registry_entries=sorted(state.registry),
        calls=sorted(state.calls),
        dispatched_attempts=sorted(state.attempts),
        environment_error=environment_error,
        admission_error=admission_error,
    )


def check_call_attempt_lifecycle() -> dict[str, object]:
    """Return diagnostics for the call attempt lifecycle invariant."""
    state = fold_events(STUDIO_EVENTS)
    assert state.attempts["attempt-a-1"].status == "succeeded"
    assert state.attempts["attempt-b-1"].status == "unknown"
    assert state.attempts["attempt-b-1"].receipt == "receipt-b-1"
    double_dispatch = STUDIO_EVENTS[:8] + (
        Event(
            "e08x",
            "attempt.dispatch",
            "attempt-a-1",
            {"attempt_id": "attempt-a-1"},
            ("e07",),
        ),
    )
    dispatch_error = _raises(double_dispatch, "requires admitted attempt")
    changed = {**CALL_A.arguments, "source_version": "studio@v8"}
    assert digest(changed) != CALL_A.argument_digest
    return _result(
        "CH27-D05",
        logical_calls=2,
        physical_attempts=2,
        statuses={key: value.status for key, value in state.attempts.items()},
        changed_arguments_create_new_call=True,
        double_dispatch_error=dispatch_error,
    )


def check_effect_recovery() -> dict[str, object]:
    """Return diagnostics for the effect recovery invariant."""
    keyed = idempotency_witness("keyed-insert")
    counter = idempotency_witness("unkeyed-increment")
    assert keyed["idempotent"] is True
    assert counter["idempotent"] is False
    assert "retry-same-key" in recovery_action("unknown", "idempotent")
    assert "retry-same-key" not in recovery_action("unknown", "nonrepeatable")
    crash = fold_events(STUDIO_CRASH_PREFIX)
    assert crash.attempts["attempt-b-1"].status == "dispatched"
    assert crash.attempts["attempt-b-1"].receipt == "receipt-b-1"
    recovered = fold_events(STUDIO_EVENTS)
    assert recovered.observations["observation-b-1"].recovered is True
    return _result(
        "CH27-D06",
        keyed_insert=keyed,
        unkeyed_increment=counter,
        nonrepeatable_unknown_actions=list(recovery_action("unknown", "nonrepeatable")),
        crash_status="dispatched-with-receipt-no-result",
        recovered_domain_status="incomplete",
        duplicate_attempts=0,
    )


def check_result_projection() -> dict[str, object]:
    """Return diagnostics for the result projection invariant."""
    raw = {
        "executor_status": "completed",
        "domain_status": "incomplete",
        "blocking_ids": ["D3"],
        "diagnostics": ["x" * 1000, "y" * 1000, "z" * 1000],
    }
    projected = project_result(
        raw,
        budget=700,
        raw_reference="artifact://large-report",
        policy_version="projection-v1",
    )
    assert projected["truncated"] is True
    assert projected["domain_status"] == raw["domain_status"]
    assert projected["raw_digest"] == digest(raw)
    projected_bytes = len(canonical_json(projected).encode("utf-8"))
    assert projected_bytes <= 700
    tampered = replace(OBSERVATION_A, raw={**OBSERVATION_A.raw, "present": True})
    tampered_event = replace_event(
        STUDIO_EVENTS[10], payload={"observation": asdict(tampered)}
    )
    tamper_error = _raises(
        STUDIO_EVENTS[:10] + (tampered_event,), "raw payload digest mismatch"
    )
    return _result(
        "CH27-D07",
        truncated=projected["truncated"],
        omitted_diagnostics=projected["omitted_diagnostics"],
        projected_bytes=projected_bytes,
        budget_bytes=700,
        status_preserved=True,
        raw_digest_preserved=True,
        tamper_error=tamper_error,
    )


def check_parallel_schedule() -> dict[str, object]:
    """Return diagnostics for the parallel schedule invariant."""
    accesses = (
        Access("A", frozenset({"x"}), frozenset(), 1),
        Access("B", frozenset({"x"}), frozenset(), 2),
        Access("C", frozenset(), frozenset({"x"}), 3),
        Access("D", frozenset(), frozenset({"y"}), 4),
        Access("E", frozenset({"x", "y"}), frozenset(), 5),
    )
    edges = conflict_edges(accesses)
    assert edges == (("A", "C"), ("B", "C"), ("C", "E"), ("D", "E"))
    relationship_cases = {
        "read_read": conflicts(accesses[0], accesses[1]),
        "read_write": conflicts(accesses[0], accesses[2]),
        "write_write": conflicts(
            Access("F", frozenset(), frozenset({"z"}), 6),
            Access("G", frozenset(), frozenset({"z"}), 7),
        ),
        "disjoint": conflicts(accesses[2], accesses[3]),
    }
    assert relationship_cases == {
        "read_read": False,
        "read_write": True,
        "write_write": True,
        "disjoint": False,
    }
    batches = schedule_batches(accesses)
    assert len(batches) == 2
    assert provider_order(("E", "B", "A", "D", "C"), accesses) == (
        "A",
        "B",
        "C",
        "D",
        "E",
    )
    return _result(
        "CH27-D08",
        conflict_edges=[list(edge) for edge in edges],
        relationship_cases=relationship_cases,
        minimum_batches=[list(batch) for batch in batches],
        raw_completion_order=["E", "B", "A", "D", "C"],
        provider_order=["A", "B", "C", "D", "E"],
    )


def check_invalid_transitions() -> dict[str, object]:
    """Return diagnostics for the invalid transitions invariant."""
    dispatch_without_admission = (
        STUDIO_EVENTS[0],
        STUDIO_EVENTS[1],
        STUDIO_EVENTS[2],
        Event(
            "bad-dispatch",
            "attempt.dispatch",
            "missing-attempt",
            {"attempt_id": "missing-attempt"},
            ("e03",),
        ),
    )
    result_without_attempt = Event(
        "bad-result",
        "observation.record",
        "observation-a-1",
        {"observation": asdict(OBSERVATION_A)},
        ("e03",),
    )
    first = _raises(dispatch_without_admission, "unknown attempt")
    second = _raises(STUDIO_EVENTS[:3] + (result_without_attempt,), "unknown attempt")
    terminal_reversal = STUDIO_EVENTS[:10] + (
        Event(
            "bad-terminal",
            "attempt.terminal",
            "attempt-a-1",
            {"attempt_id": "attempt-a-1", "status": "failed"},
            ("e10",),
        ),
    )
    third = _raises(terminal_reversal, "requires dispatched attempt")
    return _result(
        "CH27-D09",
        rejected=[first, second, third],
        invalid_transition_count=3,
    )


def check_handoff() -> dict[str, object]:
    """Return diagnostics for the handoff invariant."""
    summaries = checkpoint_summaries(STUDIO_CHECKPOINTS)
    assert summaries == expected_checkpoint_summaries()
    assert set(INVARIANT_COVERAGE) == {f"I27.{index}" for index in range(1, 21)}
    state = fold_events(STUDIO_EVENTS)
    assert state.pending == ["D3@new-source-version"]
    assert state.history[-1]["content"] == {
        "verdict": "not_ready",
        "blocking_ids": ["D3"],
        "evidence_count": 11,
    }
    unresolved = unresolved_attempts(state)
    assert unresolved == []
    return _result(
        "CH27-D10",
        record_id="studio-execution-001",
        checkpoint="K5",
        checkpoint_count=len(summaries),
        checkpoint_summaries=summaries,
        invariant_coverage=INVARIANT_COVERAGE,
        unknown_outcomes=unresolved,
        pending=state.pending,
        evidence_count=11,
        blocking_ids=["D3"],
        verdict="not_ready",
        safe_next_actions=["await-new-source-version", "create-new-logical-call"],
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Run and return every chapter invariant check."""
    checks = {
        "event_graph": check_event_graph(),
        "fold_replay": check_fold_replay(),
        "reasoning_history": check_reasoning_history(),
        "binding_admission": check_binding_admission(),
        "call_attempt_lifecycle": check_call_attempt_lifecycle(),
        "effect_recovery": check_effect_recovery(),
        "result_projection": check_result_projection(),
        "parallel_schedule": check_parallel_schedule(),
        "invalid_transitions": check_invalid_transitions(),
        "handoff": check_handoff(),
    }
    assert len(checks) == 10
    return checks
