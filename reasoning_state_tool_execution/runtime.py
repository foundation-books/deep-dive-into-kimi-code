"""Event validation, state folding, and replay for the Chapter 28 fixture."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable, Sequence

from .model import (
    Attempt,
    CanonicalCall,
    Event,
    ExecutionState,
    Observation,
    RegistryEntry,
    digest,
)


class ExecutionContractError(ValueError):
    """Raised when a candidate event violates the finite teaching contract."""


def topological_order(events: Iterable[Event]) -> tuple[Event, ...]:
    """Return one deterministic topological order, breaking ties by event ID."""

    event_list = tuple(events)
    by_id = _unique_events(event_list)
    _require_known_parents(by_id)
    indegree = {event_id: len(event.parents) for event_id, event in by_id.items()}
    children: dict[str, list[str]] = {event_id: [] for event_id in by_id}
    for event in event_list:
        for parent in event.parents:
            children[parent].append(event.event_id)
    ready = sorted(event_id for event_id, degree in indegree.items() if degree == 0)
    ordered: list[Event] = []
    while ready:
        event_id = ready.pop(0)
        ordered.append(by_id[event_id])
        for child in sorted(children[event_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(event_list):
        raise ExecutionContractError("event graph contains a cycle")
    return tuple(ordered)


def all_topological_orders(
    events: Iterable[Event],
) -> tuple[tuple[Event, ...], ...]:
    """Enumerate every linear extension of a small finite event DAG."""

    event_list = tuple(events)
    by_id = _unique_events(event_list)
    _require_known_parents(by_id)
    event_ids = tuple(sorted(by_id))
    orders: list[tuple[Event, ...]] = []

    def visit(seen: frozenset[str], prefix: tuple[Event, ...]) -> None:
        if len(prefix) == len(event_ids):
            orders.append(prefix)
            return
        ready = [
            event_id
            for event_id in event_ids
            if event_id not in seen and set(by_id[event_id].parents) <= seen
        ]
        if not ready:
            raise ExecutionContractError("event graph contains a cycle")
        for event_id in ready:
            visit(seen | {event_id}, prefix + (by_id[event_id],))

    visit(frozenset(), ())
    return tuple(orders)


def validate_order(events: Sequence[Event]) -> None:
    """Validate an explicitly supplied linear extension."""

    by_id = _unique_events(events)
    _require_known_parents(by_id)
    seen: set[str] = set()
    for event in events:
        missing = set(event.parents) - seen
        if missing:
            raise ExecutionContractError(
                f"event {event.event_id} precedes parents {sorted(missing)}"
            )
        seen.add(event.event_id)
    topological_order(events)


def fold_events(
    events: Iterable[Event],
    *,
    ordered: bool = False,
    fold_version: str = "ch27-fold-v1",
) -> ExecutionState:
    """Fold accepted events without invoking a model, tool, or external callback."""

    if fold_version != "ch27-fold-v1":
        raise ExecutionContractError(f"unsupported fold version: {fold_version}")
    sequence = tuple(events)
    if ordered:
        validate_order(sequence)
    else:
        sequence = topological_order(sequence)
    state = ExecutionState()
    for event in sequence:
        _apply_event(state, event)
    return state


EventHandler = Callable[[ExecutionState, dict[str, Any]], None]


def _append_history(state: ExecutionState, payload: dict[str, Any]) -> None:
    state.history.append(payload["message"])


def _freeze_registry(state: ExecutionState, payload: dict[str, Any]) -> None:
    if state.registry:
        raise ExecutionContractError("registry snapshot already frozen")
    entries = {row["name"]: RegistryEntry(**row) for row in payload["entries"]}
    for entry in entries.values():
        if (
            not entry.schema_digest
            or not entry.adapter_version
            or not entry.capabilities
        ):
            raise ExecutionContractError("registry entry is incomplete")
    actual = digest({key: entries[key].__dict__ for key in sorted(entries)})
    if actual != payload["registry_digest"]:
        raise ExecutionContractError("registry digest mismatch")
    state.registry = entries
    state.registry_digest = actual
    state.environment = payload["environment"]


def _create_call(state: ExecutionState, payload: dict[str, Any]) -> None:
    call = CanonicalCall(**payload["call"])
    if call.call_id in state.calls:
        raise ExecutionContractError(f"duplicate call ID: {call.call_id}")
    if call.registry_digest != state.registry_digest:
        raise ExecutionContractError("call uses unknown registry snapshot")
    if call.environment_digest != digest(state.environment):
        raise ExecutionContractError("call environment digest mismatch")
    if call.name not in state.registry:
        raise ExecutionContractError(f"unbound tool: {call.name}")
    if call.argument_digest != digest(call.arguments):
        raise ExecutionContractError("canonical argument digest mismatch")
    entry = state.registry[call.name]
    if call.interface_version != entry.interface_version:
        raise ExecutionContractError("interface version mismatch")
    if call.effect_key is not None and not call.effect_key:
        raise ExecutionContractError("effect key must be nonempty when present")
    state.calls[call.call_id] = call


def _admit_attempt(state: ExecutionState, payload: dict[str, Any]) -> None:
    attempt_id = payload["attempt_id"]
    call_id = payload["call_id"]
    if attempt_id in state.attempts:
        raise ExecutionContractError(f"duplicate attempt ID: {attempt_id}")
    if call_id not in state.calls:
        raise ExecutionContractError("attempt admission references unknown call")
    checks = payload["checks"]
    if not checks or not all(checks.values()):
        raise ExecutionContractError("attempt admission has failed precondition")
    call_attempts = [a for a in state.attempts.values() if a.call_id == call_id]
    number = payload["number"]
    if number != len(call_attempts) + 1:
        raise ExecutionContractError("attempt number is not monotone")
    state.attempts[attempt_id] = Attempt(
        attempt_id=attempt_id,
        call_id=call_id,
        number=number,
        executor=payload["executor"],
        status="admitted",
    )


def _dispatch_attempt(state: ExecutionState, payload: dict[str, Any]) -> None:
    attempt = _attempt(state, payload["attempt_id"])
    if attempt.status != "admitted":
        raise ExecutionContractError("dispatch requires admitted attempt")
    attempt.status = "dispatched"


def _record_receipt(state: ExecutionState, payload: dict[str, Any]) -> None:
    attempt = _attempt(state, payload["attempt_id"])
    if attempt.status != "dispatched":
        raise ExecutionContractError("receipt requires dispatched attempt")
    if attempt.receipt is not None:
        raise ExecutionContractError("attempt already has receipt")
    attempt.receipt = payload["receipt"]


def _terminate_attempt(state: ExecutionState, payload: dict[str, Any]) -> None:
    attempt = _attempt(state, payload["attempt_id"])
    if attempt.status != "dispatched":
        raise ExecutionContractError("terminal transition requires dispatched attempt")
    status = payload["status"]
    if status not in {"succeeded", "failed", "cancelled", "timed_out", "unknown"}:
        raise ExecutionContractError(f"unsupported terminal status: {status}")
    attempt.status = status


def _mark_unknown(state: ExecutionState, payload: dict[str, Any]) -> None:
    attempt = _attempt(state, payload["attempt_id"])
    if attempt.status != "dispatched":
        raise ExecutionContractError("unknown recovery requires dispatched attempt")
    attempt.status = "unknown"
    state.diagnostics.append(payload["diagnostic"])


def _record_observation(state: ExecutionState, payload: dict[str, Any]) -> None:
    observation = Observation(**payload["observation"])
    if observation.observation_id in state.observations:
        raise ExecutionContractError("duplicate observation ID")
    attempt = _attempt(state, observation.attempt_id)
    if observation.call_id != attempt.call_id:
        raise ExecutionContractError("observation call/attempt mismatch")
    call = state.calls[attempt.call_id]
    entry = state.registry[call.name]
    if observation.executable != attempt.executor:
        raise ExecutionContractError("observation attempt executor mismatch")
    if observation.executable != entry.executable:
        raise ExecutionContractError("observation executable mismatch")
    if observation.environment_digest != call.environment_digest:
        raise ExecutionContractError("observation environment mismatch")
    if observation.executor_status != observation.raw.get("executor_status"):
        raise ExecutionContractError("observation executor status mismatch")
    if observation.domain_status != observation.raw.get("domain_status"):
        raise ExecutionContractError("observation domain status mismatch")
    if observation.raw_digest != digest(observation.raw):
        raise ExecutionContractError("raw payload digest mismatch")
    if observation.projection_digest != digest(observation.projection):
        raise ExecutionContractError("projection digest mismatch")
    if observation.recovered:
        if attempt.status != "unknown" or attempt.receipt is None:
            raise ExecutionContractError(
                "recovered observation requires unknown attempt with receipt"
            )
    elif attempt.status != "succeeded":
        raise ExecutionContractError("ordinary observation requires succeeded attempt")
    state.observations[observation.observation_id] = observation


def _record_handoff(state: ExecutionState, payload: dict[str, Any]) -> None:
    unresolved = [
        attempt.attempt_id
        for attempt in state.attempts.values()
        if attempt.status in {"dispatched", "unknown"}
        and not any(
            observation.attempt_id == attempt.attempt_id
            for observation in state.observations.values()
        )
    ]
    if unresolved != payload["unknown_outcomes"]:
        raise ExecutionContractError("handoff unknown-outcome mismatch")
    state.pending = list(payload["pending"])
    state.history.append(payload["message"])


EVENT_HANDLERS: dict[str, EventHandler] = {
    "history.append": _append_history,
    "registry.freeze": _freeze_registry,
    "call.create": _create_call,
    "attempt.admit": _admit_attempt,
    "attempt.dispatch": _dispatch_attempt,
    "effect.receipt": _record_receipt,
    "attempt.terminal": _terminate_attempt,
    "recovery.unknown": _mark_unknown,
    "observation.record": _record_observation,
    "handoff.record": _record_handoff,
}


def _apply_event(state: ExecutionState, event: Event) -> None:
    try:
        handler = EVENT_HANDLERS[event.kind]
    except KeyError as error:
        raise ExecutionContractError(f"unsupported event kind: {event.kind}") from error
    handler(state, event.payload)


def _unique_events(events: Iterable[Event]) -> dict[str, Event]:
    by_id: dict[str, Event] = {}
    for event in events:
        if event.observed_at < 0:
            raise ExecutionContractError("event observed_at must be nonnegative")
        if not event.subject or not event.source:
            raise ExecutionContractError("event subject and source must be nonempty")
        if event.event_id in by_id:
            raise ExecutionContractError(f"duplicate event ID: {event.event_id}")
        by_id[event.event_id] = event
    return by_id


def _require_known_parents(by_id: dict[str, Event]) -> None:
    for event in by_id.values():
        missing = set(event.parents) - set(by_id)
        if missing:
            raise ExecutionContractError(
                f"event {event.event_id} has unknown parents {sorted(missing)}"
            )


def _attempt(state: ExecutionState, attempt_id: str) -> Attempt:
    try:
        return state.attempts[attempt_id]
    except KeyError as error:
        raise ExecutionContractError(f"unknown attempt: {attempt_id}") from error


def replace_event(event: Event, **changes: Any) -> Event:
    """Test helper that keeps corruptions explicit and non-mutating."""

    return replace(event, **changes)


def preserved_history_errors(state: ExecutionState) -> tuple[str, ...]:
    """Check the finite K3-shaped history and its call/result closure."""

    errors: list[str] = []
    if not state.history:
        return ("history-empty",)
    assistant = state.history[0]
    for field in ("role", "reasoning_content", "content", "tool_calls"):
        if field not in assistant:
            errors.append(f"assistant-missing:{field}")
    if assistant.get("role") != "assistant":
        errors.append("first-record-not-assistant")
    calls = {call.get("id") for call in assistant.get("tool_calls", [])}
    if len(state.history) > 1:
        results = {
            result.get("tool_call_id")
            for result in state.history[-1].get("tool_results", [])
        }
        if calls != results:
            errors.append("history-call-result-not-closed")
    return tuple(errors)
