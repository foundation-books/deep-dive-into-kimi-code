"""Finite crash/recovery state transitions adapted from the original coding study."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass
class _State:
    actions: list[dict[str, Any]]
    effects: list[dict[str, Any]] = field(default_factory=list)
    receipts: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    version: int = 0
    recovery_ms: int = 0

    def apply(self, index: int, phase: str, receipt: bool) -> None:
        action = self.actions[index]
        attempt = (
            f"pre:{action['id']}"
            if phase == "pre"
            else f"resume:{len(self.effects)}:{action['id']}"
        )
        self.effects.append(
            {
                "attempt_id": attempt,
                "action_id": action["id"],
                "effect_class": action["effect_class"],
                "version_before": self.version,
                "expected_version": action["expected_version"],
                "duplicate": any(
                    row["action_id"] == action["id"] for row in self.effects
                ),
                "stale": self.version != action["expected_version"],
            }
        )
        self.version += 1
        if receipt:
            self.receipts.append(attempt)

    def event(self, kind: str, cost_ms: int, action_id: str | None = None) -> None:
        self.events.append({"kind": kind, "action_id": action_id, "cost_ms": cost_ms})
        self.recovery_ms += cost_ms


def dispatch_jitter(task_id: str, action_id: str, repetition: int) -> int:
    """Return the original paired deterministic dispatch-cost jitter."""
    data = f"EXP29.04|v1|{task_id}|{action_id}|{repetition}".encode("utf-8")
    return sha256(data).digest()[0] % 5


def _prepare(
    task: dict[str, Any], target: int, window: str
) -> tuple[_State, dict[str, Any], bool]:
    state = _State(task["actions"])
    for index in range(target):
        state.apply(index, "pre", True)
    applied = window != "before_effect"
    receipted = window in ("receipt_before_checkpoint", "checkpoint_after_receipt")
    if applied:
        state.apply(target, "pre", receipted)
    pre = {
        "checkpoint_next": target + int(window == "checkpoint_after_receipt"),
        "transcript_entries": target + 1,
        "version": state.version,
        "effects": [row.copy() for row in state.effects],
        "receipts": state.receipts.copy(),
        "journal_available": window != "effect_before_receipt_unobservable",
        "terminal_receipt_present": receipted,
    }
    return state, pre, applied


def _resume_index(
    state: _State, target: int, policy: str, pre: dict[str, Any], applied: bool
) -> tuple[int, bool]:
    if policy == "transcript_resume":
        state.event("scan_transcript", 2 * (target + 1))
        return target + int(pre["terminal_receipt_present"]), False
    state.event("load_checkpoint", 5)
    start = pre["checkpoint_next"]
    if policy == "checkpoint_no_reconciliation":
        return start, False
    if policy != "validated_checkpoint_reconciliation":
        raise ValueError(f"unknown policy: {policy}")
    state.event("validate_checkpoint", 3)
    if start != target:
        return start, False
    action_id = state.actions[target]["id"]
    state.event("lookup_receipt", 2, action_id)
    if pre["terminal_receipt_present"]:
        return start + 1, False
    state.event("query_effect_journal", 7, action_id)
    if not pre["journal_available"]:
        state.event("hold_unknown", 1, action_id)
        return start, True
    if applied:
        state.receipts.append(f"pre:{action_id}")
        state.event("repair_receipt", 4, action_id)
        start += 1
    return start, False


def _resume_effects(
    state: _State, task_id: str, start: int, policy: str, repetition: int
) -> bool:
    for index in range(start, len(state.actions)):
        action = state.actions[index]
        if (
            policy == "validated_checkpoint_reconciliation"
            and state.version != action["expected_version"]
        ):
            state.event("hold_stale_state", 1, action["id"])
            return True
        state.event(
            "dispatch_effect",
            11 + dispatch_jitter(task_id, action["id"], repetition),
            action["id"],
        )
        state.apply(index, "resume", False)
        state.receipts.append(state.effects[-1]["attempt_id"])
        state.event("commit_receipt", 3, action["id"])
    return False


def run_cell(
    task: dict[str, Any], target: int, window: str, policy: str, repetition: int
) -> dict[str, Any]:
    """Simulate one crash cut; no repository write or external post occurs."""
    state, pre, applied = _prepare(task, target, window)
    start, held = _resume_index(state, target, policy, pre, applied)
    if not held:
        held = _resume_effects(state, task["id"], start, policy, repetition)
    duplicates = sum(row["duplicate"] for row in state.effects)
    stale = sum(row["stale"] for row in state.effects)
    lost = sum(row["attempt_id"] not in state.receipts for row in state.effects)
    complete = all(
        any(row["action_id"] == action["id"] for row in state.effects)
        for action in state.actions
    )
    safe = (
        complete
        and not held
        and state.version == 4
        and duplicates == stale == lost == 0
    )
    status = "unresolved" if held else "safe_complete" if safe else "unsafe_complete"
    return {
        "task_id": task["id"],
        "repository": task["repository"],
        "target_index": target,
        "target_action_id": state.actions[target]["id"],
        "window": window,
        "policy": policy,
        "repetition": repetition,
        "pre_crash": pre,
        "resume_events": state.events,
        "final_state": {
            "version": state.version,
            "effects": state.effects,
            "receipts": state.receipts,
            "task_status": status,
        },
        "metrics": {
            "duplicate_effects": duplicates,
            "lost_evidence": lost,
            "stale_state_admissions": stale,
            "unresolved_outcomes": int(window == "effect_before_receipt_unobservable"),
            "recovery_time_ms": state.recovery_ms,
            "safe_complete": int(status == "safe_complete"),
            "unsafe_complete": int(status == "unsafe_complete"),
            "unresolved_task": int(status == "unresolved"),
        },
    }
