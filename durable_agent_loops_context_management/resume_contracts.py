"""Deterministic resume, effect, and checkpoint cost contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def drift_set(
    recorded_versions: Mapping[str, str], live_versions: Mapping[str, str]
) -> list[str]:
    """Return the sorted version components that changed before resume."""

    return sorted(
        key
        for key in set(recorded_versions) | set(live_versions)
        if recorded_versions.get(key) != live_versions.get(key)
    )


def _resume_result(
    status: str,
    gate: str | None,
    drift: list[str],
    actions: list[str],
    **extra: object,
) -> dict[str, object]:
    return {
        "status": status,
        "first_failed_gate": gate,
        "drift": drift,
        "actions": actions,
        **extra,
    }


def resume_admission(
    checkpoint: dict[str, object],
    state: Mapping[str, object],
    required_refs: set[str],
    recorded_versions: Mapping[str, str],
    live_versions: Mapping[str, str],
    open_effects: Iterable[str],
) -> dict[str, object]:
    """Apply the ordered pre-work gates for a resumed durable session."""

    from .reference import checkpoint_valid

    if not checkpoint_valid(checkpoint, state, required_refs):
        return _resume_result(
            "blocked", "checkpoint_integrity_or_closure", [], ["pause"]
        )
    unresolved_effects = sorted(set(open_effects))
    if unresolved_effects:
        return _resume_result(
            "recovery_required",
            "open_effect_reconciliation",
            [],
            ["reconcile"],
            open_effects=unresolved_effects,
        )
    changed = drift_set(recorded_versions, live_versions)
    if changed:
        actions = sorted(
            {
                "request-authority"
                if component == "authority"
                else "refresh"
                if component == "source"
                else "replan"
                if component in {"goal", "policy"}
                else "revalidate"
                for component in changed
            }
        )
        return _resume_result("recovery_required", "version_drift", changed, actions)
    return _resume_result("ready", None, [], ["continue"])


def reconcile_open_effect(effect_class: str, observation: str) -> dict[str, object]:
    """Classify an interrupted effect without guessing an unknown outcome."""

    effect_classes = {
        "pure",
        "idempotent",
        "deduplicated",
        "compensatable",
        "nonrepeatable",
    }
    observations = {"unknown", "no_effect", "completed", "failed"}
    if effect_class not in effect_classes:
        raise ValueError(f"unknown effect class: {effect_class}")
    if observation not in observations:
        raise ValueError(f"unknown effect observation: {observation}")
    if observation == "unknown":
        return {
            "status": "unresolved",
            "action": "reconcile",
            "may_retry": False,
            "reason": "unknown outcome cannot authorize a duplicate-risk retry",
        }
    return {
        "status": "reconciled",
        "action": "reuse-completed-effect" if observation == "completed" else "retry",
        "may_retry": observation in {"no_effect", "failed"},
    }


def checkpoint_cost_frontier(
    total_units: int,
    interval: int,
    interruption_unit: int,
    write_cost_units: int = 0,
) -> dict[str, int | list[int]]:
    """Account for checkpoint count, explicit write cost, and lost work."""

    if total_units < 0 or interval <= 0:
        raise ValueError("total_units must be nonnegative and interval positive")
    if not 0 <= interruption_unit <= total_units:
        raise ValueError("interruption_unit must lie within the workload")
    if write_cost_units < 0:
        raise ValueError("write_cost_units must be nonnegative")
    checkpoints = list(range(interval, total_units + 1, interval))
    durable = max(
        (unit for unit in checkpoints if unit <= interruption_unit), default=0
    )
    return {
        "checkpoint_units": checkpoints,
        "checkpoint_count": len(checkpoints),
        "write_overhead_units": len(checkpoints) * write_cost_units,
        "last_durable_unit": durable,
        "lost_work_units": interruption_unit - durable,
    }
