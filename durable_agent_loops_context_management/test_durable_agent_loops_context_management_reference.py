from durable_agent_loops_context_management.reference import (
    all_checks,
    checkpoint_cost_frontier,
    checkpoint_valid,
    drift_set,
    is_acyclic,
    reconcile_open_effect,
    resume_admission,
    select_context,
    state_digest,
)


def test_all_checks_pass() -> None:
    results = all_checks()
    assert len(results) == 10
    assert all(result["status"] == "pass" for result in results.values())


def test_context_selection_respects_mandatory_and_dependency() -> None:
    chosen, cost, score = select_context(
        {"g": (3, 5), "d": (4, 4), "h": (6, 9), "n": (5, 7)},
        12,
        {"g", "d"},
        {},
    )
    assert chosen == ["d", "g", "n"]
    assert (cost, score) == (12, 16)


def test_plan_cycle_and_checkpoint_tamper_are_rejected() -> None:
    assert is_acyclic({"a", "b"}, {("a", "b")})
    assert not is_acyclic({"a", "b"}, {("a", "b"), ("b", "a")})

    state = {"goal": "v3", "open_items": ["D3"]}
    checkpoint = {
        "id": "q1",
        "boundary": 1,
        "state_digest": state_digest(state),
        "versions": {"goal": "v3"},
        "open_items": ["D3"],
        "closure": ["goal", "open_items"],
    }
    assert checkpoint_valid(checkpoint, state, {"goal", "open_items"})
    assert not checkpoint_valid(checkpoint, {**state, "goal": "v4"}, {"goal"})


def test_resume_requires_effect_reconciliation_before_version_repair() -> None:
    state = {"goal": "v3", "open_items": ["D3"]}
    checkpoint = {
        "id": "q1",
        "boundary": 1,
        "state_digest": state_digest(state),
        "versions": {"goal": "v3", "source": "v7"},
        "open_items": ["D3"],
        "closure": ["goal", "open_items"],
    }
    assert drift_set(
        {"source": "v7", "authority": "lease-1"},
        {"source": "v8", "authority": "lease-2"},
    ) == ["authority", "source"]
    result = resume_admission(
        checkpoint,
        state,
        {"goal", "open_items"},
        {"source": "v7", "authority": "lease-1"},
        {"source": "v8", "authority": "lease-2"},
        {"call-17"},
    )
    assert result["first_failed_gate"] == "open_effect_reconciliation"
    assert result["actions"] == ["reconcile"]


def test_resume_drift_and_effect_outcomes_are_explicit() -> None:
    state = {"goal": "v3", "open_items": []}
    checkpoint = {
        "id": "q2",
        "boundary": 2,
        "state_digest": state_digest(state),
        "versions": {"goal": "v3"},
        "open_items": [],
        "closure": ["goal", "open_items"],
    }
    result = resume_admission(
        checkpoint,
        state,
        {"goal", "open_items"},
        {"source": "v7", "authority": "lease-1"},
        {"source": "v8", "authority": "lease-2"},
        set(),
    )
    assert result["status"] == "recovery_required"
    assert result["drift"] == ["authority", "source"]
    assert result["actions"] == ["refresh", "request-authority"]
    assert reconcile_open_effect("nonrepeatable", "unknown")["may_retry"] is False
    assert reconcile_open_effect("idempotent", "no_effect")["action"] == "retry"


def test_checkpoint_cost_frontier_keeps_write_and_lost_work_separate() -> None:
    result = checkpoint_cost_frontier(10, 4, 9, write_cost_units=2)
    assert result == {
        "checkpoint_units": [4, 8],
        "checkpoint_count": 2,
        "write_overhead_units": 4,
        "last_durable_unit": 8,
        "lost_work_units": 1,
    }
