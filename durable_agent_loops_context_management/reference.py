"""Finite deterministic oracles for Chapter 29 durable-loop semantics."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from itertools import combinations, pairwise
import json

from .resume_contracts import (
    checkpoint_cost_frontier,  # noqa: F401
    drift_set,  # noqa: F401
    reconcile_open_effect,  # noqa: F401
    resume_admission,  # noqa: F401
)


FIXTURE_VERSION = "ch28-reference-v1"


def is_acyclic(nodes: set[str], edges: set[tuple[str, str]]) -> bool:
    """Return whether the directed plan graph is acyclic."""
    incoming = {node: {u for u, v in edges if v == node} for node in nodes}
    ready = [node for node in nodes if not incoming[node]]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for source, target in edges:
            if source != node or node not in incoming[target]:
                continue
            incoming[target].remove(node)
            if not incoming[target]:
                ready.append(target)
    return visited == len(nodes)


def plan_frontier(
    nodes: set[str],
    edges: set[tuple[str, str]],
    complete: set[str],
    preconditions: dict[str, bool],
) -> list[str]:
    """Return incomplete nodes whose dependencies and preconditions are satisfied."""
    incoming = {node: {u for u, v in edges if v == node} for node in nodes}
    return sorted(
        node
        for node in nodes - complete
        if incoming[node] <= complete and preconditions.get(node, False)
    )


def admissible_actions(*sets: Iterable[str]) -> list[str]:
    """Return the sorted intersection of action constraints."""
    materialized = [set(values) for values in sets]
    return sorted(set.intersection(*materialized)) if materialized else []


def select_context(
    items: dict[str, tuple[int, int]],
    capacity: int,
    mandatory: set[str],
    dependencies: dict[str, set[str]],
) -> tuple[list[str], int, int]:
    """Choose highest-value dependency-closed items within the capacity."""
    names = sorted(items)
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for size in range(len(names) + 1):
        for choice in combinations(names, size):
            selected = set(choice)
            if not mandatory <= selected:
                continue
            if any(
                not requirements <= selected
                for item, requirements in dependencies.items()
                if item in selected
            ):
                continue
            cost = sum(items[item][0] for item in selected)
            if cost <= capacity:
                score = sum(items[item][1] for item in selected)
                candidates.append((score, -cost, choice))
    score, negative_cost, choice = max(candidates)
    return list(choice), -negative_cost, score


def provenance_closure(parents: dict[str, set[str]], root: str) -> list[str]:
    """Return the root and all recursively reachable provenance parents."""
    seen: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(sorted(parents.get(node, set()) - seen))
    return sorted(seen)


def state_digest(state: Mapping[str, object]) -> str:
    """Return a deterministic digest of canonical state."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_valid(
    checkpoint: dict[str, object],
    state: Mapping[str, object],
    required_refs: set[str],
) -> bool:
    """Return whether a checkpoint matches state, open items, and required closure."""
    required_fields = {
        "id",
        "boundary",
        "state_digest",
        "versions",
        "open_items",
        "closure",
    }
    closure = checkpoint.get("closure")
    return (
        required_fields <= set(checkpoint)
        and checkpoint.get("state_digest") == state_digest(state)
        and checkpoint.get("open_items") == state.get("open_items")
        and isinstance(closure, list)
        and required_refs <= set(closure)
    )


def _result(passed: bool, **payload: object) -> dict[str, object]:
    status = "pass" if passed else "fail"
    return {"status": status, "fixture_version": FIXTURE_VERSION, **payload}


def _planning_checks() -> dict[str, dict[str, object]]:
    nodes = {"acquire", "validate", "judge", "publish", "block"}
    edges = {
        ("acquire", "validate"),
        ("validate", "judge"),
        ("judge", "publish"),
        ("judge", "block"),
    }
    initial_frontier = plan_frontier(
        nodes, edges, set(), {name: name == "acquire" for name in nodes}
    )
    after_acquire = plan_frontier(
        nodes, edges, {"acquire"}, {name: name == "validate" for name in nodes}
    )

    action_set = admissible_actions(
        {"request", "wait", "publish"},
        {"request", "wait"},
        {"request", "publish"},
        {"wait", "publish"},
    )
    feasible_action = admissible_actions(
        {"request", "wait", "publish"},
        {"request", "wait"},
        {"request", "wait"},
        {"request", "wait"},
    )
    acyclic = is_acyclic(nodes, edges)
    return {
        "plan_frontier": _result(
            acyclic
            and initial_frontier == ["acquire"]
            and after_acquire == ["validate"],
            acyclic=acyclic,
            initial=initial_frontier,
            after_acquire=after_acquire,
        ),
        "admissible_actions": _result(
            not action_set and feasible_action == ["request", "wait"],
            empty_intersection=action_set,
            feasible=feasible_action,
        ),
    }


def _progress_and_context_checks() -> dict[str, dict[str, object]]:
    certificates = [
        {"acceptance": 0, "unknowns": 0, "frontier": 0, "tool_cost": 1},
        {"acceptance": 0, "unknowns": -1, "frontier": 1, "tool_cost": 1},
    ]
    progress = [
        row["acceptance"] > 0 or row["unknowns"] < 0 or row["frontier"] > 0
        for row in certificates
    ]

    budgets = [(8000, 30, 4, 100, 2), (7400, 28, 3, 96, 2), (6800, 26, 2, 92, 2)]
    monotone = all(
        all(b <= a for a, b in zip(left, right, strict=True))
        for left, right in pairwise(budgets)
    )
    reserve_feasible = 2 - 1 >= 1

    items = {"g": (3, 5), "d": (4, 4), "h": (6, 9), "n": (5, 7)}
    selected, cost, score = select_context(items, 12, {"g", "d"}, {})
    dependency_items = {"g": (3, 5), "d": (4, 4), "p": (2, 10), "x": (2, 1)}
    dependency_selected, _, _ = select_context(
        dependency_items, 11, {"g", "d"}, {"p": {"x"}}
    )
    dependency_closed = "p" not in dependency_selected or "x" in dependency_selected

    before = {
        "goal_v3": True,
        "d3_unknown": True,
        "publish_admissible": False,
        "authority_live": False,
        "verification_reserve": True,
    }
    good_summary = dict(before)
    bad_summary = {**before, "d3_unknown": False, "publish_admissible": True}
    protected = sorted(before)
    good_retention = all(before[key] == good_summary[key] for key in protected)
    bad_violations = sorted(key for key in protected if before[key] != bad_summary[key])
    return {
        "progress": _result(progress == [False, True], certified=progress),
        "budgets": _result(
            monotone and reserve_feasible,
            monotone=monotone,
            reserve_feasible=reserve_feasible,
            states=budgets,
        ),
        "context_selection": _result(
            selected == ["d", "g", "n"]
            and cost == 12
            and score == 16
            and dependency_closed,
            selected=selected,
            cost=cost,
            score=score,
            dependency_selected=dependency_selected,
            dependency_closed=dependency_closed,
        ),
        "compaction": _result(
            good_retention and bad_violations == ["d3_unknown", "publish_admissible"],
            good_retention=good_retention,
            bad_violations=bad_violations,
        ),
    }


def _terminal_certificate_is_complete(versions: Mapping[str, str]) -> bool:
    certificate = {
        "mission": "studio-session-001",
        "goal": "release-contract@3",
        "status": "success",
        "evidence_atoms": 12,
        "unknowns": [],
        "budget": (6800, 26, 2, 92, 2),
        "versions": versions,
        "state_digest": "digest-y7",
    }
    terminal_required = {
        "mission",
        "goal",
        "status",
        "evidence_atoms",
        "unknowns",
        "budget",
        "versions",
        "state_digest",
    }
    return terminal_required <= set(certificate)


def _drift_fixture() -> tuple[list[str], dict[str, str], dict[str, str]]:
    old = {
        "source": "v7",
        "repository": "r1",
        "tool": "t1",
        "authority": "lease-1",
        "policy": "p1",
    }
    live = {
        "source": "v8",
        "repository": "r2",
        "tool": "t1",
        "authority": "lease-2",
        "policy": "p1",
    }
    drift = sorted(key for key in old if old[key] != live[key])
    dispositions = {
        "source": "refresh",
        "repository": "revalidate",
        "authority": "request-authority",
    }
    return drift, dispositions, live


def _persistence_checks() -> dict[str, dict[str, object]]:
    parents = {
        "summary": {"goal-v3", "manifest-v7", "raw-d3"},
        "manifest-v7": {"raw-d1", "raw-d2"},
    }
    closure = provenance_closure(parents, "summary")

    checkpoint_state = {
        "goal": "release-contract@3",
        "plan": "plan@2",
        "open_items": ["D3", "renew-authority"],
    }
    checkpoint = {
        "id": "checkpoint-l5",
        "boundary": 5,
        "state_digest": state_digest(checkpoint_state),
        "versions": {"goal": "v3", "source": "v7", "authority": "lease-1"},
        "open_items": ["D3", "renew-authority"],
        "closure": ["goal", "plan", "open_items"],
    }
    checkpoint_complete = checkpoint_valid(
        checkpoint, checkpoint_state, {"goal", "plan", "open_items"}
    )
    drift, dispositions, live = _drift_fixture()

    state_trace = ["A", "B", "A", "B", "A", "B"]
    oscillation = len(state_trace) >= 4 and state_trace[-4:] == ["A", "B", "A", "B"]
    certificate_complete = _terminal_certificate_is_complete(live)

    return {
        "provenance": _result(
            closure
            == ["goal-v3", "manifest-v7", "raw-d1", "raw-d2", "raw-d3", "summary"],
            closure=closure,
        ),
        "checkpoint": _result(
            checkpoint_complete,
            complete=checkpoint_complete,
            open_items=checkpoint["open_items"],
        ),
        "drift": _result(
            drift == ["authority", "repository", "source"]
            and set(drift) == set(dispositions),
            changed=drift,
            dispositions=dispositions,
        ),
        "terminal": _result(
            oscillation and certificate_complete,
            oscillation_detected=oscillation,
            certificate_complete=certificate_complete,
        ),
    }


def all_checks() -> dict[str, dict[str, object]]:
    """Run and return every chapter invariant check."""
    return {
        **_planning_checks(),
        **_progress_and_context_checks(),
        **_persistence_checks(),
    }
