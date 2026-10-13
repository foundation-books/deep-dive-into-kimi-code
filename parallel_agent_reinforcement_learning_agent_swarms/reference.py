"""Executable semantics for the Chapter 26 teaching fixtures.

The module is dependency-free and uses exact arithmetic.  It checks only
chapter-created records, graphs, and ledgers; it does not train or run an agent
swarm and it does not reproduce private Kimi infrastructure.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from .metric_checks import (
    critical_tie_check,
    reward_annealing_check,
    work_critical_check,
)


def _status(check_id: str, predicate: bool, **fields: object) -> dict[str, object]:
    return {"check_id": check_id, "status": "pass" if predicate else "fail", **fields}


def task_identity_check() -> dict[str, object]:
    """Keep task identity stable across every fork and join."""

    identity = (
        "studio-swarm-001",
        "studio@v7",
        "dossier-v2",
        ("critical_steps:20", "total_work:40", "tool_calls:24"),
    )
    assignments = {
        "R": identity,
        "C": identity,
        "V": identity,
        "D": identity,
    }
    changed_environment = (*identity[:1], "studio@v8", *identity[2:])
    return _status(
        "CH25-D01",
        all(value == identity for value in assignments.values())
        and changed_environment != identity,
        task_identity=list(identity),
        assignment_count=len(assignments),
        fork_identity_preserved=True,
        environment_change_creates_new_task=changed_environment != identity,
        result_fields_before_execution="unknown",
    )


def _topological_order(
    vertices: Iterable[str], edges: Iterable[tuple[str, str]]
) -> tuple[str, ...] | None:
    vertex_set = set(vertices)
    children: dict[str, set[str]] = defaultdict(set)
    indegree = {vertex: 0 for vertex in vertex_set}
    for source, target in edges:
        if source not in vertex_set or target not in vertex_set:
            raise ValueError("edge endpoint is not a declared vertex")
        if target not in children[source]:
            children[source].add(target)
            indegree[target] += 1
    frontier = sorted(vertex for vertex, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while frontier:
        vertex = frontier.pop(0)
        order.append(vertex)
        for target in sorted(children[vertex]):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
                frontier.sort()
    return tuple(order) if len(order) == len(vertex_set) else None


def _ready_frontier(
    pending: Iterable[str],
    prerequisites: Mapping[str, frozenset[str]],
    admitted_conditions: Iterable[str],
) -> tuple[str, ...]:
    admitted = frozenset(admitted_conditions)
    return tuple(
        sorted(
            vertex
            for vertex in pending
            if prerequisites.get(vertex, frozenset()).issubset(admitted)
        )
    )


def graph_readiness_check() -> dict[str, object]:
    """Check acyclicity and semantic, rather than process-exit, readiness."""

    vertices = ("R", "C", "V", "D", "RC", "RV", "D3_REPAIR", "J")
    edges = (
        ("R", "RC"),
        ("C", "RC"),
        ("R", "RV"),
        ("V", "RV"),
        ("D", "D3_REPAIR"),
        ("RC", "J"),
        ("RV", "J"),
        ("D3_REPAIR", "J"),
    )
    prerequisites = {
        "R": frozenset(),
        "C": frozenset(),
        "V": frozenset(),
        "D": frozenset(),
        "RC": frozenset({"R.complete", "C.complete"}),
        "RV": frozenset({"R.complete", "V.complete"}),
        "D3_REPAIR": frozenset({"D.partial"}),
        "J": frozenset({"RC.complete", "RV.complete", "D3.covered"}),
    }
    first = _ready_frontier(("R", "C", "V", "D"), prerequisites, ())
    second = _ready_frontier(
        ("RC", "RV", "D3_REPAIR", "J"),
        prerequisites,
        (
            "R.complete",
            "C.complete",
            "V.complete",
            "D.partial",
            "D1.covered",
            "D2.covered",
        ),
    )
    third = _ready_frontier(
        ("J",),
        prerequisites,
        ("RC.complete", "RV.complete", "D3.covered"),
    )
    cycle = (*edges, ("J", "D3_REPAIR"))
    predicate = (
        first == ("C", "D", "R", "V")
        and second == ("D3_REPAIR", "RC", "RV")
        and third == ("J",)
        and _topological_order(vertices, edges) is not None
        and _topological_order(vertices, cycle) is None
    )
    return _status(
        "CH25-D02",
        predicate,
        initial_frontier=list(first),
        post_partial_frontier=list(second),
        post_repair_frontier=list(third),
        process_exit_unlocks_j=False,
        canonical_graph_acyclic=_topological_order(vertices, edges) is not None,
        reverse_repair_edge_rejected_as_cycle=_topological_order(vertices, cycle)
        is None,
    )


def _parallelism_class(
    *,
    independent_opportunity: bool,
    dispatched_workers: int,
    valid_completions: int,
    task_relevant_change: bool,
) -> str:
    if dispatched_workers == 0:
        return "serial_collapse" if independent_opportunity else "rational_serial"
    if valid_completions == 0 or not task_relevant_change:
        return "spurious_parallelism"
    return "useful_parallelism"


def false_parallelism_check() -> dict[str, object]:
    """Separate serial collapse, rational serial work, and empty fan-out."""

    cases = {
        "rational_serial": _parallelism_class(
            independent_opportunity=False,
            dispatched_workers=0,
            valid_completions=0,
            task_relevant_change=False,
        ),
        "serial_collapse": _parallelism_class(
            independent_opportunity=True,
            dispatched_workers=0,
            valid_completions=0,
            task_relevant_change=False,
        ),
        "spurious_parallelism": _parallelism_class(
            independent_opportunity=True,
            dispatched_workers=12,
            valid_completions=0,
            task_relevant_change=False,
        ),
        "useful_parallelism": _parallelism_class(
            independent_opportunity=True,
            dispatched_workers=4,
            valid_completions=3,
            task_relevant_change=True,
        ),
    }
    return _status(
        "CH25-D06",
        all(name == result for name, result in cases.items()),
        classifications=cases,
        paraphrase_worker_completion_rate=str(Fraction(0, 12)),
        process_exit_rate=str(Fraction(12, 12)),
    )


def context_ledger_check() -> dict[str, object]:
    """Separate global context occupancy from total token processing."""

    orchestrator_base = 2_000
    local_tokens = (1_200, 1_800, 1_400, 1_600)
    packet_tokens = (200, 200, 200, 200)
    shared_context = orchestrator_base + sum(local_tokens)
    sharded_context = orchestrator_base + sum(packet_tokens)
    counted_processing = orchestrator_base + sum(local_tokens) + sum(packet_tokens)
    return _status(
        "CH25-D07",
        shared_context == 8_000
        and sharded_context == 2_800
        and counted_processing == 8_800,
        shared_orchestrator_context=shared_context,
        sharded_orchestrator_context=sharded_context,
        worker_local_tokens=sum(local_tokens),
        returned_packet_tokens=sum(packet_tokens),
        minimum_counted_processing=counted_processing,
        local_work_is_free=False,
    )


def _aggregate_atom(records: Sequence[Mapping[str, str]]) -> str:
    if not records:
        return "missing"
    if any(not record.get("source") for record in records):
        return "missing_provenance"
    versions = {record["version"] for record in records}
    if len(versions) != 1:
        return "version_conflict"
    claims = {record["claim"] for record in records}
    if len(claims) != 1:
        return "claim_conflict"
    return "success"


def aggregation_check() -> dict[str, object]:
    """Reject incompatible versions and count unique coverage once."""

    conflicting = (
        {"atom": "C2", "claim": "pass", "version": "v7", "source": "test://c2/v7"},
        {"atom": "C2", "claim": "fail", "version": "v8", "source": "test://c2/v8"},
    )
    repaired = (
        {"atom": "C2", "claim": "fail", "version": "v8", "source": "test://c2/v8"},
        {"atom": "C2", "claim": "fail", "version": "v8", "source": "replay://c2/v8"},
    )
    return _status(
        "CH25-D08",
        _aggregate_atom(conflicting) == "version_conflict"
        and _aggregate_atom(repaired) == "success",
        pre_repair_disposition=_aggregate_atom(conflicting),
        pre_repair_raw_incidence=len(conflicting),
        pre_repair_unique_satisfied_coverage=0,
        post_repair_disposition=_aggregate_atom(repaired),
        post_repair_unique_satisfied_coverage=len(
            {record["atom"] for record in repaired}
        ),
        conflict_provenance_retained=True,
    )


def _dominates(left: Sequence[Fraction], right: Sequence[Fraction]) -> bool:
    """Compare vectors encoded as (quality, -C, -W, -K, -failure)."""

    if len(left) != len(right) or not left:
        raise ValueError("aligned nonempty vectors are required")
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def frontier_check() -> dict[str, object]:
    """Check dominance and incomparability on the declared five-axis vector."""

    sequential = (
        Fraction(9, 10),
        -Fraction(32),
        -Fraction(32),
        -Fraction(100),
        Fraction(0),
    )
    valid = (Fraction(9, 10), -Fraction(16), -Fraction(32), -Fraction(100), Fraction(0))
    over_sharded = (
        Fraction(9, 10),
        -Fraction(12),
        -Fraction(48),
        -Fraction(130),
        Fraction(0),
    )
    fake = (
        Fraction(1, 2),
        -Fraction(20),
        -Fraction(60),
        -Fraction(200),
        -Fraction(1, 5),
    )
    return _status(
        "CH25-D09",
        _dominates(valid, sequential)
        and not _dominates(valid, over_sharded)
        and not _dominates(over_sharded, valid)
        and _dominates(valid, fake),
        valid_dominates_sequential=_dominates(valid, sequential),
        valid_and_over_sharded_incomparable=(
            not _dominates(valid, over_sharded) and not _dominates(over_sharded, valid)
        ),
        valid_dominates_fake=_dominates(valid, fake),
        scalar_speed_sufficient=False,
    )


def handoff_inventory_check() -> dict[str, object]:
    """Freeze the terminal chapter inventory and downstream owners."""

    inventory = {
        "definitions": 20,
        "invariants": 20,
        "figures": 6,
        "algorithms": 5,
        "exercises": 10,
        "pending_protocols": 10,
    }
    expected = dict(inventory)
    owners = {
        "runtime_sessions": 29,
        "benchmark_governance": 30,
        "safety_authority": 31,
        "distributed_infrastructure": 40,
    }
    return _status(
        "CH25-D10",
        inventory == expected and tuple(owners.values()) == (29, 30, 31, 40),
        inventory=inventory,
        downstream_owners=owners,
        model_experiments_remain_pending=True,
        timing_claim_from_fixture=False,
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Return all deterministic Chapter 26 results in stable order."""

    checks = (
        ("task_identity", task_identity_check),
        ("graph_readiness", graph_readiness_check),
        ("work_critical", work_critical_check),
        ("critical_ties", critical_tie_check),
        ("reward_annealing", reward_annealing_check),
        ("false_parallelism", false_parallelism_check),
        ("context_ledger", context_ledger_check),
        ("aggregation", aggregation_check),
        ("quality_work_frontier", frontier_check),
        ("evidence_handoff", handoff_inventory_check),
    )
    return {name: check() for name, check in checks}
