"""Executable semantics for the Chapter 24 teaching fixtures.

The module is deliberately dependency-free and uses exact rational arithmetic
where the chapter makes an exact claim.  It is an oracle for the chapter's
created examples, not an implementation of Kimi K3's private systems.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from typing import Mapping, Sequence

from .graphs import (
    concept_graph_check as concept_graph_check,
    connected_components as connected_components,
    contamination_check as contamination_check,
    topological_order as topological_order,
)

from .studio_world import (
    ALTERNATE_TRACE,
    BASE_TRACE,
    CANONICAL_ACTIONS,
    COORDINATE_CONVENTION,
    HARNESS_TO_INSPECTION,
    StudioTask,
    StudioWorld,
    adapt_inspection,
    box_center,
    event_policy_history,
    inspect_boxes,
    run_studio_attempt,
)

HARNESS_COUNTS = {"flat": 30, "namespaced": 20, "compact": 10}
ADMISSION_GATES = (
    "parse",
    "interface",
    "feasibility",
    "evaluator",
    "isolation",
    "split",
    "safety",
    "resource",
)
HANDOFF_INVARIANTS = (
    "closed-dynamics",
    "complete-harness",
    "adapter-legality",
    "separated-identities",
    "trace-conservation",
    "declared-equivalence",
    "harness-support",
    "acyclic-graph",
    "graph-dispositions",
    "material-provenance",
    "packet-ancestry",
    "reproducible-synthesis",
    "admission-before-sampling",
    "protected-feasibility",
    "verifier-isolation",
    "split-contamination",
    "joint-coverage",
    "event-causality",
    "final-state-verification",
    "evidence-handoff",
)


def trace_probability(
    policy_probabilities: Sequence[Fraction],
    observation_probabilities: Sequence[Fraction],
) -> Fraction:
    """Multiply the factors of one specified legal finite trace."""

    probability = Fraction(1)
    for factor in (*policy_probabilities, *observation_probabilities):
        if factor < 0 or factor > 1:
            raise ValueError("probability factors must lie in [0, 1]")
        probability *= factor
    return probability


def environment_trace_check() -> dict[str, object]:
    """Return diagnostics for the environment trace invariant."""
    _, identifiers = adapt_inspection("inspect_boxes", {"ids": ["red", "blue"]})
    observation = inspect_boxes(identifiers)
    red = box_center(observation["red"])
    blue = box_center(observation["blue"])
    probability = trace_probability(
        (Fraction(3, 5), Fraction(2, 5)), (Fraction(9, 10),)
    )
    legal_actions = set(CANONICAL_ACTIONS)
    trace_is_legal = set(CANONICAL_ACTIONS).issubset(legal_actions)
    return {
        "status": "pass"
        if (red, blue, probability, trace_is_legal)
        == ((40, 40), (40, 100), Fraction(27, 125), True)
        else "fail",
        "red_center": list(red),
        "blue_center": list(blue),
        "vertical_centers": [red[1], blue[1]],
        "two_action_probability": str(probability),
        "canonical_action_count": len(CANONICAL_ACTIONS),
    }


def semantic_pairing_check() -> dict[str, object]:
    """Return diagnostics for the semantic pairing invariant."""
    flat = adapt_inspection("inspect_boxes", {"ids": ["red", "blue"]})
    namespaced = adapt_inspection(
        "assets.inspect_bboxes", {"shape_ids": ["red", "blue"]}
    )
    states = ("s0", "s1", "s2")
    state_map = {state: f"ui.{state}" for state in states}
    action_map = {"inspect": "canvas.inspect", "submit": "task.submit"}
    transitions = {("s0", "inspect"): "s1", ("s1", "submit"): "s2"}
    mapped = {
        (state_map[state], action_map[action]): state_map[target]
        for (state, action), target in transitions.items()
    }
    commutes = all(
        mapped[(state_map[state], action_map[action])] == state_map[target]
        for (state, action), target in transitions.items()
    )
    total = set(action_map) == {action for _, action in transitions}
    paired = {name: run_studio_attempt(name) for name in HARNESS_TO_INSPECTION}
    same_observations = (
        paired["studio_flat"].policy_observations
        == paired["studio_namespaced"].policy_observations
    )
    both_succeed = all(attempt.world.succeeded() for attempt in paired.values())
    context_loss_rejected = False
    try:
        run_studio_attempt("studio_namespaced", drop_convention=True)
    except ValueError as error:
        context_loss_rejected = "coordinate convention" in str(error)
    return {
        "status": "pass"
        if commutes
        and total
        and flat == namespaced
        and same_observations
        and both_succeed
        and context_loss_rejected
        else "fail",
        "initial_state_preserved": state_map["s0"] == "ui.s0",
        "transition_square_commutes": commutes,
        "terminal_truth_preserved": state_map["s2"] == "ui.s2",
        "selected_configurations": sorted(paired),
        "paired_observations_equal": same_observations,
        "paired_outcomes_pass": both_succeed,
        "context_loss_rejected": context_loss_rejected,
    }


def harness_population_check() -> dict[str, object]:
    """Return diagnostics for the harness population invariant."""
    total = sum(HARNESS_COUNTS.values())
    masses = {name: Fraction(count, total) for name, count in HARNESS_COUNTS.items()}
    all_supported = all(count > 0 for count in HARNESS_COUNTS.values())
    paired_only = {
        ("A", "flat"): 30,
        ("A", "namespaced"): 0,
        ("B", "flat"): 0,
        ("B", "namespaced"): 30,
    }
    crossed = {pair: 15 for pair in paired_only}

    def marginals(schedule: Mapping[tuple[str, str], int]) -> dict[str, int]:
        return {
            "A": sum(count for (family, _), count in schedule.items() if family == "A"),
            "B": sum(count for (family, _), count in schedule.items() if family == "B"),
            "flat": sum(
                count for (_, harness), count in schedule.items() if harness == "flat"
            ),
            "namespaced": sum(
                count
                for (_, harness), count in schedule.items()
                if harness == "namespaced"
            ),
        }

    same_marginals = (
        marginals(paired_only)
        == marginals(crossed)
        == {"A": 30, "B": 30, "flat": 30, "namespaced": 30}
    )
    paired_empty = sum(count == 0 for count in paired_only.values())
    crossed_empty = sum(count == 0 for count in crossed.values())
    return {
        "status": "pass"
        if total == 60
        and sum(masses.values()) == 1
        and all_supported
        and same_marginals
        and paired_empty == 2
        and crossed_empty == 0
        else "fail",
        "counts": dict(HARNESS_COUNTS),
        "masses": {name: str(value) for name, value in masses.items()},
        "support_complete": all_supported,
        "joint_marginals_equal": same_marginals,
        "paired_only_empty_cells": paired_empty,
        "crossed_empty_cells": crossed_empty,
    }


def admission_check() -> dict[str, object]:
    """Return diagnostics for the admission invariant."""
    rejected = {
        "parse": 2,
        "interface": 1,
        "feasibility": 2,
        "isolation": 1,
        "split": 1,
    }
    candidates = 12
    admitted = candidates - sum(rejected.values())
    all_gates_declared = set(rejected).issubset(ADMISSION_GATES)
    reference = run_studio_attempt("studio_flat")
    missing_edit = StudioTask(
        allowed_actions=frozenset(set(CANONICAL_ACTIONS) - {"EditReport"})
    )
    missing_action_rejected = False
    try:
        run_studio_attempt("studio_flat", task=missing_edit)
    except ValueError as error:
        missing_action_rejected = "unavailable action: EditReport" in str(error)
    insufficient_budget_rejected = False
    try:
        run_studio_attempt("studio_flat", task=StudioTask(action_limit=5))
    except ValueError as error:
        insufficient_budget_rejected = "out of actions" in str(error)
    fresh = StudioWorld()
    final_words_only = StudioWorld()
    final_words_only.execute("Submit", {"final_sentence": "blue below red"})
    fabricated = run_studio_attempt("studio_flat", BASE_TRACE[:4]).world
    fabricated.execute(
        "AttachEvidenceAndClose",
        {
            "centers": {"red": (40, 40), "blue": (40, 40)},
            "asset_id": fabricated.task.asset_id,
            "coordinate_convention": COORDINATE_CONVENTION,
        },
    )
    fabricated.execute("Submit")
    fabricated_evidence_rejected = not fabricated.succeeded()
    return {
        "status": "pass"
        if admitted == 5
        and all_gates_declared
        and reference.world.succeeded()
        and missing_action_rejected
        and insufficient_budget_rejected
        and fabricated_evidence_rejected
        and not final_words_only.succeeded()
        and not fresh.succeeded()
        else "fail",
        "gate_count": len(ADMISSION_GATES),
        "candidate_count": candidates,
        "admitted_count": admitted,
        "yield": str(Fraction(admitted, candidates)),
        "fresh_reference_passes": reference.world.succeeded(),
        "fresh_learner_unsolved": not fresh.succeeded(),
        "missing_edit_report_rejected": missing_action_rejected,
        "insufficient_budget_rejected": insufficient_budget_rejected,
        "fabricated_evidence_rejected": fabricated_evidence_rejected,
        "final_words_only_rejected": not final_words_only.succeeded(),
        "rejection_fractions": {
            gate: str(Fraction(count, candidates)) for gate, count in rejected.items()
        },
    }


def persistent_aet_check() -> dict[str, object]:
    """Return diagnostics for the persistent AET invariant."""
    costs = (1, 2, 2, 1)
    event = {"occurred": 3, "exposed": 5, "consumed": 6}
    causal = event["occurred"] <= event["exposed"] <= event["consumed"]
    hidden = {"expected_centers", "distractor_map", "protected_path", "answer_key"}
    agent_view = {"messages", "asset", "report", "task_status"}
    isolated = hidden.isdisjoint(agent_view)
    base = run_studio_attempt("studio_flat", BASE_TRACE)
    alternate = run_studio_attempt("studio_flat", ALTERNATE_TRACE)
    after_edit = run_studio_attempt("studio_flat", BASE_TRACE[:4]).world
    after_close = run_studio_attempt("studio_flat", BASE_TRACE[:5]).world
    new_attempt = StudioWorld()
    state_persists = (
        after_edit.report_relation == "blue below red"
        and after_close.report_relation == "blue below red"
        and after_close.ticket_closed
        and not after_edit.ticket_closed
    )
    episode_isolated = (
        new_attempt.report_relation == "blue above red"
        and not new_attempt.ticket_closed
        and not new_attempt.evidence
    )
    same_time_four_history = event_policy_history(True, 4) == event_policy_history(
        False, 4
    )
    later_history_diverges = event_policy_history(True, 6) != event_policy_history(
        False, 6
    )
    return {
        "status": "pass"
        if sum(costs) == 6
        and causal
        and isolated
        and base.world.succeeded()
        and alternate.world.succeeded()
        and state_persists
        and episode_isolated
        and same_time_four_history
        and later_history_diverges
        else "fail",
        "least_cost": sum(costs),
        "within_execution_budget": sum(costs) <= 6,
        "submission_count": 1,
        "unreachable_without_edit_report": admission_check()[
            "missing_edit_report_rejected"
        ],
        "event_times": event,
        "time_four_action_was_stale_but_legal": same_time_four_history,
        "later_event_history_diverges": later_history_diverges,
        "state_persists_across_actions": state_persists,
        "new_episode_isolated": episode_isolated,
        "base_and_alternate_pass": base.world.succeeded()
        and alternate.world.succeeded(),
        "base_action_count": len(base.world.events),
        "alternate_action_count": len(alternate.world.events),
        "hidden_dependencies_isolated": isolated,
    }


def handoff_check() -> dict[str, object]:
    """Return diagnostics for the handoff invariant."""
    evidence_statuses = (
        "source-reported",
        "derived",
        "chapter-created",
        "deterministically-checked",
        "measured",
        "pending",
        "not-publicly-specified",
    )
    base_verdict = run_studio_attempt("studio_flat").world.verdict()
    record = {
        "task": "studio-ops-001",
        "initial_asset_id": StudioTask().initial_observation()["asset_id"],
        "harness": "studio_flat",
        "canonical_actions": list(BASE_TRACE),
        "base_verdict": base_verdict,
        "hidden_case": "unrun",
        "invariants": list(HANDOFF_INVARIANTS),
        "evidence_statuses": list(evidence_statuses),
        "pending_protocols": 8,
    }
    digest = sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return {
        "status": "pass"
        if len(set(HANDOFF_INVARIANTS)) == 20 and len(evidence_statuses) == 7
        else "fail",
        "task": record["task"],
        "invariant_count": len(HANDOFF_INVARIANTS),
        "evidence_status_count": len(evidence_statuses),
        "pending_protocol_count": record["pending_protocols"],
        "hidden_case_status": record["hidden_case"],
        "base_verdict_pass": all(base_verdict.values()),
        "record_digest": digest,
    }


def all_checks() -> dict[str, Mapping[str, object]]:
    """Return all deterministic Chapter 24 evidence records."""

    return {
        "environment_trace": environment_trace_check(),
        "semantic_pairing": semantic_pairing_check(),
        "harness_population": harness_population_check(),
        "concept_graph": concept_graph_check(),
        "admission": admission_check(),
        "contamination": contamination_check(),
        "persistent_aet": persistent_aet_check(),
        "evidence_handoff": handoff_check(),
    }


if __name__ == "__main__":
    print(json.dumps(all_checks(), indent=2, sort_keys=True))
