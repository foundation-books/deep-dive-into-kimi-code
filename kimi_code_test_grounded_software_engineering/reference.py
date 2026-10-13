"""Finite deterministic oracles for Chapter 29 repository engineering.

The module models only the chapter's registered teaching fixture.  It performs
no filesystem mutation, process execution, network access, or model inference.
"""

from __future__ import annotations

from itertools import combinations
from collections.abc import Iterable, Mapping
from typing import cast

from .reference_repository import (
    canonical_digest as canonical_digest,
    repository_fingerprint as repository_fingerprint,
)

FIXTURE_VERSION = "ch29-studio-code-v1"


def resolve_instructions(
    records: Iterable[Mapping[str, object]], target: str
) -> list[str]:
    """Resolve applicable instruction identities by precedence then locality."""
    applicable: list[tuple[int, int, str]] = []
    for record in records:
        scope = str(record["scope"]).rstrip("/")
        if scope and target != scope and not target.startswith(scope + "/"):
            continue
        applicable.append(
            (
                int(str(record["precedence"])),
                len(scope.split("/")) if scope else 0,
                str(record["id"]),
            )
        )
    return [identity for _, _, identity in sorted(applicable)]


def select_context(
    items: dict[str, tuple[int, int]], capacity: int, mandatory: set[str]
) -> tuple[list[str], int, int]:
    """Maximize value subject to capacity and mandatory-item retention."""
    feasible: list[tuple[int, int, tuple[str, ...]]] = []
    names = sorted(items)
    for count in range(len(names) + 1):
        for choice in combinations(names, count):
            selected = set(choice)
            cost = sum(items[name][0] for name in choice)
            if mandatory <= selected and cost <= capacity:
                value = sum(items[name][1] for name in choice)
                feasible.append((value, -cost, choice))
    if not feasible:
        raise ValueError("no context set satisfies mandatory items and capacity")
    value, negative_cost, choice = max(feasible)
    return list(choice), -negative_cost, value


def distinguishing_power(first_prediction: object, second_prediction: object) -> int:
    """Return one exactly when two hypotheses predict different observations."""
    return int(first_prediction != second_prediction)


def accepted_hypotheses(
    predictions: Mapping[str, object], observation: object
) -> list[str]:
    """Accept a unique matching hypothesis only when alternatives differ."""
    matching = [
        name for name, prediction in predictions.items() if prediction == observation
    ]
    if len(set(predictions.values())) < 2 or len(matching) != 1:
        return []
    return matching


def action_admissible(
    action: Mapping[str, object],
    authority: Mapping[str, set[str]],
    permitted_paths: set[str],
    protected_paths: set[str],
    budget: int,
) -> bool:
    """Check fixed authority, scope, preservation, precondition, and budget gates."""
    effect_class = str(action["effect_class"])
    effects = set(cast(Iterable[str], action["effects"]))
    allowed_effects = authority.get(effect_class, set())
    paths = set(cast(Iterable[str], action["paths"]))
    return (
        bool(action["arguments_valid"])
        and bool(action["preconditions_hold"])
        and bool(action["approval_current"])
        and int(str(action["cost"])) <= budget
        and effects <= allowed_effects
        and paths <= permitted_paths
        and not paths & protected_paths
    )


def preservation_holds(
    incoming: dict[str, str], final: dict[str, str], task_owned_paths: set[str]
) -> bool:
    """Require protected incoming paths to remain byte-identical and unclaimed."""
    return all(
        path not in task_owned_paths and final.get(path) == digest
        for path, digest in incoming.items()
    )


def validation_complete(
    required: set[str],
    evidence: Iterable[Mapping[str, object]],
    final_state: str,
    environment: str,
) -> tuple[bool, set[str]]:
    """Compute predicates covered by current supporting evidence."""
    covered: set[str] = set()
    for atom in evidence:
        if (
            atom["judgment"] == "support"
            and atom["repository_state"] == final_state
            and atom["environment"] == environment
        ):
            covered.add(str(atom["predicate"]))
    return required <= covered, covered


def attribute_failure(
    observation: str | None,
    baseline: str | None,
    terminal: bool,
) -> str:
    """Classify three finite failure cases without inventing causality."""
    if not terminal:
        return "unknown"
    if observation == baseline and observation is not None:
        return "pre_existing"
    if observation == "target_failure":
        return "target_not_fixed"
    return "introduced_regression"


def reconcile_resume(
    checkpoint_versions: dict[str, str],
    live_versions: dict[str, str],
    open_attempt: bool,
    effect_observation: str | None,
) -> dict[str, object]:
    """Compare live state and retain an open effect until explicitly reconciled."""
    drift = sorted(
        key
        for key in set(checkpoint_versions) | set(live_versions)
        if checkpoint_versions.get(key) != live_versions.get(key)
    )
    unresolved = open_attempt and effect_observation not in {"no_effect", "completed"}
    return {
        "drift": drift,
        "effect_status": "unknown" if unresolved else (effect_observation or "none"),
        "may_mutate": not drift and not unresolved,
    }


def conflict_edges(
    actions: Mapping[str, tuple[set[str], set[str] | None]],
) -> set[tuple[str, str]]:
    """Build read/write conflicts; an unknown write set conflicts universally."""
    edges: set[tuple[str, str]] = set()
    names = sorted(actions)
    for left, right in combinations(names, 2):
        left_reads, left_writes = actions[left]
        right_reads, right_writes = actions[right]
        if left_writes is None or right_writes is None:
            conflict = True
        else:
            conflict = bool(
                left_writes & (right_reads | right_writes)
                or right_writes & (left_reads | left_writes)
            )
        if conflict:
            edges.add((left, right))
    return edges


def maximal_safe_sets(names: set[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    """Enumerate maximal independent sets in a small conflict graph."""
    independent: list[set[str]] = []
    for count in range(len(names) + 1):
        for choice in combinations(sorted(names), count):
            selected = set(choice)
            if all(not ({left, right} <= selected) for left, right in edges):
                independent.append(selected)
    maximal = [
        choice
        for choice in independent
        if not any(choice < other for other in independent)
    ]
    return [
        sorted(choice) for choice in sorted(maximal, key=lambda value: sorted(value))
    ]


def _result(passed: bool, **payload: object) -> dict[str, object]:
    return {
        "status": "pass" if passed else "fail",
        "fixture_version": FIXTURE_VERSION,
        **payload,
    }


def _repository_context_checks() -> dict[str, dict[str, object]]:
    incoming_digest = canonical_digest("user README note\n")
    state = {
        "repository": "studio",
        "base": "r0",
        "branch": "repair-coordinate",
        "tracked_diff": {"README.md": incoming_digest},
        "untracked": [],
        "environment": "python-fixture-v1",
    }
    clean_state = {**state, "tracked_diff": {}}
    state_check = repository_fingerprint(state) != repository_fingerprint(clean_state)

    records = [
        {"id": "root", "scope": "", "precedence": 0},
        {"id": "tests-local", "scope": "tests", "precedence": 1},
        {"id": "docs-local", "scope": "docs", "precedence": 1},
    ]
    production_stack = resolve_instructions(records, "transforms.py")
    test_stack = resolve_instructions(records, "tests/test_transforms.py")

    chosen, context_cost, context_value = select_context(
        {"one": (7, 8), "two": (6, 7), "three": (5, 6), "four": (4, 5)},
        18,
        {"one", "four"},
    )
    return {
        "repository_fingerprint": _result(
            state_check,
            dirty=repository_fingerprint(state),
            clean=repository_fingerprint(clean_state),
        ),
        "instruction_resolution": _result(
            production_stack == ["root"] and test_stack == ["root", "tests-local"],
            production=production_stack,
            test=test_stack,
        ),
        "context_selection": _result(
            chosen == ["four", "one", "two"]
            and context_cost == 17
            and context_value == 20,
            selected=chosen,
            cost=context_cost,
            value=context_value,
        ),
    }


def _localization_patch_checks() -> dict[str, dict[str, object]]:
    incoming_digest = canonical_digest("user README note\n")

    symmetric = {"canvas": (40, 80), "transform": (40, 80)}
    asymmetric = {"canvas": (40, 60), "transform": (40, 80)}
    accepted_symmetric = accepted_hypotheses(symmetric, (40, 80))
    accepted_asymmetric = accepted_hypotheses(asymmetric, (40, 80))

    authority = {
        "read_only": {"local_read", "local_test"},
        "reversible_local": {"source_edit", "test_edit"},
        "destructive_local": set(),
        "external": set(),
    }
    permitted = {"transforms.py", "tests/test_transforms.py"}
    protected = {"README.md"}
    edit = {
        "effect_class": "reversible_local",
        "effects": {"source_edit"},
        "paths": {"transforms.py"},
        "arguments_valid": True,
        "preconditions_hold": True,
        "approval_current": True,
        "cost": 2,
    }
    push = {**edit, "effect_class": "external", "effects": {"push"}, "paths": set()}
    user_edit = {**edit, "paths": {"README.md"}}
    admission = action_admissible(edit, authority, permitted, protected, 5)
    rejected = not action_admissible(push, authority, permitted, protected, 5)
    protected_rejected = not action_admissible(
        user_edit, authority, permitted, protected, 5
    )

    final = {
        "README.md": incoming_digest,
        "transforms.py": "new",
        "tests/test_transforms.py": "new",
    }
    preservation = preservation_holds(
        {"README.md": incoming_digest},
        final,
        {"transforms.py", "tests/test_transforms.py"},
    )
    absorbed = preservation_holds({"README.md": incoming_digest}, final, {"README.md"})
    return {
        "localization": _result(
            not accepted_symmetric
            and accepted_asymmetric == ["transform"]
            and distinguishing_power((40, 60), (40, 80)) == 1,
            symmetric=accepted_symmetric,
            asymmetric=accepted_asymmetric,
        ),
        "action_admission": _result(
            admission and rejected and protected_rejected,
            edit=admission,
            push_rejected=rejected,
            protected_rejected=protected_rejected,
        ),
        "patch_preservation": _result(
            preservation and not absorbed,
            preservation=preservation,
            absorption_rejected=not absorbed,
        ),
    }


def _validation_recovery_checks() -> dict[str, dict[str, object]]:

    required = {"vertical", "round_trip", "component", "scope", "user_diff"}
    atoms = [
        {
            "predicate": name,
            "judgment": "support",
            "repository_state": "rf",
            "environment": "e0",
        }
        for name in sorted(required)
    ]
    complete, covered = validation_complete(required, atoms, "rf", "e0")
    incomplete, _ = validation_complete(required, atoms[:-1], "rf", "e0")

    attributions = [
        attribute_failure("target_failure", "pass", True),
        attribute_failure("optional_import", "optional_import", True),
        attribute_failure(None, None, False),
    ]

    checkpoint_versions = {
        "repository": "r0+patch",
        "instructions": "i0",
        "authority": "a0",
        "tools": "t0",
        "environment": "e0",
    }
    unresolved = reconcile_resume(checkpoint_versions, checkpoint_versions, True, None)
    reconciled = reconcile_resume(
        checkpoint_versions, checkpoint_versions, True, "no_effect"
    )
    drifted = reconcile_resume(
        checkpoint_versions, {**checkpoint_versions, "instructions": "i1"}, False, None
    )

    actions = {
        "one": ({"a", "b"}, {"c"}),
        "two": ({"c", "d"}, {"e"}),
        "three": ({"f"}, set()),
    }
    edges = conflict_edges(actions)
    safe_sets = maximal_safe_sets(set(actions), edges)
    return {
        "validation_coverage": _result(
            complete and not incomplete and covered == required,
            covered=sorted(covered),
            required=sorted(required),
        ),
        "failure_attribution": _result(
            attributions == ["target_not_fixed", "pre_existing", "unknown"],
            classes=attributions,
        ),
        "resume_reconciliation": _result(
            not bool(unresolved["may_mutate"])
            and bool(reconciled["may_mutate"])
            and not bool(drifted["may_mutate"]),
            unresolved=unresolved,
            reconciled=reconciled,
            drifted=drifted,
        ),
        "conflict_graph": _result(
            edges == {("one", "two")}
            and safe_sets == [["one", "three"], ["three", "two"]],
            edges=sorted([list(edge) for edge in edges]),
            maximal_safe_sets=safe_sets,
        ),
    }


def all_checks() -> dict[str, dict[str, object]]:
    """Run the ten registered finite checks."""
    return {
        **_repository_context_checks(),
        **_localization_patch_checks(),
        **_validation_recovery_checks(),
    }
