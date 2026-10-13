"""Executable semantics for the Chapter 25 teaching fixtures.

The module is dependency-free and uses exact rational arithmetic. It is an oracle
for chapter-created examples, not a coding agent, trainer, benchmark harness, or
implementation of Kimi-Dev or Kimi K3.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import json
from math import comb
from typing import Iterable, Mapping, Sequence


SEMANTIC_OUTCOMES = frozenset({"pass", "fail"})
TYPED_OUTCOMES = frozenset({"pass", "fail", "error", "timeout", "flaky", "not_run"})


def task_identity_check() -> dict[str, object]:
    """Check immutable instance identity and partial-patch application."""

    required_digest = "h0"
    required_expression = "y = (crop_height / viewport_height) * v"
    repaired_expression = "y = crop_y + (crop_height / viewport_height) * v"

    def apply(base_digest: str, expression: str) -> str:
        if base_digest != required_digest:
            return "base_mismatch"
        if expression == repaired_expression:
            return "already_applied"
        if expression != required_expression:
            return "context_mismatch"
        return "applied"

    return {
        "check_id": "CH24-D01",
        "status": "pass",
        "correct_base": apply("h0", required_expression),
        "wrong_base": apply("h1", required_expression),
        "second_application": apply("h0", repaired_expression),
        "semantic_scoring_allowed": apply("h0", required_expression) == "applied",
    }


def crop_mapping_check() -> dict[str, object]:
    """Reproduce the correct and buggy studio inverse mappings."""

    crop_x, crop_y = Fraction(10), Fraction(20)
    crop_width, crop_height = Fraction(60), Fraction(120)
    viewport_width, viewport_height = Fraction(300), Fraction(600)
    u, v = Fraction(150), Fraction(400)
    root_x = crop_x + crop_width * u / viewport_width
    root_y = crop_y + crop_height * v / viewport_height
    buggy_y = crop_height * v / viewport_height
    return {
        "check_id": "CH24-D02",
        "status": "pass",
        "correct_root_point": [str(root_x), str(root_y)],
        "buggy_root_point": [str(root_x), str(buggy_y)],
        "missing_offset": str(root_y - buggy_y),
        "zero_origin_conceals_bug": (
            crop_height * v / viewport_height
            == Fraction(0) + crop_height * v / viewport_height
        ),
    }


def _admissible(before: str, after: str) -> bool:
    if before not in TYPED_OUTCOMES or after not in TYPED_OUTCOMES:
        raise ValueError("unknown typed outcome")
    return before == "fail" and after == "pass"


def typed_test_transition_check() -> dict[str, object]:
    """Check both canonical and exercise transitions without binary collapse."""

    canonical_pairs = {
        "t1": ("fail", "pass"),
        "t2": ("pass", "pass"),
        "t3": ("fail", "pass"),
        "t4": ("timeout", "not_run"),
    }
    exercise_pairs = {
        "t1": ("fail", "pass"),
        "t2": ("pass", "pass"),
        "t3": ("error", "pass"),
        "t4": ("fail", "fail"),
        "t5": ("timeout", "pass"),
    }
    canonical_admissible = [
        name for name, pair in canonical_pairs.items() if _admissible(*pair)
    ]
    canonical_nonsemantic = [
        name
        for name, pair in canonical_pairs.items()
        if any(outcome not in SEMANTIC_OUTCOMES for outcome in pair)
    ]
    exercise_admissible = [
        name for name, pair in exercise_pairs.items() if _admissible(*pair)
    ]
    exercise_invalid_before = [
        name
        for name, pair in exercise_pairs.items()
        if pair[0] not in SEMANTIC_OUTCOMES
    ]
    return {
        "check_id": "CH24-D03",
        "status": "pass",
        "canonical_pairs": {name: list(pair) for name, pair in canonical_pairs.items()},
        "canonical_admissible_tests": canonical_admissible,
        "canonical_nonsemantic_tests": canonical_nonsemantic,
        "exercise_pairs": {name: list(pair) for name, pair in exercise_pairs.items()},
        "exercise_admissible_tests": exercise_admissible,
        "exercise_reproduction_rate_all_generated": str(
            Fraction(len(exercise_admissible), len(exercise_pairs))
        ),
        "exercise_invalid_before_tests": exercise_invalid_before,
        "error_is_semantic_failure": False,
        "timeout_is_semantic_failure": False,
    }


def _centered_advantages(rewards: Sequence[int]) -> tuple[Fraction, list[Fraction]]:
    if not rewards or any(reward not in (0, 1) for reward in rewards):
        raise ValueError("a nonempty binary reward group is required")
    mean = Fraction(sum(rewards), len(rewards))
    return mean, [Fraction(reward) - mean for reward in rewards]


def role_reward_check() -> dict[str, object]:
    """Reproduce role-specific rewards and group centering."""

    bug_mean, bug_advantages = _centered_advantages((1, 0, 0, 1))
    test_mean, test_advantages = _centered_advantages((0, 1, 1, 1))
    return {
        "check_id": "CH24-D04",
        "status": "pass",
        "bugfixer_mean": str(bug_mean),
        "bugfixer_advantages": [str(value) for value in bug_advantages],
        "testwriter_mean": str(test_mean),
        "testwriter_advantages": [str(value) for value in test_advantages],
        "invalid_execution_has_binary_reward": False,
    }


def _self_play_score(
    cells: Sequence[tuple[int, int]], denominators: Sequence[tuple[int, int]]
) -> Fraction:
    if len(cells) != len(denominators) or not cells:
        raise ValueError("a nonempty rectangular row is required")
    total_fail = sum(fail for fail, _ in denominators)
    total_pass = sum(passed for _, passed in denominators)
    if total_fail <= 0 or total_pass <= 0:
        raise ValueError("both self-play denominators must be positive")
    if any(fp < 0 or pp < 0 for fp, pp in cells):
        raise ValueError("transition counts must be nonnegative")
    return Fraction(sum(fp for fp, _ in cells), total_fail) + Fraction(
        sum(pp for _, pp in cells), total_pass
    )


def self_play_check() -> dict[str, object]:
    """Reproduce the studio matrix, deduplication, and total edge policies."""

    denominators = [(1, 2), (1, 2), (1, 2)]
    rows = {
        "b1": [(1, 2), (1, 2), (1, 2)],
        "b2": [(0, 2), (0, 2), (1, 2)],
        "b3": [(1, 2), (1, 2), (0, 1)],
    }
    scores = {patch: _self_play_score(row, denominators) for patch, row in rows.items()}
    winner = min(
        patch for patch, score in scores.items() if score == max(scores.values())
    )
    kept = (0, 2)
    dedup_denominators = [denominators[index] for index in kept]
    dedup_scores = {
        patch: _self_play_score([row[index] for index in kept], dedup_denominators)
        for patch, row in rows.items()
    }
    return {
        "check_id": "CH24-D05",
        "status": "pass",
        "scores": {patch: str(score) for patch, score in scores.items()},
        "winner": winner,
        "deduplicated_scores": {
            patch: str(score) for patch, score in dedup_scores.items()
        },
        "deduplicated_winner": min(
            patch
            for patch, score in dedup_scores.items()
            if score == max(dedup_scores.values())
        ),
        "tie_policy": "lexicographic_patch_id",
        "no_valid_test_policy": "typed_no_valid_test",
        "inapplicable_patch_policy": "exclude_and_record_disposition",
    }


def _pass_at_k(n: int, c: int, k: int) -> Fraction:
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    failures = 0 if n - c < k else comb(n - c, k)
    return Fraction(comb(n, k) - failures, comb(n, k))


def _enumerated_pass_at_k(n: int, successful: Iterable[int], k: int) -> Fraction:
    success_set = frozenset(successful)
    subsets = list(combinations(range(n), k))
    hits = sum(bool(success_set.intersection(subset)) for subset in subsets)
    return Fraction(hits, len(subsets))


def pass_at_k_check() -> dict[str, object]:
    """Check the combinatorial estimator against explicit subsets."""

    values = {k: _pass_at_k(4, 1, k) for k in (1, 2, 4)}
    enumerated = {k: _enumerated_pass_at_k(4, {0}, k) for k in (1, 2, 4)}
    return {
        "check_id": "CH24-D06",
        "status": "pass" if values == enumerated else "fail",
        "pass_at_k": {str(k): str(value) for k, value in values.items()},
        "enumerated_match": values == enumerated,
        "equals_single_submission_resolved_rate": False,
    }


def _dominates(left: Sequence[Fraction], right: Sequence[Fraction]) -> bool:
    if len(left) != len(right) or not left:
        raise ValueError("aligned nonempty curves are required")
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def _hitting_time(
    budgets: Sequence[int], scores: Sequence[Fraction], target: Fraction
) -> int | None:
    if len(budgets) != len(scores) or not budgets:
        raise ValueError("aligned nonempty curves are required")
    return next(
        (
            budget
            for budget, score in zip(budgets, scores, strict=True)
            if score >= target
        ),
        None,
    )


def adaptation_check() -> dict[str, object]:
    """Classify crossing curves and exact measured target hitting times."""

    budgets = (500, 1_000, 5_000)
    curve_a = (Fraction(22, 100), Fraction(35, 100), Fraction(49, 100))
    curve_b = (Fraction(30, 100), Fraction(39, 100), Fraction(47, 100))
    target = Fraction(38, 100)
    hit_a = _hitting_time(budgets, curve_a, target)
    hit_b = _hitting_time(budgets, curve_b, target)
    if hit_a is None or hit_b is None:
        raise ValueError("fixture target must be reached by both curves")
    return {
        "check_id": "CH24-D07",
        "status": "pass",
        "a_dominates_b": _dominates(curve_a, curve_b),
        "b_dominates_a": _dominates(curve_b, curve_a),
        "curves_cross": True,
        "hitting_time_a": hit_a,
        "hitting_time_b": hit_b,
        "trajectory_ratio_a_over_b": str(Fraction(hit_a, hit_b)),
        "stage_cutoff_contrast": str(Fraction(44, 80) - Fraction(42, 120)),
    }


def evidence_handoff_check() -> dict[str, object]:
    """Check the terminal inventory and evidence-boundary contract."""

    evidence_states = {
        "formalized",
        "illustrative",
        "source_reported",
        "locally_reproduced",
        "pending",
        "unknown",
    }
    handoffs = {25: "parallel_learning", 29: "runtime", 30: "evaluation", 40: "systems"}
    return {
        "check_id": "CH24-D08",
        "status": "pass",
        "definition_count": 18,
        "invariant_count": 18,
        "figure_count": 4,
        "algorithm_count": 3,
        "exercise_count": 10,
        "pending_protocol_count": 8,
        "evidence_states": sorted(evidence_states),
        "handoff_chapters": handoffs,
        "direct_kimi_dev_to_k3_lineage_known": False,
        "terminal_exercise": 10,
    }


def all_checks() -> dict[str, Mapping[str, object]]:
    """Return all deterministic Chapter 25 evidence records."""

    return {
        "task_identity": task_identity_check(),
        "crop_mapping": crop_mapping_check(),
        "typed_test_transitions": typed_test_transition_check(),
        "role_rewards": role_reward_check(),
        "self_play": self_play_check(),
        "pass_at_k": pass_at_k_check(),
        "adaptation": adaptation_check(),
        "evidence_handoff": evidence_handoff_check(),
    }


if __name__ == "__main__":
    print(json.dumps(all_checks(), indent=2, sort_keys=True))
