"""Tests for the Chapter 29 finite repository-engineering oracles."""

from __future__ import annotations

import unittest
from fractions import Fraction

from kimi_code_test_grounded_software_engineering.reference import (
    accepted_hypotheses,
    action_admissible,
    all_checks,
    attribute_failure,
    conflict_edges,
    distinguishing_power,
    maximal_safe_sets,
    preservation_holds,
    reconcile_resume,
    resolve_instructions,
    select_context,
    validation_complete,
)
from kimi_code_test_grounded_software_engineering.revision_case import (
    A,
    B,
    C,
    P_ZERO,
    SUITES,
    edit_match_count,
    main_timing_eligible,
    offset_output,
    passes,
    scale_output,
    selection_counts,
    selection_score,
    wrong_round_trip,
)


class Chapter29ReferenceTests(unittest.TestCase):
    def test_all_registered_checks_pass(self) -> None:
        checks = all_checks()
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(check["status"] == "pass" for check in checks.values()))

    def test_nested_instruction_resolution(self) -> None:
        records = [
            {"id": "root", "scope": "", "precedence": 0},
            {"id": "tests", "scope": "tests", "precedence": 1},
        ]
        self.assertEqual(resolve_instructions(records, "src/a.py"), ["root"])
        self.assertEqual(resolve_instructions(records, "tests/a.py"), ["root", "tests"])

    def test_mandatory_context_is_not_traded_for_value(self) -> None:
        chosen, cost, value = select_context(
            {"one": (7, 8), "two": (6, 7), "three": (5, 6), "four": (4, 5)},
            18,
            {"one", "four"},
        )
        self.assertEqual((chosen, cost, value), (["four", "one", "two"], 17, 20))

    def test_localization_needs_different_predictions(self) -> None:
        self.assertEqual(distinguishing_power(80, 80), 0)
        self.assertEqual(accepted_hypotheses({"a": 80, "b": 80}, 80), [])
        self.assertEqual(accepted_hypotheses({"a": 60, "b": 80}, 80), ["b"])

    def test_action_admission_checks_protected_paths(self) -> None:
        action = {
            "effect_class": "reversible_local",
            "effects": {"edit"},
            "paths": {"README.md"},
            "arguments_valid": True,
            "preconditions_hold": True,
            "approval_current": True,
            "cost": 1,
        }
        self.assertFalse(
            action_admissible(
                action, {"reversible_local": {"edit"}}, {"README.md"}, {"README.md"}, 3
            )
        )

    def test_preservation_rejects_claiming_incoming_work(self) -> None:
        incoming = {"README.md": "digest"}
        final = {"README.md": "digest", "code.py": "new"}
        self.assertTrue(preservation_holds(incoming, final, {"code.py"}))
        self.assertFalse(preservation_holds(incoming, final, {"README.md"}))

    def test_only_current_supporting_atoms_cover_predicates(self) -> None:
        atoms = [
            {
                "predicate": "a",
                "judgment": "support",
                "repository_state": "rf",
                "environment": "e0",
            },
            {
                "predicate": "b",
                "judgment": "support",
                "repository_state": "old",
                "environment": "e0",
            },
        ]
        complete, covered = validation_complete({"a", "b"}, atoms, "rf", "e0")
        self.assertFalse(complete)
        self.assertEqual(covered, {"a"})

    def test_failure_classes_preserve_baseline_and_unknown(self) -> None:
        self.assertEqual(attribute_failure("x", "x", True), "pre_existing")
        self.assertEqual(attribute_failure(None, None, False), "unknown")

    def test_resume_blocks_on_unknown_effect_or_drift(self) -> None:
        versions = {"repository": "r0", "instructions": "i0"}
        self.assertFalse(reconcile_resume(versions, versions, True, None)["may_mutate"])
        self.assertFalse(
            reconcile_resume(versions, {**versions, "instructions": "i1"}, False, None)[
                "may_mutate"
            ]
        )

    def test_conflict_graph_and_maximal_sets(self) -> None:
        actions = {
            "one": ({"a"}, {"c"}),
            "two": ({"c"}, {"e"}),
            "three": ({"f"}, set()),
        }
        edges = conflict_edges(actions)
        self.assertEqual(edges, {("one", "two")})
        self.assertEqual(
            maximal_safe_sets(set(actions), edges),
            [["one", "three"], ["three", "two"]],
        )


class Chapter29RevisionCaseTests(unittest.TestCase):
    def test_offset_candidates_and_independent_input(self) -> None:
        self.assertEqual(
            [
                offset_output(A, candidate)
                for candidate in ("original", "b1", "b2", "b3")
            ],
            [(40, 80), (40, 100), (40, 90), (40, 100)],
        )
        self.assertEqual(
            [
                offset_output(C, candidate)
                for candidate in ("original", "b1", "b2", "b3")
            ],
            [(13, 20), (13, 29), (13, 23), (13, 20)],
        )
        self.assertEqual(offset_output(B, "b3"), (7, 8))
        self.assertTrue(passes(P_ZERO, offset_output(P_ZERO, "b2")))

    def test_test_time_selection_uses_two_failures_and_three_preserved_cases(
        self,
    ) -> None:
        self.assertEqual(selection_counts("b1"), (2, 2, 3, 3))
        self.assertEqual(selection_counts("b2"), (0, 2, 3, 3))
        self.assertEqual(selection_counts("b3"), (1, 2, 3, 3))
        self.assertEqual(
            [selection_score(candidate) for candidate in ("b1", "b2", "b3")],
            [Fraction(2), Fraction(1), Fraction(3, 2)],
        )
        self.assertNotIn(B, [case for suite in SUITES for case in suite])

    def test_scale_witness_and_wrong_round_trip(self) -> None:
        self.assertEqual(scale_output(asymmetric=False, repaired=False), (40, 80))
        self.assertEqual(scale_output(asymmetric=True, repaired=False), (40, 60))
        self.assertEqual(scale_output(asymmetric=True, repaired=True), (40, 80))
        self.assertEqual(wrong_round_trip(), (60, 200))

    def test_edit_matching_rejects_absence_and_ambiguity(self) -> None:
        old = "y = y0 + (wc / Wv) * v"
        new = "y = y0 + (hc / Hv) * v"
        self.assertEqual(edit_match_count(old, old, new), ("applied", new, 1))
        self.assertEqual(edit_match_count("other", old, new), ("absent", "other", 0))
        twice = old + "\n" + old
        self.assertEqual(edit_match_count(twice, old, new), ("ambiguous", twice, 2))

    def test_candidate_timing_condition_has_three_distinct_states(self) -> None:
        self.assertFalse(main_timing_eligible(False, False, True))
        self.assertFalse(main_timing_eligible(True, False, True))
        self.assertFalse(main_timing_eligible(True, True, False))
        self.assertTrue(main_timing_eligible(True, True, True))


if __name__ == "__main__":
    unittest.main()
