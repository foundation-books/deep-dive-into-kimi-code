"""Tests for the Chapter 22 dependency-free semantic companion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from dataclasses import replace
import json
import signal
import unittest

from . import reference
from . import judge_investigation as judge
from . import math_investigation as math_probe
from . import reward_bridge
from . import state_action_investigation as state_probe
from . import artifact_candidates
from .test_reward_error_update import (
    test_false_acceptance_reverses_wrong_response_change,
)


class RewardsVerifiersAgenticGrmReferenceTests(unittest.TestCase):
    def test_reward_error_update_derivative(self) -> None:
        test_false_acceptance_reverses_wrong_response_change()

    def test_artifact_gate_and_allowed_view(self) -> None:
        data = artifact_candidates.packet(16)
        good = artifact_candidates.artifacts(data, pooled=False)
        wrong = artifact_candidates.artifacts(data, pooled=True)
        self.assertTrue(artifact_candidates.independent_acceptance(data, good))
        self.assertFalse(artifact_candidates.independent_acceptance(data, wrong))
        self.assertFalse(artifact_candidates.independent_acceptance(data, {}))
        view = artifact_candidates.allowed_judge_view(
            "Compare equal class recalls", good
        )
        artifact_candidates.check_view(view)
        with self.assertRaises(ValueError):
            artifact_candidates.check_view({**view, "reward": 1})
        with self.assertRaises(ValueError):
            artifact_candidates.check_view(
                {"task": "protected_truth=1", "artifacts": good}
            )

    def test_verdict_criteria_and_candidate_mapping(self) -> None:
        row = {"delivery": "pass", "metric": "fail", "reason": "inspected"}
        verdict = {"u17": row, "u39": row}
        self.assertTrue(artifact_candidates.valid_verdict(verdict, ("u39", "u17")))
        self.assertFalse(artifact_candidates.valid_verdict(verdict, ("u17", "stale")))
        self.assertFalse(
            artifact_candidates.valid_verdict({"u17": row}, ("u17", "u39"))
        )
        self.assertFalse(
            artifact_candidates.valid_verdict(
                {"u17": {**row, "metric": "pass|fail|abstain"}}, ("u17",)
            )
        )

    def test_dispositions_and_gate_precedence(self) -> None:
        passed = reference.VerifierResult("pass", "ok", "v1")
        failed = reference.VerifierResult("fail", "false", "v1")
        invalid = reference.VerifierResult("invalid", "bad", "v1")
        unresolved = reference.VerifierResult("indeterminate", "timeout", "v1")
        self.assertEqual(reference.hard_gate((passed, passed)), 1)
        self.assertEqual(reference.hard_gate((passed, failed)), 0)
        self.assertEqual(reference.hard_gate((unresolved, invalid)), 0)
        self.assertIsNone(reference.hard_gate((passed, unresolved)))
        with self.assertRaises(ValueError):
            reference.VerifierResult("zero", None, "v1").validate()

    def test_evidence_and_domain_verifiers(self) -> None:
        red = reference.center_record("A", "red", (50, 40))
        blue = reference.center_record("A", "blue", (50, 100))
        self.assertTrue(red.validate())
        self.assertEqual(
            reference.relation_verifier("below", red, blue).disposition,
            "pass",
        )
        self.assertEqual(
            reference.relation_verifier("above", red, blue).disposition,
            "fail",
        )
        self.assertEqual(
            reference.relation_verifier("below", None, None).disposition,
            "invalid",
        )
        self.assertEqual(reference.exact_verifier(37 * 14, 518).disposition, "pass")
        self.assertEqual(reference.exact_verifier("518", 518).disposition, "invalid")
        level = reference.center_record("equal", "blue", (50, 40))
        self.assertEqual(
            reference.relation_verifier("level", red, level).disposition, "pass"
        )
        self.assertEqual(
            reference.relation_verifier("above", red, level).disposition, "fail"
        )
        self.assertEqual(
            reference.relation_verifier("unclear", red, blue).disposition, "invalid"
        )
        witness = reference.relation_verifier("above", red, blue).witness
        self.assertEqual(witness["claim"], "above")
        self.assertEqual(witness["expected"], "below")

    def test_tolerance_units_and_visual_iou(self) -> None:
        result = reference.tolerance_verifier(
            0.3000004,
            0.3,
            absolute=0.000001,
            relative=0,
            unit="ratio",
            expected_unit="ratio",
        )
        self.assertEqual(result.disposition, "pass")
        at_boundary = reference.tolerance_verifier(
            101,
            100,
            absolute=1,
            relative=0,
            unit="m",
            expected_unit="m",
        )
        self.assertEqual(at_boundary.disposition, "pass")
        wrong_units = reference.tolerance_verifier(
            1,
            1,
            absolute=0,
            relative=0,
            unit="m",
            expected_unit="s",
        )
        self.assertEqual(wrong_units.disposition, "invalid")
        self.assertEqual(
            reference.mask_iou({(0, 0), (0, 1)}, {(0, 1), (1, 1)}),
            Fraction(1, 3),
        )
        with self.assertRaises(ValueError):
            reference.mask_iou(set(), set())

    def test_hard_first_composition_and_missingness(self) -> None:
        ledgers = reference.studio_ledgers()
        self.assertEqual(reference.compose_reward(ledgers["A"]), Fraction(59, 50))
        self.assertEqual(reference.compose_reward(ledgers["B"]), Fraction(89, 100))
        self.assertEqual(reference.compose_reward(ledgers["C"]), Fraction(-1))
        self.assertEqual(reference.compose_reward(ledgers["D"]), Fraction(0))
        changed_invalid = reference.RewardLedger(
            0,
            reference.inapplicable(),
            reference.inapplicable(),
            reference.inapplicable(),
            reference.inapplicable(),
            reference.inapplicable(),
            reference.inapplicable(),
            "v2",
        )
        self.assertEqual(reference.compose_reward(changed_invalid), Fraction(-1))
        unresolved = reference.RewardLedger(
            None,
            reference.observed(1),
            reference.observed(0),
            reference.observed(100),
            reference.observed(0),
            reference.observed(0),
            reference.observed(0),
            "v1",
        )
        self.assertIsNone(reference.compose_reward(unresolved))

    def test_public_hidden_rates_and_validation(self) -> None:
        public, hidden, gap = reference.public_hidden_gap(4, 4, 2, 4)
        expected = (Fraction(1), Fraction(1, 2), Fraction(1, 2))
        self.assertEqual((public, hidden, gap), expected)
        with self.assertRaises(ValueError):
            reference.public_hidden_gap(1, 0, 1, 1)
        with self.assertRaises(ValueError):
            reference.public_hidden_gap(2, 1, 1, 1)

    def test_tournament_cycle_ties_and_abstention(self) -> None:
        records = (
            reference.ComparisonRecord("X", "Y", ("X", "Y"), "X", "r1", "j1"),
            reference.ComparisonRecord("Y", "Z", ("Y", "Z"), "Y", "r1", "j1"),
            reference.ComparisonRecord("Z", "X", ("Z", "X"), "Z", "r1", "j1"),
            reference.ComparisonRecord("X", "Y", ("Y", "X"), "tie", "r1", "j1"),
            reference.ComparisonRecord("X", "Z", ("X", "Z"), "abstain", "r1", "j1"),
        )
        self.assertEqual(reference.win_fraction(records[:3], "X"), Fraction(1, 2))
        self.assertEqual(reference.win_fraction(records, "X"), Fraction(1, 2))
        abstain = (
            reference.ComparisonRecord("A", "B", ("A", "B"), "abstain", "r1", "j1"),
        )
        self.assertIsNone(reference.win_fraction(abstain, "A"))

    def test_rubric_scorepad_and_verbosity_boundary(self) -> None:
        rubric = reference.Rubric(
            ("grounding",),
            {"grounding": (Fraction(0), Fraction(1))},
            ("fabrication",),
            {"grounding": ("red", "blue")},
            "chapter-created",
            "r1",
        )
        self.assertTrue(rubric.validate())
        cell = reference.ScorepadCell(
            "A",
            "grounding",
            Fraction(1),
            ("red", "blue"),
            "observed",
        )
        self.assertTrue(cell.validate(rubric))
        with self.assertRaises(ValueError):
            reference.ScorepadCell(
                "A",
                "grounding",
                Fraction(1),
                ("red",),
                "observed",
            ).validate(rubric)
        self.assertFalse(reference.verbosity_event(15, 8, Fraction(2)))
        self.assertFalse(reference.verbosity_event(16, 8, Fraction(2)))
        self.assertTrue(reference.verbosity_event(18, 8, Fraction(2)))

    def test_complete_handoffs_and_protocol_registry(self) -> None:
        handoffs = reference.studio_handoffs()
        self.assertEqual(set(handoffs), {"A", "B", "C", "D"})
        self.assertTrue(all(handoff.validate() for handoff in handoffs.values()))
        checks = reference.all_checks()
        self.assertEqual(len(checks), 7)
        self.assertTrue(all(record["status"] == "pass" for record in checks.values()))
        result = checks["RESULT_CH20_STUDIO_HANDOFF_ORACLE"]
        self.assertEqual(result["measured_protocol_outcomes"], "pending")

    def test_comparison_producer_and_timeout(self) -> None:
        def skipped_judge() -> str:
            self.fail("automatic or unresolved-policy branch called the judge")

        for left, right, lengths in (("A", "B", (8, 18)), ("B", "A", (18, 8))):
            result = reference.compare_candidates(left, right, *lengths, skipped_judge)
            self.assertEqual(result.outcome, "A")
            self.assertEqual(result.decision_source, "automatic_verbosity")
            self.assertIsNone(result.judge_version)
        unresolved = reference.compare_candidates("A", "B", 18, 18, skipped_judge)
        self.assertIsNone(unresolved.outcome)
        self.assertEqual(unresolved.decision_source, "unresolved_policy")
        self.assertIsNone(reference.win_fraction((unresolved,), "A"))
        calls = []

        def control_judge() -> str:
            calls.append("called")
            return "A_ctl"

        control = reference.compare_candidates("A_ctl", "B_ctl", 8, 16, control_judge)
        self.assertEqual(calls, ["called"])
        self.assertEqual(control.outcome, "A_ctl")
        self.assertEqual(control.decision_source, "learned_comparison")

        def timeout() -> str:
            raise TimeoutError

        timed_out = reference.compare_candidates("A_ctl", "B_ctl", 8, 16, timeout)
        self.assertIsNone(timed_out.outcome)
        self.assertEqual(timed_out.decision_source, "unresolved_execution")
        self.assertIsNone(reference.win_fraction((timed_out,), "A_ctl"))

    def test_canonical_statuses_and_fixed_group_admission(self) -> None:
        records = reference.studio_handoffs()
        c = records["C"]
        self.assertEqual(c.evidence_disposition, "invalid")
        self.assertEqual(c.ledger.task.status, "not_evaluated")
        self.assertEqual(c.ledger.grm.status, "not_evaluated")
        self.assertEqual(records["D"].ledger.grm.status, "inapplicable")
        self.assertEqual(reference.missing().status, "unresolved")
        group = tuple(records.values())
        mean, coefficients = reference.centered_studio_group(group)
        self.assertEqual(mean, Fraction(107, 400))
        self.assertEqual(
            coefficients, tuple(Fraction(n, 400) for n in (365, 249, -507, -107))
        )
        self.assertEqual(sum(coefficients), 0)
        self.assertEqual(c.scalar_purpose, "blocked_diagnostic_and_training_outcome")
        for invalid_c in (
            replace(c, sampling_provenance_valid=False),
            replace(c, infrastructure_resolved=False),
            replace(c, admission_policy="reject-only-v1"),
            replace(c, candidate_evidence_violation=False),
        ):
            self.assertFalse(invalid_c.training_eligible)
            self.assertEqual(invalid_c.scalar_purpose, "diagnostic_only")
            with self.assertRaises(ValueError):
                reference.centered_studio_group(
                    (group[0], group[1], invalid_c, group[3])
                )
        with self.assertRaises(ValueError):
            reference.centered_studio_group((group[0], group[1], group[3]))
        with self.assertRaises(ValueError):
            reference.centered_studio_group((group[0], group[1], group[1], group[3]))

    def test_printed_exercise_fixture_arithmetic(self) -> None:
        self.assertEqual(
            reference.public_hidden_gap(3, 4, 2, 4),
            (Fraction(3, 4), Fraction(1, 2), Fraction(1, 4)),
        )
        self.assertEqual(
            reference.public_hidden_gap(2, 3, 1, 3),
            (Fraction(2, 3), Fraction(1, 3), Fraction(1, 3)),
        )
        tournament = reference.all_checks()["RESULT_CH20_TOURNAMENT_ORACLE"]
        self.assertEqual(tournament["studio_A_win_fraction"], "5/6")

    def test_math_extraction_direction_and_status_adapter(self) -> None:
        matches = (("unparsed high-priority match", None), ("3/3", Fraction(1)))
        self.assertEqual(
            math_probe.select_answer(matches, mode="first_match"), matches[0][0]
        )
        self.assertEqual(
            math_probe.select_answer(matches, mode="any_match"), Fraction(1)
        )
        compare = math_probe.compare_constructed_answers
        self.assertTrue(compare(("number", Fraction(1)), ("number", Fraction(3, 3))))
        relation = ("relation", (Fraction(1), Fraction(2)))
        interval = ("interval", (Fraction(1), Fraction(2)))
        self.assertTrue(compare(relation, interval))
        self.assertFalse(compare(interval, relation))
        self.assertTrue(compare(interval, relation, allow_set_relation_comp=True))
        self.assertTrue(compare(("string", "x"), ("string", "x")))
        adapter = math_probe.adapt_math_result
        self.assertEqual(
            adapter(False, reference_valid=True, candidate_valid=True).disposition,
            "fail",
        )
        self.assertEqual(
            adapter(None, reference_valid=True, candidate_valid=False).witness[
                "origin"
            ],
            "candidate",
        )
        self.assertEqual(
            adapter(None, reference_valid=False, candidate_valid=True).witness[
                "origin"
            ],
            "reference",
        )
        self.assertEqual(
            adapter(
                None, reference_valid=True, candidate_valid=True, error="timeout"
            ).disposition,
            "indeterminate",
        )

    def test_state_action_target_and_composition_control_flow(self) -> None:
        expected = {"S1": (100, 1, 1), "S2": (100, 1, 0), "S3": (80, 0, 1)}
        for name, actions in state_probe.STATE_TRAJECTORIES.items():
            final_y, db, action = expected[name]
            self.assertEqual(state_probe.final_blue_y(actions), final_y)
            self.assertEqual(
                state_probe.component_results(actions), {"DB": db, "ACTION": action}
            )
        self.assertEqual(
            state_probe.selected_product({"DB": 0, "ACTION": 1}, ()),
            (Fraction(1), ()),
        )
        for basis, scores in {
            ("DB",): (1, 1, 0),
            ("ACTION",): (1, 0, 1),
            ("DB", "ACTION"): (1, 0, 0),
        }.items():
            actual = tuple(
                state_probe.selected_product(
                    state_probe.component_results(actions), basis
                )[0]
                for actions in state_probe.STATE_TRAJECTORIES.values()
            )
            self.assertEqual(actual, scores)
        calls: list[int] = []

        def soft_judge() -> Fraction:
            calls.append(1)
            return Fraction(9, 10)

        self.assertEqual(state_probe.hard_first_with_judge(0, soft_judge), -1)
        self.assertIsNone(state_probe.hard_first_with_judge(None, soft_judge))
        self.assertEqual(calls, [])
        self.assertEqual(
            state_probe.hard_first_with_judge(1, soft_judge), Fraction(9, 10)
        )
        self.assertEqual(calls, [1])

    def test_ordered_rubric_and_verdict_validation(self) -> None:
        criteria = (
            judge.Criterion(
                "clarity",
                ("very poor", "poor", "adequate", "good", "excellent"),
                Fraction(3),
            ),
            judge.Criterion("economy", ("unfocused", "concise"), Fraction(1)),
        )
        u = (
            judge.Verdict("clarity", "supplied", "excellent"),
            judge.Verdict("economy", "supplied", "concise"),
        )
        v = (
            judge.Verdict("clarity", "supplied", "good"),
            judge.Verdict("economy", "supplied", "concise"),
        )
        self.assertEqual(judge.score_verdicts(criteria, u)[1], Fraction(1))
        self.assertEqual(judge.score_verdicts(criteria, v)[1], Fraction(13, 16))
        reversed_clarity = replace(
            criteria[0], choices=tuple(reversed(criteria[0].choices))
        )
        self.assertEqual(
            judge.score_verdicts((reversed_clarity, criteria[1]), v)[1], Fraction(7, 16)
        )
        zero_weight = replace(criteria[1], weight=Fraction(0))
        self.assertEqual(
            judge.score_verdicts((criteria[0], zero_weight), v)[1], Fraction(3, 4)
        )
        for bad_criteria, bad_verdicts in (
            (criteria, v[:1]),
            (criteria, v + (v[0],)),
            (criteria, (judge.Verdict("clarity", "supplied", "illegal"), v[1])),
            ((replace(criteria[0], weight=Fraction(0)), zero_weight), v),
        ):
            with self.assertRaises(ValueError):
                judge.score_verdicts(bad_criteria, bad_verdicts)

    def test_judge_workspace_verdict_ownership_and_evidence_links(self) -> None:
        criterion = (judge.Criterion("solved", ("no", "yes"), Fraction(1)),)
        solver_files = {"scratch.txt": "unpublished", "verdict.json": "stale"}
        published = {"artifact.txt": "delivered"}
        shared = judge.prepare_workspace(
            solver_files, published, shared=True, trace="trace"
        )
        isolated = judge.prepare_workspace(
            solver_files, published, shared=False, trace="trace"
        )
        self.assertIn("scratch.txt", shared)
        self.assertNotIn("scratch.txt", isolated)
        self.assertIn("artifact.txt", isolated)
        self.assertNotIn("verdict.json", shared)
        with self.assertRaises(ValueError):
            judge.read_verdict_file(shared, criterion)
        isolated["verdict.json"] = json.dumps(
            {"verdicts": [{"name": "solved", "reason": "event e9", "verdict": "yes"}]}
        )
        self.assertEqual(judge.read_verdict_file(isolated, criterion), Fraction(1))
        self.assertFalse(judge.evidence_links_valid(("e9",), ("e1", "e2")))
        self.assertTrue(judge.evidence_links_valid(("e2",), ("e1", "e2")))
        isolated["verdict.json"] = json.dumps({"verdicts": []})
        with self.assertRaises(ValueError):
            judge.read_verdict_file(isolated, criterion)

    def test_numeric_boundary_keeps_original_slots_and_status(self) -> None:
        solution, extras = reward_bridge.callback_view(
            ("earlier tool text", "final answer"),
            extra_info={"case": "C"},
            tool_fields={"witness": "explicit transport"},
        )
        self.assertEqual(solution, "final answer")
        self.assertEqual(extras["witness"], "explicit transport")
        handoffs = reference.studio_handoffs()
        admitted = reward_bridge.route_handoff(handoffs["C"])
        self.assertEqual(admitted.numeric_result["reward_score"], -1.0)
        self.assertEqual(admitted.numeric_result["reward_extra_info"]["slot_id"], 3)
        self.assertEqual(
            admitted.numeric_result["reward_extra_info"]["evidence_disposition"],
            "invalid",
        )
        deferred = reward_bridge.route_handoff(
            replace(handoffs["C"], infrastructure_resolved=False)
        )
        self.assertIsNone(deferred.numeric_result)
        self.assertEqual(deferred.group_action, "defer_original_slot")
        quarantined = reward_bridge.route_handoff(
            replace(handoffs["C"], sampling_provenance_valid=False)
        )
        self.assertIsNone(quarantined.numeric_result)
        self.assertEqual(quarantined.group_action, "quarantine_group")

    def test_bounded_process_worker_and_thread_constraint(self) -> None:
        worker = reward_bridge.bounded_rational_check
        self.assertEqual(worker("3/3", "1", timeout_seconds=2).disposition, "pass")
        self.assertEqual(worker("2", "1", timeout_seconds=2).disposition, "fail")
        self.assertEqual(
            worker("1", "bad", timeout_seconds=2).witness["origin"], "reference"
        )
        self.assertEqual(
            worker("bad", "1", timeout_seconds=2).witness["origin"], "candidate"
        )
        self.assertEqual(
            worker("1", "1", timeout_seconds=0.05, delay_seconds=0.3).disposition,
            "indeterminate",
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            attempt = pool.submit(signal.signal, signal.SIGALRM, signal.SIG_DFL)
            with self.assertRaises(ValueError):
                attempt.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
