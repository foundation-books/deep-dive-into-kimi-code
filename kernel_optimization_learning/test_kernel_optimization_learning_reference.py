"""Numerical semantics, discriminating padding probes, and one policy step."""

import math
import unittest
from dataclasses import replace
from fractions import Fraction

from kernel_optimization_learning.learning_example import (
    SCORES_D,
    SCORES_E,
    discounted_returns,
    first_decision_policy_step,
    grouped_turn_coefficients,
    teaching_score,
    whole_operation_speedup,
)
from kernel_optimization_learning.reference import (
    combined_padding_defect_rows,
    fused_padded_rows,
    normalized_gated_rows,
    padded_divisor_defect_rows,
    partitioned_reduction_rows,
    unmasked_padding_defect_rows,
)
from kernel_optimization_learning.selection import (
    ASSIGNED_COSTS,
    CANDIDATES,
    HOLDOUT,
    QUALIFICATION,
    assigned_scores,
    final_check,
    passes_cases,
    select_fixture,
)


class OperatorSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = ((3.0, 4.0, 0.0),)
        self.gates = ((0.0, math.log(3.0), -math.log(3.0)),)
        self.gain = (1.0, 2.0, 1.0)
        self.epsilon = 1.0

    def test_tiny_exact_row_and_independent_token_head_row(self) -> None:
        first = normalized_gated_rows(self.values, self.gates, self.gain, self.epsilon)[
            0
        ]
        denominator = math.sqrt(28.0 / 3.0)
        expected = (1.5 / denominator, 6.0 / denominator, 0.0)
        for actual, target in zip(first, expected, strict=True):
            self.assertAlmostEqual(actual, target, places=14)
        self.assertAlmostEqual(first[0], 0.4909902530, places=10)
        self.assertAlmostEqual(first[1], 1.9639610121, places=10)

        rows = normalized_gated_rows(
            (*self.values, (0.0, 0.0, 6.0)),
            (*self.gates, (0.0, 0.0, 0.0)),
            self.gain,
            self.epsilon,
        )
        self.assertEqual(rows[0], first)
        self.assertAlmostEqual(rows[1][2], 3.0 / math.sqrt(13.0))

    def test_zero_padding_and_wrong_divisor_have_distinct_effects(self) -> None:
        expected = normalized_gated_rows(
            self.values, self.gates, self.gain, self.epsilon
        )
        repaired = fused_padded_rows(
            self.values, self.gates, self.gain, self.epsilon, padded_width=4
        )
        wrong_divisor = padded_divisor_defect_rows(
            self.values, self.gates, self.gain, self.epsilon, padded_width=4
        )
        self.assertEqual(repaired, expected)
        self.assertNotEqual(wrong_divisor, expected)
        self.assertAlmostEqual(wrong_divisor[0][0], 1.5 / math.sqrt(29.0 / 4.0))

        # A power-of-two width gives this defect no extra lane to expose it.
        four = ((3.0, 4.0, 0.0, 0.0),)
        four_gates = ((0.0, math.log(3.0), -math.log(3.0), 0.0),)
        four_gain = (1.0, 2.0, 1.0, 1.0)
        self.assertEqual(
            normalized_gated_rows(four, four_gates, four_gain, 1.0),
            padded_divisor_defect_rows(
                four, four_gates, four_gain, 1.0, padded_width=4
            ),
        )

    def test_nonzero_padding_separates_unmasked_sum_from_divisor_bug(self) -> None:
        expected = normalized_gated_rows(
            self.values, self.gates, self.gain, self.epsilon
        )
        masked_probe = unmasked_padding_defect_rows(
            self.values,
            self.gates,
            self.gain,
            self.epsilon,
            padded_width=4,
            padding_value=0.0,
        )
        unmasked_probe = unmasked_padding_defect_rows(
            self.values,
            self.gates,
            self.gain,
            self.epsilon,
            padded_width=4,
            padding_value=1.0,
        )
        wrong_divisor = padded_divisor_defect_rows(
            self.values, self.gates, self.gain, self.epsilon, padded_width=4
        )
        self.assertEqual(masked_probe, expected)
        self.assertNotEqual(unmasked_probe, expected)
        self.assertNotEqual(unmasked_probe, wrong_divisor)
        self.assertAlmostEqual(unmasked_probe[0][0], 1.5 / math.sqrt(29.0 / 3.0))

    def test_four_denominators_with_sentinel_five(self) -> None:
        correct = normalized_gated_rows(self.values, self.gates, self.gain, 1.0)
        wrong_divisor = padded_divisor_defect_rows(
            self.values, self.gates, self.gain, 1.0, padded_width=4
        )
        unmasked = unmasked_padding_defect_rows(
            self.values,
            self.gates,
            self.gain,
            1.0,
            padded_width=4,
            padding_value=5.0,
        )
        both = combined_padding_defect_rows(
            self.values,
            self.gates,
            self.gain,
            1.0,
            padded_width=4,
            padding_value=5.0,
        )
        for actual, denominator in zip(
            (correct, wrong_divisor, unmasked, both),
            (
                math.sqrt(28.0 / 3.0),
                math.sqrt(29.0 / 4.0),
                math.sqrt(53.0 / 3.0),
                math.sqrt(54.0 / 4.0),
            ),
            strict=True,
        ):
            self.assertAlmostEqual(actual[0][0], 1.5 / denominator)

    def test_partitioned_reduction_combines_before_dividing_and_gating(self) -> None:
        values = ((3.0, -4.0, 1.0, 0.0, 2.0),)
        gates = ((0.0, 0.5, -0.5, 1.0, -1.0),)
        gain = (1.0, 2.0, 1.0, 0.5, 1.5)
        expected = normalized_gated_rows(values, gates, gain, 1e-5)
        for partitions in (2, 4):
            actual = partitioned_reduction_rows(
                values, gates, gain, 1e-5, partitions=partitions
            )
            for got, want in zip(actual[0], expected[0], strict=True):
                self.assertAlmostEqual(got, want, places=14)
        with self.assertRaises(ValueError):
            partitioned_reduction_rows(values, gates, gain, 1e-5, partitions=6)

    def test_power_of_two_and_tail_width_probe(self) -> None:
        for width, padded_width in ((64, 64), (65, 128)):
            values = ((1.0,) * width,)
            gates = ((0.0,) * width,)
            gain = (1.0,) * width
            reference = normalized_gated_rows(values, gates, gain, 1.0)
            c1 = padded_divisor_defect_rows(
                values, gates, gain, 1.0, padded_width=padded_width
            )
            c2 = fused_padded_rows(values, gates, gain, 1.0, padded_width=padded_width)
            self.assertEqual(c2, reference)
            if width == padded_width:
                self.assertEqual(c1, reference)
            else:
                self.assertNotEqual(c1, reference)

    def test_rejects_changed_operator_domain(self) -> None:
        with self.assertRaises(ValueError):
            normalized_gated_rows(self.values, self.gates, self.gain, 0.0)
        with self.assertRaises(ValueError):
            normalized_gated_rows(self.values, self.gates, (1.0, 2.0), 1.0)
        with self.assertRaises(ValueError):
            normalized_gated_rows(self.values, ((0.0, 0.0),), self.gain, 1.0)
        with self.assertRaises(ValueError):
            fused_padded_rows(self.values, self.gates, self.gain, 1.0, padded_width=2)
        with self.assertRaises(ValueError):
            normalized_gated_rows(self.values, ((math.inf, 0.0, 0.0),), self.gain, 1.0)


class LearningArithmeticTests(unittest.TestCase):
    def test_score_keeps_rejection_and_missing_measurement_distinct(self) -> None:
        self.assertEqual(teaching_score("candidate_rejected"), 0.0)
        self.assertIsNone(teaching_score("unresolved"))
        self.assertIsNone(teaching_score("passed", baseline=1.0))
        self.assertIsNone(teaching_score("passed", runtime=math.nan, baseline=1.0))
        self.assertIsNone(teaching_score("passed", runtime=1.0, baseline=math.inf))
        self.assertIsNone(teaching_score("passed", runtime=0.0, baseline=1.0))
        self.assertEqual(teaching_score("passed", runtime=1.0, baseline=1.0), 0.4)
        self.assertEqual(teaching_score("passed", runtime=0.4, baseline=1.0), 1.0)

    def test_turn_grouping_and_loss_mask_change_actor_coefficients(self) -> None:
        # Episode A has scores (1,4); B has (3,2). Their turn returns stay
        # attached to the episode even when both share the same task prompt.
        returns = (
            *discounted_returns((1.0, 4.0), 1.0),
            *discounted_returns((3.0, 2.0), 1.0),
        )
        self.assertEqual(returns, (5.0, 4.0, 5.0, 2.0))
        by_turn = (("task", 1), ("task", 2), ("task", 1), ("task", 2))
        all_actions = (True, True, True, True)
        coefficients = grouped_turn_coefficients(returns, by_turn, all_actions)
        self.assertEqual(coefficients, (0.0, 2.0, 0.0, -2.0))

        # Grouping only by prompt incorrectly mixes returns from both turns.
        by_prompt = (("task", None),) * 4
        mixed = grouped_turn_coefficients(returns, by_prompt, all_actions)
        self.assertAlmostEqual(mixed[0], 4.0 / 3.0)
        self.assertEqual(mixed[1], 0.0)
        self.assertAlmostEqual(mixed[2], 4.0 / 3.0)
        self.assertAlmostEqual(mixed[3], -8.0 / 3.0)
        self.assertNotEqual(mixed[1], coefficients[1])

        # Excluding A's second assistant action removes its loss coefficient;
        # it remains in the complete turn group used for B's baseline.
        masked = grouped_turn_coefficients(returns, by_turn, (True, False, True, True))
        self.assertEqual(masked, (0.0, 0.0, 0.0, -2.0))
        self.assertNotEqual(masked[1], coefficients[1])

    def test_turn_grouping_rejects_incomplete_groups(self) -> None:
        with self.assertRaises(ValueError):
            grouped_turn_coefficients((5.0,), (("task", 1),), (True,))
        with self.assertRaises(ValueError):
            grouped_turn_coefficients((5.0, math.inf), (("task", 1),) * 2, (True,) * 2)
        with self.assertRaises(ValueError):
            grouped_turn_coefficients((5.0, 5.0), (("task", 1),) * 2, (True,))

    def test_return_sequences_and_first_decision_preference(self) -> None:
        cases = (
            (0.5, (0.70, 0.60, 0.40), (0.45, 0.90, 1.00)),
            (1.0, (1.20, 0.80, 0.40), (1.40, 1.40, 1.00)),
        )
        for gamma, expected_d, expected_e in cases:
            for actual, expected in zip(
                discounted_returns(SCORES_D, gamma), expected_d, strict=True
            ):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(
                discounted_returns(SCORES_E, gamma), expected_e, strict=True
            ):
                self.assertAlmostEqual(actual, expected)
        self.assertGreater(
            discounted_returns(SCORES_D, 0.5)[0],
            discounted_returns(SCORES_E, 0.5)[0],
        )
        self.assertLess(
            discounted_returns(SCORES_D, 1.0)[0],
            discounted_returns(SCORES_E, 1.0)[0],
        )

    def test_same_first_decision_objective_updates_policy(self) -> None:
        step = first_decision_policy_step(
            SCORES_D, SCORES_E, gamma=1.0, logit=0.0, learning_rate=0.5
        )
        self.assertAlmostEqual(step.return_d, 1.2)
        self.assertAlmostEqual(step.return_e, 1.4)
        self.assertAlmostEqual(step.baseline, 1.3)
        self.assertAlmostEqual(step.advantage_d, -0.1)
        self.assertAlmostEqual(step.advantage_e, 0.1)
        self.assertEqual(step.probability_before, 0.5)
        self.assertAlmostEqual(step.gradient, 0.05)
        self.assertAlmostEqual(step.logit_after, 0.025)
        self.assertAlmostEqual(step.probability_after, 0.5062496745, places=10)
        self.assertGreater(step.expected_return_after, step.expected_return_before)

        # Enumerating both score-function terms reproduces the analytic step.
        p = step.probability_before
        sampled_terms = (1.0 - p) * step.advantage_d * (-p) + p * step.advantage_e * (
            1.0 - p
        )
        self.assertAlmostEqual(sampled_terms, step.gradient)
        discounted = first_decision_policy_step(
            SCORES_D, SCORES_E, gamma=0.5, logit=0.0, learning_rate=0.5
        )
        self.assertAlmostEqual(discounted.gradient, -0.0625)
        self.assertLess(discounted.probability_after, 0.5)

    def test_complete_operation_cost_model_and_input_validation(self) -> None:
        self.assertAlmostEqual(whole_operation_speedup(0.9, 10.0), 1.0 / 0.91)
        for scores, gamma in (((), 1.0), ((0.4,), -0.1), ((math.inf,), 1.0)):
            with self.assertRaises(ValueError):
                discounted_returns(scores, gamma)
        with self.assertRaises(ValueError):
            first_decision_policy_step(
                SCORES_D, SCORES_E, gamma=1.0, logit=0.0, learning_rate=0.0
            )


class CandidateSelectionTests(unittest.TestCase):
    def test_scores_require_four_positive_fraction_costs(self) -> None:
        for key in ("C0", "C3"):
            for index in (0, 1):
                for invalid in (None, Fraction(0), Fraction(-1)):
                    with self.subTest(key=key, index=index, invalid=invalid):
                        costs = dict(ASSIGNED_COSTS)
                        pair = list(costs[key])
                        pair[index] = invalid
                        costs[key] = (pair[0], pair[1])
                        self.assertEqual(
                            assigned_scores("C3", True, costs), (None, None, None)
                        )

    def test_tail_gate_and_two_workload_selection(self) -> None:
        self.assertEqual((QUALIFICATION[0].rows, QUALIFICATION[0].width), (128, 1024))
        self.assertEqual((QUALIFICATION[1].rows, QUALIFICATION[1].width), (1, 4096))
        self.assertFalse(passes_cases(CANDIDATES[1], QUALIFICATION))
        self.assertTrue(all(passes_cases(c, QUALIFICATION) for c in CANDIDATES[2:]))
        choice = select_fixture()
        self.assertEqual(choice.last_candidate, "C4")
        self.assertEqual(choice.selected_candidate, "C3")
        self.assertEqual(
            [r.incumbent for r in choice.records], ["C0", "C0", "C0", "C3", "C3"]
        )
        self.assertEqual(choice.records[1].mean_score, None)
        self.assertEqual(choice.records[2].mean_score, Fraction(2, 5))
        self.assertEqual(choice.records[3].mean_score, Fraction(3, 4))
        self.assertEqual(choice.records[4].score_a, Fraction(4, 7))
        self.assertEqual(choice.records[4].score_b, Fraction(2, 3))
        self.assertEqual(choice.records[4].mean_score, Fraction(13, 21))
        self.assertEqual(final_check(choice), HOLDOUT)

    def test_rejection_tie_missing_input_and_final_snapshot_gate(self) -> None:
        costs = dict(ASSIGNED_COSTS)
        costs["C1"] = (Fraction(1, 100), Fraction(1, 100))
        self.assertEqual(select_fixture(costs=costs).selected_candidate, "C3")
        costs["C4"] = (None, Fraction(3, 5))
        self.assertEqual(assigned_scores("C4", True, costs), (None, None, None))
        costs["C0"] = (None, Fraction(1))
        with self.assertRaisesRegex(ValueError, "baseline"):
            select_fixture(costs=costs)

        choice = select_fixture()
        changed = tuple(
            replace(c, partitions=4) if c.name == "C3" else c for c in CANDIDATES
        )
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            final_check(choice, candidates=changed)
        with self.assertRaisesRegex(ValueError, "failed final validation"):
            final_check(choice, checker=lambda candidate, cases: False)


if __name__ == "__main__":
    unittest.main()
