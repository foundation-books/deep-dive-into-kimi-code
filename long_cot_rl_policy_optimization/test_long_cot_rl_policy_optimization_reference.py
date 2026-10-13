"""Deterministic tests for the Chapter 19 companion."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import unittest

from . import reference
from .test_implementation_insights import ImplementationInsightsTests


class ImplementationInsightsManifestTests(ImplementationInsightsTests):
    """Include the new focused checks in the chapter's manifest smoke gate."""


class LongCotRlPolicyOptimizationReferenceTests(unittest.TestCase):
    def test_group_contract_and_exact_centering(self) -> None:
        group = reference.studio_group()
        self.assertEqual(
            reference.validate_group(group),
            {"task_id": "studio-layout-001", "K": 4, "N": 48, "update_lag": 3},
        )
        mean, advantages = reference.group_advantages(group.outcomes)
        self.assertEqual(mean, Fraction(3, 4))
        self.assertEqual(
            advantages,
            (
                Fraction(1, 4),
                Fraction(1, 4),
                Fraction(-3, 4),
                Fraction(1, 4),
            ),
        )
        translated_mean, translated = reference.group_advantages((8, 8, 7, 8))
        self.assertEqual(translated_mean, Fraction(31, 4))
        self.assertEqual(translated, advantages)
        self.assertEqual(
            reference.group_advantages((1, 1, 1, 1))[1],
            (Fraction(0),) * 4,
        )

    def test_invalid_group_records_are_rejected(self) -> None:
        group = reference.studio_group()
        invalid = (
            replace(group, task_ids=("a", "a", "b", "a")),
            replace(group, behavior_version=None),
            replace(group, promised_size=5),
            replace(group, completed=(True, True, True, False)),
            replace(group, lengths=(8, 12, 0, 20)),
        )
        for case in invalid:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    reference.validate_group(case)

    def test_exponential_tilt_and_limits(self) -> None:
        result = reference.exponential_tilt((0.6, 0.4), (1.0, 0.0), 0.5)
        expected = 0.6 * math.exp(2) / (0.6 * math.exp(2) + 0.4)
        self.assertTrue(math.isclose(result[0], expected, rel_tol=1e-12))
        self.assertTrue(math.isclose(result[1], 1 - expected, rel_tol=1e-12))
        near_reference = reference.exponential_tilt((0.6, 0.4), (1.0, 0.0), 1e8)
        self.assertTrue(math.isclose(near_reference[0], 0.6, rel_tol=1e-8))
        near_maximum = reference.exponential_tilt((0.6, 0.4), (1.0, 0.0), 0.01)
        self.assertGreater(near_maximum[0], 1 - 1e-12)
        self.assertEqual(
            reference.exponential_tilt((1.0, 0.0), (0.0, 10.0), 0.5),
            (1.0, 0.0),
        )
        with self.assertRaises(ValueError):
            reference.exponential_tilt((0.5, 0.4), (1.0, 0.0), 0.5)
        with self.assertRaises(ValueError):
            reference.exponential_tilt((0.6, 0.4), (1.0, 0.0), 0.0)

    def test_ratio_factorization_and_cancellation(self) -> None:
        result = reference.ratio_factorization((0.30, 0.20), (0.25, 0.25))
        self.assertTrue(result["factorization_holds"])
        self.assertTrue(
            math.isclose(result["sequence_ratio_product"], 0.96, rel_tol=1e-12)
        )
        cancellation = reference.regularization_comparison((0.3, -0.3))
        self.assertTrue(
            math.isclose(cancellation["sequence_square"], 0.0, abs_tol=1e-15)
        )
        self.assertTrue(math.isclose(cancellation["local_square_sum"], 0.18))
        coherent = reference.regularization_comparison((0.3, 0.3))
        self.assertTrue(math.isclose(coherent["sequence_square"], 0.36))
        self.assertTrue(math.isclose(coherent["local_square_sum"], 0.18))

    def test_local_gradient_boundaries_and_penalties(self) -> None:
        flattened = [
            value for trajectory in reference.STUDIO_LOG_RATIOS for value in trajectory
        ]
        eligible = [
            reference.STUDIO_BOUNDS[0] <= value <= reference.STUDIO_BOUNDS[1]
            for value in flattened
        ]
        self.assertEqual(sum(eligible), 13)
        self.assertEqual(
            {
                value
                for value, is_eligible in zip(flattened, eligible, strict=True)
                if not is_eligible
            },
            {0.31, -0.24, 0.27},
        )
        self.assertTrue(
            math.isclose(
                reference.local_factor(0.31, -0.2, 0.2),
                math.exp(0.2),
                rel_tol=1e-12,
            )
        )
        self.assertGreater(
            reference.local_gradient_coefficient(0.08, 0.25, -0.2, 0.2, 0.1),
            0,
        )
        self.assertLess(
            reference.local_gradient_coefficient(0.07, -0.75, -0.2, 0.2, 0.1),
            0,
        )
        self.assertTrue(
            math.isclose(
                reference.local_gradient_coefficient(0.31, 0.25, -0.2, 0.2, 0.1),
                -0.062,
            )
        )
        self.assertLess(max(reference.finite_difference_local_checks().values()), 1e-6)
        with self.assertRaises(ValueError):
            reference.local_factor(0.0, 0.2, -0.2)

    def test_reward_and_restoration_for_both_signs_and_tails(self) -> None:
        for advantage in (0.25, -0.75):
            diagnostics = reference.local_diagnostics(
                advantage, (-0.3, -0.1, 0.0, 0.1, 0.3), -0.2, 0.2, 0.1
            )
            for row in diagnostics["records"]:
                value = row["log_ratio"]
                self.assertAlmostEqual(
                    row["restoring_gradient_coefficient"], -0.2 * value
                )
                if abs(value) > 0.2:
                    self.assertEqual(row["reward_gradient_coefficient"], 0)
                    self.assertGreater(row["penalty_loss"], 0)
                else:
                    self.assertAlmostEqual(
                        row["reward_gradient_coefficient"], advantage * math.exp(value)
                    )
            self.assertLess(
                max(reference.finite_difference_local_checks(advantage).values()), 1e-6
            )

    def test_exact_target_is_not_zero_surrogate_residual(self) -> None:
        oracle = reference.all_checks()["RESULT_CH17_MIRROR_DESCENT_ORACLE"]
        self.assertTrue(oracle["centered_exact_identity"])
        self.assertAlmostEqual(oracle["log_partition_constant"], 0.7877785588258128)
        self.assertEqual(oracle["empirical_reward_mean"], 0.5)
        for residual in oracle["surrogate_residuals_at_exact_target"]:
            self.assertAlmostEqual(residual, 0.2877785588258128)

    def test_complete_zero_tail_score_and_coefficient_ledger(self) -> None:
        trace = reference.complete_studio_score()
        self.assertEqual(trace["lengths"], (8, 12, 8, 20))
        self.assertEqual(trace["N"], 48)
        self.assertEqual(trace["U"], 45)
        self.assertEqual(
            [row["eligible_count"] for row in trace["paths"]], [8, 11, 7, 19]
        )
        for original, complete in zip(
            reference.STUDIO_LOG_RATIOS, trace["log_ratios"], strict=True
        ):
            self.assertEqual(complete[:4], original)
            self.assertTrue(all(value == 0 for value in complete[4:]))
        expected = {
            "forward_reward_sum": 4.391331259668419,
            "penalty_sum": 0.03672,
            "score": 0.09072106790975874,
            "reward_coefficient_sum": 4.39467794539682,
            "restoring_coefficient_sum": -0.112,
            "normalized_coefficient_sum": 0.08922245719576709,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(trace[key], value, places=12)
        self.assertNotEqual(
            trace["forward_reward_sum"], trace["reward_coefficient_sum"]
        )
        self.assertNotEqual(trace["score"], trace["normalized_coefficient_sum"])

    def test_tiny_policy_step_has_normalized_shared_parameter_gradient(self) -> None:
        logit = math.log(3 / 7)
        step = reference.tiny_policy_step(logit)
        self.assertAlmostEqual(step["blue"] + step["red"], 1)
        self.assertAlmostEqual(step["next_blue"] + step["next_red"], 1)
        self.assertAlmostEqual(step["score"], 0.06476660834727743)
        self.assertAlmostEqual(step["parameter_gradient"], 0.26516770487981467)
        self.assertAlmostEqual(step["next_blue"], 0.3227350650556533)
        epsilon = 1e-6
        numerical = (
            reference.tiny_policy_step(logit + epsilon)["score"]
            - reference.tiny_policy_step(logit - epsilon)["score"]
        ) / (2 * epsilon)
        self.assertAlmostEqual(numerical, step["parameter_gradient"], places=8)
        self.assertNotAlmostEqual(
            (step["blue_coefficient"] + step["red_coefficient"]) / 2,
            step["parameter_gradient"],
        )

    def test_matched_reductions_keep_the_same_eligible_numerators(self) -> None:
        ledger = reference.matched_reduction_fixture()
        self.assertEqual(ledger["active_counts"], (8, 6, 8, 10))
        self.assertEqual(ledger["active_tokens"], 32)
        self.assertEqual(ledger["total_tokens"], 48)
        self.assertEqual(
            ledger["total_token_coefficients"], (1 / 24, 1 / 32, -1 / 8, 5 / 96)
        )
        self.assertEqual(
            ledger["per_trajectory_coefficients"], (1 / 16, 1 / 32, -3 / 16, 1 / 32)
        )
        self.assertEqual(
            ledger["active_token_coefficients"], (1 / 16, 3 / 64, -3 / 16, 5 / 64)
        )
        for total, eligible in zip(
            ledger["total_token_coefficients"],
            ledger["active_token_coefficients"],
            strict=True,
        ):
            self.assertAlmostEqual(eligible, total * 48 / 32)
        self.assertAlmostEqual(ledger["regularizer_total_token_mean"], 0.003)

    def test_length_ledgers_have_distinct_semantics(self) -> None:
        advantages = (
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(-3, 4),
            Fraction(1, 4),
        )
        ledger = reference.length_ledger(reference.STUDIO_LENGTHS, advantages)
        self.assertEqual(ledger["total_tokens"], 48)
        expected = (1 / 24, 1 / 16, -1 / 8, 5 / 48)
        for actual, wanted in zip(
            ledger["total_token_coefficients"], expected, strict=True
        ):
            self.assertTrue(math.isclose(actual, wanted, rel_tol=1e-12))
        self.assertTrue(math.isclose(ledger["total_token_sum"], 1 / 12))
        self.assertTrue(math.isclose(ledger["per_trajectory_sum"], 0.0, abs_tol=1e-15))
        self.assertEqual(ledger["active_counts"], reference.STUDIO_LENGTHS)
        self.assertEqual(ledger["active_tokens"], 48)
        self.assertEqual(
            ledger["active_token_coefficients"],
            ledger["total_token_coefficients"],
        )
        two_path = reference.length_ledger((5, 20), (0.25, 0.25))
        self.assertEqual(two_path["total_token_coefficients"], (0.05, 0.2))
        self.assertEqual(two_path["per_trajectory_coefficients"], (0.125, 0.125))
        partial_active = reference.length_ledger(
            (5, 20), (0.25, 0.25), active_counts=(5, 10)
        )
        self.assertEqual(partial_active["active_tokens"], 15)
        self.assertEqual(partial_active["total_token_coefficients"], (0.05, 0.1))
        self.assertEqual(partial_active["per_trajectory_coefficients"], (0.125, 0.0625))
        self.assertEqual(partial_active["active_token_coefficients"], (1 / 12, 1 / 6))
        with self.assertRaises(ValueError):
            reference.length_ledger((5, 20), (0.25, 0.25), active_counts=(0, 0))
        with self.assertRaises(ValueError):
            reference.length_ledger((5, 20), (0.25, 0.25), ratio=0.0)

    def test_complete_trace_and_six_protocols(self) -> None:
        trace = reference.studio_trace()
        self.assertEqual(trace["mean_reward"], "3/4")
        self.assertEqual(trace["advantages"], ["1/4", "1/4", "-3/4", "1/4"])
        self.assertEqual(trace["representative_eligible_counts"], [4, 3, 3, 3])
        self.assertTrue(math.isclose(trace["representative_penalty_mass"], 0.03672))
        self.assertEqual(len(trace["content_sha256"]), 64)
        checks = reference.all_checks()
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(record["status"] == "pass" for record in checks.values()))
        self.assertTrue(
            checks["RESULT_CH17_GROUP_BASELINE_ORACLE"]["all_invalid_fixtures_rejected"]
        )
        self.assertTrue(
            checks["RESULT_CH17_MIRROR_DESCENT_ORACLE"][
                "small_eta_concentrates_on_maximum"
            ]
        )
        self.assertTrue(
            checks["RESULT_CH17_LENGTH_LEDGER_ORACLE"][
                "all_invalid_denominators_rejected"
            ]
        )


if __name__ == "__main__":
    unittest.main()
