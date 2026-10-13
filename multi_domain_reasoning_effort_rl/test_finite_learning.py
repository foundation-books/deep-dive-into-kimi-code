"""Behavioral checks for the Chapter 21 executed four-route teaching study."""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
import json
import math
import random
import unittest

from .finite_policy import (
    BASELINE,
    INITIAL_PROBABILITIES,
    MULTIPLIERS,
    ROUTES,
    Route,
    cap,
    cutoff_metrics,
    exact_gradient,
    exact_step,
    expected_rewards,
    initial_logits,
    metrics,
    route_states,
    softmax,
)
from .reference_core import effective_reward
from .train_fixture import (
    SampledState,
    Stage,
    exact_run,
    group_gradient,
    sampled_group_step,
    sampled_run,
)


class FiniteLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = initial_logits(INITIAL_PROBABILITIES)
        self.high = cap(BASELINE, MULTIPLIERS["high"])

    def test_checked_first_update_and_selected_values(self) -> None:
        self.assertEqual(expected_rewards(ROUTES, self.high), (0.35, -1.0, 0.85, 0.75))
        self.assertAlmostEqual(
            metrics(ROUTES, self.logits, self.high)["effective_reward"], -0.095
        )
        gradient = exact_gradient(ROUTES, self.logits, self.high)
        for got, expected in zip(
            gradient, (0.178, -0.362, 0.14175, 0.04225), strict=True
        ):
            self.assertAlmostEqual(got, expected, places=12)
        updated = exact_step(ROUTES, self.logits, self.high, 1)
        for got, expected in zip(
            softmax(updated),
            (0.48696622, 0.28377871, 0.17611118, 0.05314389),
            strict=True,
        ):
            self.assertAlmostEqual(got, expected, places=7)

    def test_gradient_matches_central_difference(self) -> None:
        for logits in (self.logits, (3.0, -1.0, 0.2, -2.3), (-2.0, 0.5, 1.3, 4.0)):
            analytic = exact_gradient(ROUTES, logits, self.high)
            for index in range(len(logits)):
                plus, minus = list(logits), list(logits)
                plus[index] += 1e-6
                minus[index] -= 1e-6
                objective_plus = metrics(ROUTES, plus, self.high)["effective_reward"]
                objective_minus = metrics(ROUTES, minus, self.high)["effective_reward"]
                self.assertAlmostEqual(
                    analytic[index], (objective_plus - objective_minus) / 2e-6, places=8
                )

    def test_probability_and_zero_update_guards(self) -> None:
        p = softmax((1000.0, 1001.0, -1000.0))
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in p))
        self.assertAlmostEqual(sum(p), 1)
        self.assertEqual(exact_step(ROUTES, self.logits, self.high, 0), self.logits)
        with self.assertRaises(ValueError):
            cap(Fraction(0), Fraction(1))
        with self.assertRaises(ValueError):
            cap(Fraction(1), Fraction(-1))
        with self.assertRaises(ValueError):
            cap(float("nan"), Fraction(1))

    def test_no_hidden_length_bonus_and_all_over_budget(self) -> None:
        equal_reward_routes = (Route("short", 40, 1.0), Route("long", 70, 1.0))
        self.assertEqual(
            exact_gradient(equal_reward_routes, (0.0, 0.0), Fraction(80)), (0.0, 0.0)
        )
        self.assertEqual(exact_gradient(ROUTES, self.logits, Fraction(10)), (0.0,) * 4)
        updated, group = sampled_group_step(
            ROUTES, self.logits, Fraction(10), 1, 32, random.Random(0)
        )
        self.assertEqual(updated, self.logits)
        self.assertEqual(group["gradient"], [0.0] * 4)
        self.assertTrue(group["no_centered_signal"])

    def test_strict_threshold_and_frozen_cap_rescore(self) -> None:
        equal, over = route_states(
            (Route("equal", 80, 1), Route("over", 81, 1)), Fraction(80)
        )
        self.assertEqual(effective_reward(1, equal), 1)
        self.assertEqual(effective_reward(0, equal), 0)
        self.assertEqual(effective_reward(1, over), -1)
        self.assertEqual(effective_reward(0, over), -1)
        max_metrics = metrics(ROUTES, self.logits, Fraction(500))
        high_metrics = metrics(ROUTES, self.logits, self.high)
        self.assertEqual(max_metrics["probabilities"], high_metrics["probabilities"])
        self.assertEqual(max_metrics["raw_success"], high_metrics["raw_success"])
        self.assertEqual(max_metrics["expected_cost"], high_metrics["expected_cost"])
        self.assertNotEqual(
            max_metrics["effective_reward"], high_metrics["effective_reward"]
        )

    def test_exact_stage_boundary_keeps_previous_cap_for_checkpoint(self) -> None:
        schedule = (Stage("max", 2, Fraction(500)), Stage("high", 3, self.high))
        trace = exact_run(ROUTES, self.logits, schedule, 1)
        self.assertEqual(trace[2]["stage"], "max")
        self.assertEqual(trace[3]["stage"], "high")
        self.assertEqual(trace[2]["cap"], 500.0)
        self.assertEqual(trace[3]["cap"], 160.0)

    def test_group_mean_scaling_and_corrected_expectation(self) -> None:
        probabilities = (0.4, 0.6)
        exact = (0.4 * (1 - 0.4), 0.6 * (0 - 0.4))
        inclusive = [0.0, 0.0]
        corrected = [0.0, 0.0]
        for first in (0, 1):
            for second in (0, 1):
                weight = probabilities[first] * probabilities[second]
                rewards = (float(first == 0), float(second == 0))
                for total, correction in ((inclusive, False), (corrected, True)):
                    estimate = group_gradient(
                        (first, second), rewards, probabilities, corrected=correction
                    )
                    for axis in range(2):
                        total[axis] += weight * estimate[axis]
        for axis in range(2):
            self.assertAlmostEqual(inclusive[axis], 0.5 * exact[axis])
            self.assertAlmostEqual(corrected[axis], exact[axis])

    def test_fresh_collection_resume_and_stage_boundaries(self) -> None:
        schedule = (Stage("max", 2, Fraction(500)), Stage("high", 3, self.high))
        full, saved = sampled_run(
            ROUTES, self.logits, schedule, 1, 32, 42, capture=frozenset((1, 2, 3, 5))
        )
        first, _ = sampled_run(ROUTES, self.logits, schedule, 1, 32, 42, stop_at=2)
        restored = SampledState.from_json(json.loads(json.dumps(first.to_json())))
        resumed, _ = sampled_run(
            ROUTES, self.logits, schedule, 1, 32, 42, resume=restored
        )
        self.assertEqual(full, resumed)
        self.assertEqual(saved[1]["stage"], "max")
        self.assertEqual(saved[2]["stage"], "high")
        for previous, following in pairwise(saved):
            if following["completed_updates"] == previous["completed_updates"] + 1:
                self.assertEqual(
                    previous["collected"]["updated_probabilities"],
                    following["collected"]["collection_probabilities"],
                )

    def test_support_and_cutoff_accounting(self) -> None:
        routes = ROUTES[:3]
        p0 = tuple(value / 0.95 for value in INITIAL_PROBABILITIES[:3])
        z = initial_logits(p0)
        final, _ = sampled_run(routes, z, (Stage("low", 20, Fraction(80)),), 1, 32, 0)
        self.assertEqual(len(final.route_counts), 3)
        self.assertIsNone(final.first_d_majority_update)
        self.assertAlmostEqual(
            metrics(routes, final.logits, Fraction(80))["raw_success"],
            sum(
                p * r.success_probability
                for p, r in zip(softmax(final.logits), routes, strict=True)
            ),
        )
        cutoff = cutoff_metrics(ROUTES, self.logits, 80)
        self.assertAlmostEqual(cutoff["raw_success"], 0.4 * 0.35 + 0.05 * 0.75)
        self.assertAlmostEqual(
            cutoff["consumed_cost"], 0.4 * 40 + 0.4 * 80 + 0.15 * 80 + 0.05 * 70
        )


if __name__ == "__main__":
    unittest.main()
