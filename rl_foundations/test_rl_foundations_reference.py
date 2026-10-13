"""Independent finite oracles for the RL foundations teaching examples."""

from __future__ import annotations

from itertools import product
import math
import random
import unittest

from .reference import (
    Episode,
    Policy,
    STUDIO_LENGTHS,
    STUDIO_REWARDS,
    action_logprobs,
    batch_gradient,
    categorical_kl,
    enumerated_gradient,
    exact_gradient,
    expected_reward,
    group_coefficients,
    learning_demo,
    masked_log_probability,
    policy_loss,
    ppo_ratio_derivative,
    ppo_score,
    sample,
    sequence_ratio,
    sigmoid,
    update,
)


class RLFoundationsTests(unittest.TestCase):
    def test_exact_enumeration_matches_direct_and_finite_difference_gradients(self):
        for u, v in ((0.0, -math.log(3)), (-1.2, 0.3), (1.7, -2.4)):
            policy = Policy(u, v)
            expected = exact_gradient(policy)
            for baseline in (-0.7, 0.0, expected_reward(policy), 1.2):
                for actual, wanted in zip(
                    enumerated_gradient(policy, baseline), expected, strict=True
                ):
                    self.assertAlmostEqual(actual, wanted, places=12)
            step = 1e-5
            numeric = (
                (
                    expected_reward(Policy(u + step, v))
                    - expected_reward(Policy(u - step, v))
                )
                / (2 * step),
                (
                    expected_reward(Policy(u, v + step))
                    - expected_reward(Policy(u, v - step))
                )
                / (2 * step),
            )
            for actual, wanted in zip(numeric, expected, strict=True):
                self.assertAlmostEqual(actual, wanted, places=9)

    def test_seeded_complete_update_matches_printed_values(self):
        behavior = Policy()
        episode = sample(behavior, random.Random(7))
        self.assertEqual(episode, Episode(True, True))
        self.assertAlmostEqual(
            policy_loss(behavior, [episode], 0.5), 0.4904146265058631
        )
        gradient = batch_gradient(behavior, [episode], 0.5)
        self.assertEqual(gradient, (0.25, 0.125))
        current = update(behavior, gradient, 0.4)
        self.assertEqual(current, Policy(0.1, -math.log(3) + 0.05))
        self.assertAlmostEqual(expected_reward(current), 0.5218582522872478)
        self.assertEqual(behavior, Policy())
        self.assertAlmostEqual(sigmoid(-math.log(3) + 0.3), 0.31032244201237035)

    def test_loss_gradient_matches_finite_differences_for_both_coefficient_signs(self):
        policy = Policy()
        step = 1e-5
        for episode in (Episode(True, True), Episode(False, False)):
            ascent = batch_gradient(policy, [episode], 0.5)
            for axis in (0, 1):
                offsets = [0.0, 0.0]
                offsets[axis] = step
                plus = Policy(policy.u + offsets[0], policy.v + offsets[1])
                minus = Policy(policy.u - offsets[0], policy.v - offsets[1])
                numeric = (
                    policy_loss(plus, [episode], 0.5)
                    - policy_loss(minus, [episode], 0.5)
                ) / (2 * step)
                self.assertAlmostEqual(numeric, -ascent[axis], places=9)

    def test_action_mask_and_probability_factorization(self):
        logs = action_logprobs(Policy(), Episode(True, True))
        self.assertAlmostEqual(math.exp(sum(logs)), 3 / 8)
        for observation in (None, -99.0, float("nan")):
            self.assertEqual(
                masked_log_probability((logs[0], observation, logs[1]), (1, 0, 1)),
                sum(logs),
            )
        with self.assertRaises(ValueError):
            masked_log_probability((logs[0], None, logs[1]), (1, 1, 1))
        with self.assertRaises(ValueError):
            masked_log_probability(logs, (1,))
        # Masking the observation does not detach the answer's dependence on v.
        changed = action_logprobs(Policy(v=Policy().v + 0.1), Episode(True, True))
        self.assertNotEqual(changed[1], logs[1])

    def test_frozen_group_and_zero_variance(self):
        self.assertEqual(sum(STUDIO_LENGTHS), 48)
        self.assertEqual(sum(STUDIO_REWARDS) / 4, 0.75)
        self.assertEqual(
            group_coefficients(STUDIO_REWARDS, expected_count=4),
            (0.25, 0.25, -0.75, 0.25),
        )
        self.assertEqual(
            group_coefficients(STUDIO_REWARDS, expected_count=4, leave_one_out=True),
            (1 / 3, 1 / 3, -1.0, 1 / 3),
        )
        for standardize in (False, True):
            self.assertEqual(
                group_coefficients(
                    (1.0,) * 4, expected_count=4, standardize=standardize
                ),
                (0.0,) * 4,
            )
        for actual, wanted in zip(
            group_coefficients(STUDIO_REWARDS, expected_count=4, standardize=True),
            (1 / math.sqrt(3), 1 / math.sqrt(3), -math.sqrt(3), 1 / math.sqrt(3)),
            strict=True,
        ):
            self.assertAlmostEqual(actual, wanted)

    def test_missing_invalid_and_singleton_groups_are_rejected(self):
        for rewards, count in (
            ((), 0),
            ((1.0,), 1),
            ((1.0, 1.0, 1.0), 4),
            ((1.0, None), 2),
            ((1.0, float("nan")), 2),
        ):
            with self.assertRaises(ValueError):
                group_coefficients(rewards, expected_count=count)

    def test_all_sixteen_groups_verify_inclusive_and_leave_one_out_expectations(self):
        probability = 0.25
        totals = {False: 0.0, True: 0.0}
        groups = list(product((0.0, 1.0), repeat=4))
        self.assertEqual(len(groups), 16)
        for rewards in groups:
            mass = math.prod(
                probability if reward else 1 - probability for reward in rewards
            )
            for loo in totals:
                coefficients = group_coefficients(
                    rewards, expected_count=4, leave_one_out=loo
                )
                estimate = (
                    sum(
                        coefficient * (reward - probability)
                        for coefficient, reward in zip(
                            coefficients, rewards, strict=True
                        )
                    )
                    / 4
                )
                totals[loo] += mass * estimate
        self.assertAlmostEqual(totals[False], (3 / 4) * (3 / 16))
        self.assertAlmostEqual(totals[True], 3 / 16)

    def test_importance_identity_and_sequence_product(self):
        behavior = Policy()
        current = Policy(0.8, -0.7)
        weighted = 0.0
        for inspect, blue in product((False, True), repeat=2):
            episode = Episode(inspect, blue)
            old = action_logprobs(behavior, episode)
            new = action_logprobs(current, episode)
            ratio = sequence_ratio(new, old)
            self.assertAlmostEqual(
                ratio, math.prod(math.exp(n - o) for n, o in zip(new, old, strict=True))
            )
            weighted += math.exp(sum(old)) * ratio * float(blue)
        self.assertAlmostEqual(weighted, expected_reward(current))
        with self.assertRaises(ValueError):
            sequence_ratio((-0.1,), (float("-inf"),))

    def test_ppo_signs_tails_and_kink_convention(self):
        cases = (
            (1, 0.6, 0.6, 1),
            (1, 1.4, 1.2, 0),
            (-1, 0.6, -0.8, 0),
            (-1, 1.4, -1.4, -1),
        )
        for advantage, ratio, value, derivative in cases:
            self.assertAlmostEqual(ppo_score(ratio, advantage), value)
            self.assertEqual(ppo_ratio_derivative(ratio, advantage), derivative)
            step = 1e-6
            numeric = (
                ppo_score(ratio + step, advantage) - ppo_score(ratio - step, advantage)
            ) / (2 * step)
            self.assertAlmostEqual(numeric, derivative, places=8)
        for advantage, kink in ((1.0, 1.2), (-1.0, 0.8)):
            self.assertEqual(ppo_ratio_derivative(kink, advantage), 0)
            slopes = {
                ppo_ratio_derivative(kink - 1e-8, advantage),
                ppo_ratio_derivative(kink + 1e-8, advantage),
            }
            self.assertEqual(slopes, {0.0, advantage})
        self.assertEqual(ppo_ratio_derivative(1.0, 0.0), 0.0)

    def test_kl_directions_and_support(self):
        self.assertAlmostEqual(
            categorical_kl((0.75, 0.25), (0.5, 0.5)), 0.130812035941137
        )
        self.assertAlmostEqual(
            categorical_kl((0.5, 0.5), (0.75, 0.25)), 0.1438410362258904
        )
        with self.assertRaises(ValueError):
            categorical_kl((0.5, 0.5), (1.0, 0.0))

    def test_fresh_batch_learning_is_reproducible_and_changes_policy(self):
        learned = learning_demo()
        self.assertEqual(learned, learning_demo())
        self.assertNotEqual(learned, Policy())
        self.assertGreater(expected_reward(learned), expected_reward(Policy()))


if __name__ == "__main__":
    unittest.main()
