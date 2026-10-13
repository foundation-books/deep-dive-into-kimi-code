"""Dependency-free tests for the Chapter 25 semantic companion."""

from __future__ import annotations

from fractions import Fraction
import unittest

from . import reference


class CodingAgentsKimiDevReferenceTests(unittest.TestCase):
    def test_task_identity(self) -> None:
        result = reference.task_identity_check()
        self.assertEqual(result["correct_base"], "applied")
        self.assertEqual(result["wrong_base"], "base_mismatch")
        self.assertEqual(result["second_application"], "already_applied")

    def test_crop_mapping(self) -> None:
        result = reference.crop_mapping_check()
        self.assertEqual(result["correct_root_point"], ["40", "100"])
        self.assertEqual(result["buggy_root_point"], ["40", "80"])
        self.assertEqual(result["missing_offset"], "20")

    def test_typed_test_transitions(self) -> None:
        result = reference.typed_test_transition_check()
        self.assertEqual(result["canonical_admissible_tests"], ["t1", "t3"])
        self.assertEqual(result["canonical_nonsemantic_tests"], ["t4"])
        self.assertEqual(result["canonical_pairs"]["t4"], ["timeout", "not_run"])
        self.assertEqual(result["exercise_admissible_tests"], ["t1"])
        self.assertEqual(result["exercise_reproduction_rate_all_generated"], "1/5")
        self.assertEqual(result["exercise_invalid_before_tests"], ["t3", "t5"])

    def test_role_rewards(self) -> None:
        result = reference.role_reward_check()
        self.assertEqual(result["bugfixer_mean"], "1/2")
        self.assertEqual(result["bugfixer_advantages"], ["1/2", "-1/2", "-1/2", "1/2"])
        self.assertEqual(result["testwriter_mean"], "3/4")
        self.assertEqual(result["testwriter_advantages"], ["-3/4", "1/4", "1/4", "1/4"])

    def test_self_play(self) -> None:
        result = reference.self_play_check()
        self.assertEqual(result["scores"], {"b1": "2", "b2": "4/3", "b3": "3/2"})
        self.assertEqual(
            result["deduplicated_scores"], {"b1": "2", "b2": "3/2", "b3": "5/4"}
        )
        self.assertEqual(result["winner"], "b1")
        self.assertEqual(result["deduplicated_winner"], "b1")

    def test_self_play_zero_denominator_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            reference._self_play_score([(1, 0)], [(1, 0)])

    def test_pass_at_k(self) -> None:
        result = reference.pass_at_k_check()
        self.assertEqual(result["pass_at_k"], {"1": "1/4", "2": "1/2", "4": "1"})
        self.assertTrue(result["enumerated_match"])

    def test_adaptation_curves(self) -> None:
        result = reference.adaptation_check()
        self.assertFalse(result["a_dominates_b"])
        self.assertFalse(result["b_dominates_a"])
        self.assertEqual(result["hitting_time_a"], 5_000)
        self.assertEqual(result["hitting_time_b"], 1_000)
        self.assertEqual(result["trajectory_ratio_a_over_b"], "5")
        self.assertEqual(result["stage_cutoff_contrast"], "1/5")

    def test_evidence_handoff(self) -> None:
        result = reference.evidence_handoff_check()
        self.assertEqual(result["definition_count"], 18)
        self.assertEqual(result["invariant_count"], 18)
        self.assertEqual(result["pending_protocol_count"], 8)
        self.assertEqual(result["exercise_count"], 10)
        self.assertFalse(result["direct_kimi_dev_to_k3_lineage_known"])

    def test_invalid_pass_at_k_parameters_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            reference._pass_at_k(4, 5, 1)
        with self.assertRaises(ValueError):
            reference._pass_at_k(4, 1, 0)

    def test_fraction_fixture_is_exact(self) -> None:
        self.assertEqual(Fraction(44, 80) - Fraction(42, 120), Fraction(1, 5))


if __name__ == "__main__":
    unittest.main()
