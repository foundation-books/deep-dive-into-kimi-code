"""Check the isolated semantic consequences printed in Volume I."""

from __future__ import annotations

from fractions import Fraction
from math import exp, log
import unittest

from . import implementation_insights as INSIGHTS


class ImplementationInsightTests(unittest.TestCase):
    """Verify each witness against its analytical expectation."""

    def test_unit_sum_does_not_protect_router_weights(self) -> None:
        result = INSIGHTS.routing_comparison()
        self.assertEqual(result["selected"], (4, 1))
        self.assertAlmostEqual(sum(result["raw_score_weights"]), 1.0)
        self.assertAlmostEqual(sum(result["biased_score_weights"]), 1.0)
        self.assertNotEqual(result["raw_score_output"], result["biased_score_output"])

    def test_history_loss_outlives_convolution_window(self) -> None:
        good = INSIGHTS.convolution_continuation()
        bad = INSIGHTS.convolution_continuation(reset=True)
        self.assertEqual(good[0], Fraction(5, 4))
        self.assertEqual(bad[0], Fraction(1, 2))
        self.assertEqual(
            tuple(a - b for a, b in zip(good, bad, strict=True)),
            tuple(map(Fraction, ("3/4", "13/16", "37/64", "37/256", "37/1024"))),
        )

    def test_zero_query_still_distinguishes_value_normalization(self) -> None:
        result = INSIGHTS.depth_value_comparison()
        self.assertEqual(result["raw_values"], (1.0, 4.0 / 3.0))
        self.assertEqual(result["weights"], (1.0 / 3.0,) * 3)
        self.assertNotEqual(result["raw_values"], result["normalized_values"])

    def test_fixed_latent_requires_dynamic_gate(self) -> None:
        self.assertEqual(INSIGHTS.gate_comparison()["dynamic_outputs"], (5, 3))

    def test_zero_candidate_isolates_base_rate_decay(self) -> None:
        result = INSIGHTS.muon_decay_comparison()
        self.assertAlmostEqual(result["base_rate_decay"], 0.1249875)
        self.assertAlmostEqual(result["adjusted_rate_decay"], 0.12499)
        self.assertAlmostEqual(result["difference"], 0.0000025)

    def test_masked_boundary_does_not_isolate_remaining_target(self) -> None:
        loss = INSIGHTS.packed_target_loss
        self.assertAlmostEqual(loss(0.0, False), log(2))
        self.assertAlmostEqual(loss(4.0, False), log(1 + exp(-3)))
        self.assertEqual(loss(0.0, True), loss(4.0, True))
        self.assertNotEqual(loss(0.0, False), loss(4.0, False))

    def test_count_preserving_permutation_changes_composition(self) -> None:
        result = INSIGHTS.schedule_comparison()
        self.assertTrue(result["same_counts"])
        self.assertTrue(result["permutation_rejected"])
        self.assertEqual(result["correct_scalar_result"], 44)
        self.assertEqual(result["permuted_scalar_result"], 40)
