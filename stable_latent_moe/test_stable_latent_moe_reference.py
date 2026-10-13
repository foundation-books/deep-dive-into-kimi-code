"""Tests for the Chapter 8 Stable LatentMoE reference semantics."""

from __future__ import annotations

import unittest
from typing import cast

from . import reference as REFERENCE


class StableLatentMoEForwardTests(unittest.TestCase):
    def test_complete_forward_matches_independent_loop(self) -> None:
        first = REFERENCE.stable_latent_moe_forward()
        second = REFERENCE.stable_latent_moe_forward_oracle()
        self.assertEqual(len(first.tokens), 10)
        self.assertLessEqual(REFERENCE.max_forward_error(first, second), 1e-12)
        self.assertTrue(all(len(token.output) == 8 for token in first.tokens))
        self.assertTrue(all(len(token.latent_input) == 4 for token in first.tokens))
        self.assertTrue(
            all(
                len(expert.situ_intermediate) == 3
                for token in first.tokens
                for expert in token.experts
            )
        )

    def test_chapter_two_routes_and_x3_weights_are_preserved(self) -> None:
        routes = tuple(
            decision.selected for decision in REFERENCE.running_decisions().values()
        )
        self.assertEqual(REFERENCE.expert_loads(routes), (7, 6, 4, 3))
        x3 = REFERENCE.running_decisions()["x3"]
        self.assertEqual(x3.selected, (4, 1))
        self.assertEqual(len(x3.raw_scores), 4)
        self.assertAlmostEqual(x3.weights[0], 0.551543422096722)
        self.assertAlmostEqual(x3.weights[1], 0.44845657790327803)
        self.assertAlmostEqual(sum(x3.weights), 1.0)


class NormalizationAndActivationTests(unittest.TestCase):
    def test_rmsnorm_scale_contract(self) -> None:
        vector = (1.0, -2.0, 3.0, -4.0)
        base = REFERENCE.rms_norm(vector, 0.0)
        scaled = REFERENCE.rms_norm(tuple(7.0 * value for value in vector), 0.0)
        self.assertLessEqual(
            max(abs(a - b) for a, b in zip(base, scaled, strict=True)), 1e-12
        )
        stabilized = REFERENCE.rms_norm(vector, 1e-6)
        stabilized_scaled = REFERENCE.rms_norm(
            tuple(7.0 * value for value in vector), 1e-6
        )
        self.assertGreater(
            max(abs(a - b) for a, b in zip(stabilized, stabilized_scaled, strict=True)),
            0.0,
        )
        with self.assertRaises(ValueError):
            REFERENCE.rms_norm((0.0, 0.0), 0.0)

    def test_situ_coordinate_bound_and_limit(self) -> None:
        ratio = REFERENCE.max_situ_grid_ratio(
            (-1000.0, -25.0, -4.0, 0.0, 4.0, 25.0, 1000.0)
        )
        self.assertLessEqual(ratio, 1.0)
        gate, up = 1.25, -0.75
        situ = REFERENCE.situ_coordinate(gate, up, 1e6, 1e6)
        swiglu = REFERENCE.swiglu_coordinate(gate, up)
        self.assertAlmostEqual(situ, swiglu, places=10)


class QuantileBalancingTests(unittest.TestCase):
    def test_exact_coordinate_update_and_centering(self) -> None:
        update = REFERENCE.exact_qb_update()
        self.assertEqual(REFERENCE.RUNNING_DIMS.target_load, 5)
        self.assertEqual(update["target_load"], 5)
        self.assertEqual(update["current_loads"], (7, 6, 4, 3))
        self.assertAlmostEqual(cast(float, update["centered_mean"]), 0.0)
        self.assertEqual(update["frozen_cutoff_counts"], (3, 5, 5, 3))
        self.assertEqual(update["endpoint_equality_counts"], (3, 1, 4, 6))
        frozen_counts = cast(tuple[int, ...], update["frozen_cutoff_counts"])
        current_loads = cast(tuple[int, ...], update["current_loads"])
        recomputed_loads = cast(tuple[int, ...], update["recomputed_loads"])
        centered_bias = cast(tuple[float, ...], update["centered_bias"])
        self.assertTrue(all(count <= 5 for count in frozen_counts))
        self.assertEqual(sum(current_loads), 20)
        self.assertEqual(sum(recomputed_loads), 20)
        shifted = tuple(value + 3.0 for value in centered_bias)
        self.assertEqual(
            REFERENCE.selected_sets(biases=shifted),
            REFERENCE.selected_sets(biases=centered_bias),
        )
        witness = REFERENCE.interior_qb_coordinate_witness()
        self.assertEqual(witness["alpha"], update["alpha"])
        self.assertEqual(witness["frozen_cutoff_counts"], (3, 5, 5, 3))
        self.assertEqual(witness["midpoint_equality_counts"], (3, 0, 0, 6))
        witness_beta = cast(tuple[float, ...], witness["beta"])
        endpoint_beta = cast(tuple[float, ...], update["beta"])
        self.assertTrue(
            all(
                interior >= endpoint - 1e-12
                for interior, endpoint in zip(witness_beta, endpoint_beta, strict=True)
            )
        )
        self.assertAlmostEqual(cast(float, witness["centered_mean"]), 0.0)

    def test_four_token_update_matches_printed_exact_decimal_example(self) -> None:
        scores = (
            (0.90, 0.80, 0.20, 0.10),
            (0.85, 0.30, 0.75, 0.15),
            (0.25, 0.88, 0.65, 0.20),
            (0.20, 0.70, 0.30, 0.95),
        )
        update = REFERENCE.exact_qb_update(scores, (0.10, -0.05, 0.0, 0.05), 2)
        self.assertEqual(update["current_routes"], ((1, 2), (1, 3), (2, 3), (4, 2)))
        self.assertEqual(update["current_loads"], (2, 3, 2, 1))
        self.assertEqual(update["frozen_cutoff_counts"], (2, 2, 2, 1))
        self.assertEqual(update["endpoint_equality_counts"], (2, 1, 2, 2))
        self.assertEqual(update["next_routes"], ((1, 2), (1, 3), (3, 2), (4, 1)))
        self.assertEqual(update["recomputed_loads"], (3, 2, 2, 1))
        self.assertEqual(sum(cast(tuple[int, ...], update["current_loads"])), 8)
        self.assertEqual(sum(cast(tuple[int, ...], update["recomputed_loads"])), 8)

    def test_balanced_assignment_dynamic_program(self) -> None:
        objective, assignment, loads = REFERENCE.balanced_assignment_dp(
            REFERENCE.QB_SCORES, 2
        )
        self.assertGreater(objective, 0.0)
        self.assertEqual(len(assignment), 10)
        self.assertEqual(loads, (5, 5, 5, 5))
        self.assertTrue(all(len(route) == 2 for route in assignment))

    def test_histogram_is_partition_invariant_and_bin_accurate(self) -> None:
        pooled = REFERENCE.histogram_quantile_estimate(bins=32)
        partitioned = REFERENCE.partitioned_histogram_estimate(
            ((0, 3, 6, 9), (1, 4, 7), (2, 5, 8)), bins=32
        )
        self.assertEqual(pooled.counts, partitioned.counts)
        self.assertEqual(pooled.recovered, partitioned.recovered)
        exact = REFERENCE.exact_histogram_targets()
        self.assertTrue(
            all(
                abs(estimate - target) <= pooled.width + 1e-12
                for estimate, target in zip(pooled.recovered, exact, strict=True)
            )
        )


class LedgersAndFaultsTests(unittest.TestCase):
    def test_separate_logical_ledgers(self) -> None:
        payload = REFERENCE.payload_ledger()
        self.assertEqual(payload["route_records"], 20)
        self.assertEqual(payload["full_width_one_way_scalars"], 160)
        self.assertEqual(payload["latent_one_way_scalars"], 80)
        self.assertEqual(payload["full_to_latent_ratio"], 2.0)
        ledger = REFERENCE.resource_ledger()
        self.assertEqual(ledger.router_parameters, 32)
        self.assertEqual(ledger.down_parameters, 32)
        self.assertEqual(ledger.routed_expert_parameters, 144)
        self.assertEqual(ledger.aggregate_norm_parameters, 4)
        self.assertEqual(ledger.up_parameters, 32)
        self.assertEqual(ledger.matrix_multiply_adds_per_token, 168)
        self.assertEqual(ledger.route_payload_scalars, 80)

    def test_all_injected_faults_are_detected(self) -> None:
        faults = REFERENCE.fault_detection_summary()
        self.assertEqual(len(faults), 6)
        self.assertTrue(all(faults.values()))


if __name__ == "__main__":
    unittest.main()
