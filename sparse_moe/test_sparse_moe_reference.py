from __future__ import annotations

from dataclasses import replace
import unittest

from .reference import (
    ORIGINS,
    PRODUCTION_COMPARISON,
    RUNNING_BIASES,
    balancing_demo,
    build_route_records,
    capacity_ledger,
    combine,
    dense_ffn,
    dispatch,
    execute,
    expert_loads,
    fault_detection_summary,
    incidence_matrix,
    incorrect_biased_weights,
    max_abs_difference,
    payload_ledger,
    placement_ledger,
    production_comparison,
    route,
    running_token_payloads,
    standard_latent_shape_check,
    token_major_reference,
    validate_returns,
)


class RouterReferenceTests(unittest.TestCase):
    def test_selection_and_unbiased_weights(self) -> None:
        result = route((0.1, 0.7, 0.4, 0.6), RUNNING_BIASES, 2)
        self.assertEqual(result.selected, (4, 1))
        self.assertAlmostEqual(result.weights[0], 0.551543422096722)
        self.assertAlmostEqual(result.weights[1], 0.44845657790327803)
        self.assertAlmostEqual(sum(result.weights), 1.0)

    def test_biased_weight_counterexample_changes_mixture(self) -> None:
        result = route((0.1, 0.7, 0.4, 0.6), RUNNING_BIASES, 2)
        wrong = incorrect_biased_weights(result, RUNNING_BIASES)
        self.assertAlmostEqual(wrong[0], 0.5070299944600027)
        self.assertNotAlmostEqual(wrong[0], result.weights[0])
        self.assertAlmostEqual(sum(wrong), 1.0)

    def test_ties_prefer_lower_expert_index(self) -> None:
        result = route((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2)
        self.assertEqual(result.selected, (1, 2))

    def test_extreme_negative_logits_keep_normalized_relative_weights(self) -> None:
        result = route((-1001.0, -1000.0), (0.0, 0.0), 2)
        self.assertEqual(result.scores, (0.0, 0.0))
        self.assertEqual(result.selected, (2, 1))
        self.assertAlmostEqual(result.weights[0], 0.7310585786300049)
        self.assertAlmostEqual(sum(result.weights), 1.0)


class DenseBaselineTests(unittest.TestCase):
    def test_dense_ffn_preserves_external_width(self) -> None:
        output = dense_ffn(
            (1.0, -2.0),
            ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        )
        self.assertEqual(output, (1.0, 4.0))

    def test_dense_ffn_rejects_an_untyped_contraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "return to the model width"):
            dense_ffn((1.0, 2.0), ((1.0, 0.0),), ((1.0,),))


class RouteConservationTests(unittest.TestCase):
    def test_incidence_and_loads(self) -> None:
        matrix = incidence_matrix()
        self.assertEqual(len(matrix), 10)
        self.assertTrue(all(sum(row) == 2 for row in matrix))
        self.assertEqual(expert_loads(matrix), (7, 6, 4, 3))

    def test_dispatch_execute_combine_matches_token_major_reference(self) -> None:
        payloads = running_token_payloads()
        dispatched = dispatch(build_route_records(payloads))
        returned = execute(dispatched)
        self.assertEqual(validate_returns(dispatched, returned), ())
        combined = combine(payloads, dispatched, returned)
        direct = token_major_reference(payloads)
        self.assertEqual(tuple(combined), ORIGINS)
        error = max(
            max_abs_difference(combined[origin], direct[origin]) for origin in ORIGINS
        )
        self.assertLessEqual(error, 1e-12)

    def test_all_declared_faults_are_detected(self) -> None:
        self.assertTrue(all(fault_detection_summary().values()))

    def test_nonfinite_returned_weight_is_rejected_before_combine(self) -> None:
        payloads = running_token_payloads()
        dispatched = dispatch(build_route_records(payloads))
        returned = list(execute(dispatched))
        returned[0] = replace(returned[0], weight=float("nan"))

        self.assertIn(
            "non-finite weight", " ".join(validate_returns(dispatched, returned))
        )
        with self.assertRaisesRegex(ValueError, "non-finite weight"):
            combine(payloads, dispatched, returned)

    def test_route_weight_tolerance_must_be_finite_and_nonnegative(self) -> None:
        dispatched = dispatch(build_route_records())
        returned = execute(dispatched)
        for tolerance in (float("nan"), float("inf"), -1.0):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    validate_returns(dispatched, returned, tolerance)


class CapacityBalanceAndPlacementTests(unittest.TestCase):
    def test_capacity_sweep(self) -> None:
        expected = {
            1.0: (5, 3, 3),
            1.2: (6, 1, 5),
            1.5: (8, 0, 12),
        }
        for factor, values in expected.items():
            ledger = capacity_ledger((7, 6, 4, 3), factor)
            self.assertEqual(
                (
                    ledger["capacity"],
                    ledger["overflow_routes"],
                    ledger["padding_slots"],
                ),
                values,
            )

    def test_balancing_demos(self) -> None:
        demo = balancing_demo()
        self.assertEqual(demo["fixed_step_directions"], (-1, -1, 1, 1))
        self.assertEqual(demo["quantile_initial_loads"], (4, 3, 1, 0))
        self.assertEqual(demo["quantile_final_loads"], (2, 2, 2, 2))

    def test_placement_counterexample(self) -> None:
        ledger = placement_ledger()
        self.assertEqual(ledger["A"]["rank_received_loads"], [13, 7])
        self.assertEqual(ledger["A"]["crossing_dispatch_routes"], 5)
        self.assertEqual(ledger["B"]["rank_received_loads"], [11, 9])
        self.assertEqual(ledger["B"]["crossing_dispatch_routes"], 9)


class LatentAndProductionLedgerTests(unittest.TestCase):
    def test_payload_values_and_controlled_shapes(self) -> None:
        self.assertEqual(
            payload_ledger(10, 2, 8, 4),
            {
                "route_records": 20,
                "full_width_one_way_scalars": 160,
                "latent_one_way_scalars": 80,
                "full_to_latent_ratio": 2.0,
            },
        )
        result = standard_latent_shape_check()
        self.assertEqual(result["output_shape"], [10, 8])
        self.assertLessEqual(result["controlled_max_abs_error"], 1e-12)

    def test_production_values_and_deductions(self) -> None:
        result = production_comparison()
        self.assertEqual(PRODUCTION_COMPARISON["K2"]["routed_experts"], 384)
        self.assertEqual(PRODUCTION_COMPARISON["K3"]["routed_experts"], 896)
        deductions = result["deduced"]
        self.assertEqual(deductions["K2_routed_sparsity"], 48.0)
        self.assertEqual(deductions["K3_routed_sparsity"], 56.0)
        self.assertEqual(deductions["full_to_latent_width_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
