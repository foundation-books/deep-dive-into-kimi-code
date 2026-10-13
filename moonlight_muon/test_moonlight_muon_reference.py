"""Deterministic tests for the Chapter 12 companion."""

from __future__ import annotations

import math
import unittest

from . import reference


class MoonlightMuonReferenceTests(unittest.TestCase):
    def test_role_partition_is_semantic(self) -> None:
        partition = reference.partition_parameters(reference.running_parameters())
        self.assertEqual(partition["Muon"], ("W_Q", "W_up", "W_route"))
        self.assertEqual(partition["AdamW"], ("E_tok", "W_out", "gamma"))

    def test_polar_rms_rank_law(self) -> None:
        for rows, columns, rank in ((4, 4, 4), (16, 4, 4), (8, 4, 4), (16, 4, 2)):
            entries = tuple(
                float(i + 1) if i < rank else 0.0 for i in range(min(rows, columns))
            )
            polar = reference.diagonal_polar(rows, columns, entries)
            self.assertAlmostEqual(
                reference.rms(polar), reference.polar_rms(rows, columns, rank)
            )

    def test_shape_scale_equalizes_only_full_rank(self) -> None:
        for rows, columns in ((4, 4), (16, 4), (8, 4), (4, 16)):
            rank = min(rows, columns)
            self.assertAlmostEqual(
                reference.corrected_polar_rms(rows, columns, rank), 0.2
            )
        self.assertAlmostEqual(
            reference.corrected_polar_rms(16, 4, 2), 0.2 / math.sqrt(2)
        )

    def test_finite_newton_schulz_is_approximate(self) -> None:
        matrix = reference.diagonal_matrix(4, 4, (4.0, 2.0, 1.0, 0.25))
        exact = reference.diagonal_polar(4, 4, (4.0, 2.0, 1.0, 0.25))
        iterate = reference.newton_schulz_zeroth_power(matrix, steps=5)
        error = reference.max_abs_difference(iterate, exact)
        self.assertGreater(error, 0.0)
        self.assertLess(error, 0.6)

    def test_printed_diagonal_trace_matches_fp64_map(self) -> None:
        probe = reference.diagonal_matrix(4, 4, (4.0, 2.0, 1.0, 0.25))
        expected = {
            0: (0.871576, 0.435788, 0.217894, 0.054473),
            1: (0.862420, 1.137817, 0.702135, 0.186863),
            2: (0.876921, 0.759573, 1.112320, 0.612956),
            3: (0.854025, 1.037416, 0.719034, 1.187437),
            4: (0.890319, 0.683168, 1.092066, 0.891268),
            5: (0.833292, 1.132988, 0.698079, 0.831842),
        }
        for steps, row in expected.items():
            actual = reference.newton_schulz_zeroth_power(probe, steps=steps)
            for index, printed in enumerate(row):
                self.assertAlmostEqual(actual[index][index], printed, delta=6e-7)

    def test_momentum_branches_have_distinct_step_two_candidates(self) -> None:
        zero = reference.zeros(4, 4)
        first = reference.running_gradients(1)["W_Q"]
        second = reference.running_gradients(2)["W_Q"]
        classical_buffer, _ = reference.momentum_candidate(zero, first, nesterov=False)
        nesterov_buffer, _ = reference.momentum_candidate(zero, first, nesterov=True)
        _, classical = reference.momentum_candidate(
            classical_buffer, second, nesterov=False
        )
        _, nesterov = reference.momentum_candidate(
            nesterov_buffer, second, nesterov=True
        )
        self.assertGreater(reference.max_abs_difference(classical, nesterov), 0.0)

    def test_nesterov_second_step_expansion_matches_recurrence(self) -> None:
        first = reference.running_gradients(1)["W_Q"]
        second = reference.running_gradients(2)["W_Q"]
        stored, _ = reference.momentum_candidate(reference.zeros(4, 4), first)
        next_buffer, direct = reference.momentum_candidate(stored, second)
        expanded = reference.add(
            reference.scale(first, 0.95**2), reference.scale(second, 1.95)
        )
        self.assertLess(reference.max_abs_difference(direct, expanded), 1e-14)
        self.assertAlmostEqual(next_buffer[0][0], 2.85)
        self.assertAlmostEqual(direct[0][0], 2.7075)
        self.assertAlmostEqual(direct[0][1], 1.95)
        classical = reference.add(reference.scale(first, 0.95), second)
        classical_finite = reference.newton_schulz_zeroth_power(classical)
        nesterov_finite = reference.newton_schulz_zeroth_power(direct)
        self.assertGreater(
            reference.max_abs_difference(classical_finite, nesterov_finite), 1e-3
        )

    def test_printed_query_endpoints_match_canonical_fixture(self) -> None:
        initial = reference.running_initial_muon_parameters()["W_Q"]
        gradients = (
            reference.running_gradients(1)["W_Q"],
            reference.running_gradients(2)["W_Q"],
        )
        first = reference.muon_step(initial, gradients[0], reference.zeros(4, 4))
        second = reference.muon_step(first.weight, gradients[1], first.momentum)
        self.assertAlmostEqual(first.direction[0][0], 0.443973936, delta=1e-9)
        self.assertAlmostEqual(first.weight[0][0], 0.999456026, delta=1e-9)
        self.assertAlmostEqual(
            reference.frobenius_norm(second.weight), 1.248590266183, delta=1e-12
        )
        self.assertAlmostEqual(
            sum(reference.flatten(second.weight)), 1.247545774709, delta=1e-12
        )

    def test_decoupled_decay_is_separate_shrink(self) -> None:
        weight = reference.diagonal_matrix(2, 2, (2.0, -1.0))
        decayed = reference.decoupled_weight_decay(weight, 1e-3, 0.1)
        self.assertAlmostEqual(
            reference.frobenius_norm(decayed), 0.9999 * reference.frobenius_norm(weight)
        )

    def test_decay_uses_base_rate_with_zero_direction(self) -> None:
        weight = reference.constant_matrix(16, 4, 0.125)
        result = reference.muon_step(
            weight, reference.zeros(16, 4), reference.zeros(16, 4)
        )
        self.assertAlmostEqual(result.weight[0][0], 0.1249875)
        self.assertNotAlmostEqual(result.weight[0][0], 0.12499)

    def test_state_ledger_is_dtype_qualified(self) -> None:
        ledger = reference.state_ledger(reference.running_parameters())
        self.assertEqual(ledger.muon_momentum_elements, 112)
        self.assertEqual(ledger.adamw_moment_elements, 200)
        self.assertEqual(ledger.adamw_step_counters, 3)
        self.assertEqual(ledger.total_elements, 315)
        self.assertEqual(ledger.fp32_bytes, 1260)

    def test_gather_transform_slice_matches_full_transform(self) -> None:
        full = reference.diagonal_matrix(8, 4, (4.0, 2.0, 1.0, 0.5))
        shards = (reference.slice_rows(full, 0, 4), reference.slice_rows(full, 4, 8))
        gathered = reference.gathered_muon_slices(shards)
        full_update = reference.concatenate_rows(gathered)
        expected = reference.scale(
            reference.newton_schulz_zeroth_power(full),
            reference.shape_scale(8, 4),
        )
        self.assertLess(reference.max_abs_difference(full_update, expected), 1e-12)
        local = reference.concatenate_rows(
            reference.independently_transformed_shards(shards)
        )
        self.assertGreater(reference.max_abs_difference(local, expected), 1e-3)

    def test_global_mean_gradient_precedes_sharded_update(self) -> None:
        rank_means = (
            reference.as_matrix(((1.0,), (3.0,))),
            reference.as_matrix(((5.0,), (7.0,))),
        )
        equal = reference.global_mean_gradient(rank_means, (1.0, 1.0))
        weighted = reference.global_mean_gradient(rank_means, (1.0, 3.0))
        self.assertEqual(reference.flatten(equal), (3.0, 5.0))
        self.assertEqual(reference.flatten(weighted), (4.0, 6.0))

        weight = reference.as_matrix(((10.0,), (20.0,)))
        serial = reference.muon_step(weight, weighted, reference.zeros(2, 1))
        candidate_shards = (
            reference.slice_rows(serial.candidate, 0, 1),
            reference.slice_rows(serial.candidate, 1, 2),
        )
        direction_shards = reference.gathered_muon_slices(candidate_shards)
        owned_updates = tuple(
            reference.add(
                reference.decoupled_weight_decay(
                    reference.slice_rows(weight, j, j + 1), 1e-3, 0.1
                ),
                reference.scale(direction_shards[j], -1e-3),
            )
            for j in range(2)
        )
        self.assertLess(
            reference.max_abs_difference(
                reference.concatenate_rows(owned_updates), serial.weight
            ),
            1e-12,
        )

    def test_scaling_ratio_depends_on_target_loss(self) -> None:
        high_loss = reference.moonlight_compute_ratio(2.4)
        low_loss = reference.moonlight_compute_ratio(2.2)
        self.assertNotAlmostEqual(high_loss, low_loss)
        self.assertTrue(0.0 < high_loss < 1.0)
        self.assertTrue(0.0 < low_loss < 1.0)

    def test_spectral_entropy_extremes(self) -> None:
        self.assertAlmostEqual(reference.normalized_spectral_entropy((1, 1, 1, 1)), 1.0)
        self.assertAlmostEqual(reference.normalized_spectral_entropy((1, 0, 0, 0)), 0.0)
        expected = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / math.log(2)
        self.assertAlmostEqual(reference.normalized_spectral_entropy((2, 1)), expected)

    def test_complete_two_step_hybrid_trace(self) -> None:
        trace = reference.running_hybrid_trace()
        checkpoints = trace["checkpoints"]
        self.assertEqual(len(checkpoints), 12)
        self.assertEqual(trace["final_state_elements"], 315)

        expected_owners = {
            "W_Q": "Muon",
            "W_up": "Muon",
            "W_route": "Muon",
            "E_tok": "AdamW",
            "W_out": "AdamW",
            "gamma": "AdamW",
        }
        for name, owner in expected_owners.items():
            rows = [row for row in checkpoints if row["parameter"] == name]
            self.assertEqual([row["step"] for row in rows], [1, 2])
            self.assertEqual({row["optimizer"] for row in rows}, {owner})
            self.assertTrue(all(row["parameter_delta_norm"] > 0 for row in rows))
            self.assertTrue(
                all(math.isfinite(row["parameter_frobenius_norm"]) for row in rows)
            )

        expected_checksums = {
            "W_Q": 1.2475457747090613,
            "W_up": 7.990072736365771,
            "W_route": -7.999617923476308,
            "E_tok": 23.90246041838307,
            "W_out": -11.952131236244163,
            "gamma": 3.9992000400000003,
        }
        for name, expected in expected_checksums.items():
            self.assertAlmostEqual(trace["final_parameter_checksums"][name], expected)


if __name__ == "__main__":
    unittest.main()
