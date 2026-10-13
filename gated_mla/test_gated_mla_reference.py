"""Tests for the Chapter 6 dependency-free Gated MLA reference."""

from __future__ import annotations

from fractions import Fraction
import unittest

from . import reference as REFERENCE


class GatedMlaReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dims = REFERENCE.RUNNING_DIMS
        self.weights = REFERENCE.running_weights()
        self.inputs = REFERENCE.running_inputs()

    def test_running_dimensions(self) -> None:
        self.assertEqual(self.dims.query_projection, 6)
        self.assertEqual(self.dims.compact_record, 3)
        self.assertEqual(self.dims.materialized_record, 10)

    def test_production_dimensions_match_official_config(self) -> None:
        dims = REFERENCE.K3_DIMS
        self.assertEqual((dims.model, dims.heads, dims.query_rank), (7168, 96, 1536))
        self.assertEqual(
            (dims.latent, dims.content_key, dims.shared_key, dims.value),
            (512, 128, 64, 128),
        )
        self.assertEqual(dims.compact_record, 576)
        self.assertEqual(dims.materialized_record, 30720)

    def test_projection_shapes(self) -> None:
        projection = REFERENCE.project_token(self.inputs[0][1], self.weights)
        self.assertEqual(tuple(map(len, projection.query_content)), (2, 2))
        self.assertEqual(tuple(map(len, projection.query_shared)), (1, 1))
        self.assertEqual(len(projection.record.latent), 2)
        self.assertEqual(len(projection.record.shared_key), 1)
        self.assertEqual(tuple(map(len, projection.keys)), (2, 2))
        self.assertEqual(tuple(map(len, projection.values)), (2, 2))
        self.assertEqual(len(projection.gate), 4)

    def test_invalid_weight_shape_fails(self) -> None:
        broken = self.weights.__class__(
            self.weights.q_down[:-1],
            self.weights.q_up,
            self.weights.kv_down,
            self.weights.k_up,
            self.weights.v_up,
            self.weights.gate,
            self.weights.output,
        )
        with self.assertRaises(ValueError):
            REFERENCE.validate_weights(broken)

    def test_reference_and_absorbed_paths_agree(self) -> None:
        reference = REFERENCE.reference_forward(self.inputs, self.weights)
        absorbed = REFERENCE.absorbed_forward(self.inputs, self.weights)
        self.assertTrue(REFERENCE.traces_close(reference, absorbed))

    def test_attention_rows_are_causal_probabilities(self) -> None:
        trace = REFERENCE.absorbed_forward(self.inputs, self.weights)
        for position in trace.positions:
            self.assertEqual(
                tuple(map(len, position.weights)), (position.position,) * 2
            )
            for row in position.weights:
                self.assertAlmostEqual(sum(row), 1.0, places=13)

    def test_exact_key_absorption(self) -> None:
        f = Fraction
        matrix = ((f(1, 2), f(-2, 3)), (f(3, 5), f(4, 7)))
        direct, absorbed = REFERENCE.exact_key_absorption(
            (f(2), f(-1)), matrix, (f(5, 3), f(7, 4))
        )
        self.assertEqual(direct, absorbed)

    def test_exact_value_linearity(self) -> None:
        f = Fraction
        direct, aggregated = REFERENCE.exact_value_linearity(
            (f(1, 3), f(2, 3)),
            ((f(1), f(2)), (f(-1), f(4))),
            ((f(2), f(-1)), (f(3, 2), f(1, 5))),
        )
        self.assertEqual(direct, aggregated)

    def test_gate_blocks_general_fixed_output_fusion(self) -> None:
        first, second = REFERENCE.gate_fusion_counterexample(self.weights)
        self.assertNotEqual(first, second)

    def test_prefill_cache_contract(self) -> None:
        trace, cache = REFERENCE.prefill(self.inputs, self.weights)
        self.assertEqual(len(trace.positions), 10)
        self.assertEqual(cache.identity.endpoint, 10)
        self.assertEqual(len(cache.records), 10)
        REFERENCE.validate_cache(cache)

    def test_decode_continuation_equals_full_evaluation(self) -> None:
        _, cache = REFERENCE.prefill(self.inputs, self.weights)
        decoded, updated = REFERENCE.decode_one(
            REFERENCE.decode_input(),
            cache,
            self.weights,
            expected=cache.identity,
        )
        full = REFERENCE.absorbed_forward(
            self.inputs + (REFERENCE.decode_input(),), self.weights
        )
        self.assertLess(
            REFERENCE.max_abs_difference(decoded.output, full.positions[-1].output),
            1e-11,
        )
        self.assertEqual(updated.identity.endpoint, 11)

    def test_suffix_perturbation_preserves_earlier_outputs(self) -> None:
        baseline, changed = REFERENCE.causal_perturbation(6)
        self.assertEqual(
            tuple(position.output for position in baseline.positions[:6]),
            tuple(position.output for position in changed.positions[:6]),
        )
        self.assertNotEqual(baseline.positions[-1].output, changed.positions[-1].output)

    def test_nope_simultaneous_record_permutation(self) -> None:
        projections = [
            REFERENCE.project_token(hidden, self.weights)
            for _, hidden in self.inputs[:5]
        ]
        label, query = self.inputs[4]
        records = tuple(item.record for item in projections)
        baseline = REFERENCE.absorbed_position(
            label, query, records, self.weights, position=5
        )
        permuted = REFERENCE.absorbed_position(
            label,
            query,
            REFERENCE.permute_records(records, (2, 0, 4, 3, 1)),
            self.weights,
            position=5,
        )
        self.assertLess(
            REFERENCE.max_abs_difference(baseline.output, permuted.output), 1e-11
        )

    def test_invalid_permutation_fails(self) -> None:
        with self.assertRaises(ValueError):
            REFERENCE.permute_records((object(), object()), (0, 0))

    def test_cache_endpoint_fault_fails(self) -> None:
        cache = REFERENCE.new_cache()
        record = REFERENCE.project_token(self.inputs[0][1], self.weights).record
        with self.assertRaises(ValueError):
            REFERENCE.append_record(cache, record, accepted_position=2)

    def test_cache_identity_fault_fails(self) -> None:
        _, cache = REFERENCE.prefill(self.inputs[:2], self.weights)
        wrong = REFERENCE.CacheIdentity("another-request", "toy-v1", 1, 2)
        with self.assertRaises(ValueError):
            REFERENCE.decode_one(
                REFERENCE.decode_input(),
                cache,
                self.weights,
                expected=wrong,
            )

    def test_cache_width_fault_fails(self) -> None:
        record = REFERENCE.CompactRecord((1.0,), (2.0,))
        with self.assertRaises(ValueError):
            REFERENCE.validate_record(record)

    def test_cache_ledger(self) -> None:
        self.assertEqual(REFERENCE.logical_cache_elements(10), 30)
        self.assertEqual(REFERENCE.materialized_cache_elements(10), 100)
        self.assertEqual(REFERENCE.logical_cache_elements(10, REFERENCE.K3_DIMS), 5760)


if __name__ == "__main__":
    unittest.main()
