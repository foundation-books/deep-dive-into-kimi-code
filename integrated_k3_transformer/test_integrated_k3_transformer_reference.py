"""Tests for the Chapter 10 integrated K3 reference contracts."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import unittest

from . import reference as REFERENCE


class ScheduleAndCoordinateTests(unittest.TestCase):
    def test_source_resolved_full_schedule(self) -> None:
        summary = REFERENCE.full_scale_schedule_summary()
        self.assertEqual(summary["layers"], 93)
        self.assertEqual(summary["kda_layers"], 69)
        self.assertEqual(summary["global_layers"], 24)
        self.assertEqual(summary["transformation_points"], 186)
        self.assertEqual(summary["depth_partition"], (12,) * 7 + (9,))
        self.assertEqual(summary["terminal_sources"], 9)
        self.assertEqual(summary["channel_schedule"][0], "D")
        self.assertTrue(all(kind == "M" for kind in summary["channel_schedule"][1:]))

    def test_wrong_order_and_missing_terminal_are_rejected(self) -> None:
        validate = REFERENCE.validate_schedule
        wrong = list(REFERENCE.terminal_global_schedule(2))
        wrong[2], wrong[3] = wrong[3], wrong[2]
        with self.assertRaises(ValueError):
            validate(wrong)
        with self.assertRaises(ValueError):
            validate(REFERENCE.terminal_global_schedule(2)[:-1])


class InputAndForwardTests(unittest.TestCase):
    def test_printed_integration_fixture_checkpoints(self) -> None:
        """Keep the reader's worked row and actual routes tied to the oracle."""

        trace = REFERENCE.running_prefill()
        first = trace.checkpoints[0]
        second = trace.checkpoints[1]
        self.assertEqual(REFERENCE.running_tokens()[0].values[0], Fraction(-4, 5))
        self.assertEqual(first.attention_output[0][0], Fraction(-79, 100))
        self.assertEqual(first.channel_output[0][0], Fraction(-1873, 5040))
        self.assertEqual(second.attention_output[0][0], Fraction(-1559, 1512))
        self.assertEqual(
            tuple(
                (record.expert, record.ordinal, record.weight)
                for record in second.routes
            ),
            tuple(
                (expert, ordinal, weight)
                for pair in ((1, 0), (2, 1), (3, 2), (0, 3)) * 2 + ((1, 0), (2, 1))
                for ordinal, (expert, weight) in enumerate(
                    zip(pair, (Fraction(2, 3), Fraction(1, 3)), strict=True),
                    start=1,
                )
            ),
        )
        self.assertTrue(
            all(
                len(record) == 2
                for state in trace.prefix_state.global_layers
                for record in state.records
            )
        )
        self.assertAlmostEqual(
            float(trace.terminal_grid[-1][0]), -1020.918687, places=6
        )
        self.assertAlmostEqual(float(trace.logits[-1][0]), -274.414292, places=6)
        self.assertAlmostEqual(float(trace.logits[-1][3]), 2764.769638, places=6)

    def test_running_prefix_conserves_origins(self) -> None:
        tokens = REFERENCE.running_tokens()
        self.assertEqual(len(tokens), 10)
        self.assertEqual(sum(token.modality == "visual" for token in tokens), 4)
        self.assertEqual(sum(token.modality == "text" for token in tokens), 6)
        self.assertEqual(tuple(token.position for token in tokens), tuple(range(1, 11)))
        self.assertTrue(all(token.media_id == 1 for token in tokens[:4]))
        self.assertTrue(all(token.media_id is None for token in tokens[4:]))

    def test_shape_valid_origin_swap_is_rejected(self) -> None:
        swapped = REFERENCE.splice_running_prefix(swap_visual_halves=True)
        state = REFERENCE.empty_prefix_state()
        self.assertEqual(
            tuple(token.position for token in swapped), tuple(range(1, 11))
        )
        REFERENCE.validate_input_segment(swapped, state)
        with self.assertRaises(ValueError):
            REFERENCE.validate_origin_sequence(
                swapped,
                REFERENCE.RUNNING_LABELS,
            )

    def test_integrated_primary_and_oracle_match(self) -> None:
        result = REFERENCE.integrated_forward_equivalence()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["terminal_equal"])
        self.assertTrue(result["logits_equal"])
        self.assertTrue(result["routes_equal"])
        self.assertEqual(result["checkpoints"], 9)

    def test_dense_exception_and_route_conservation(self) -> None:
        trace = REFERENCE.running_prefill()
        ledger = REFERENCE.route_ledger(trace)
        self.assertEqual(ledger["dense_layers"], 1)
        self.assertEqual(ledger["routed_layers"], 8)
        self.assertEqual(ledger["route_records"], 160)
        self.assertEqual(ledger["expected_route_records"], 160)
        self.assertEqual(trace.checkpoints[0].routes, ())
        self.assertTrue(all(len(step.routes) == 20 for step in trace.checkpoints[1:]))

    def test_partial_tail_reaches_terminal_aggregation(self) -> None:
        trace = REFERENCE.running_prefill()
        self.assertEqual(trace.checkpoints[0].depth_points_after_layer, (1, 2))
        self.assertEqual(
            trace.checkpoints[3].depth_points_after_layer,
            tuple(range(1, 9)),
        )
        self.assertEqual(trace.checkpoints[-1].depth_points_after_layer, (17, 18))
        self.assertEqual(
            trace.terminal_sources,
            ("embedding", "block-1", "block-2", "partial-3"),
        )
        self.assertEqual(REFERENCE.depth_partition(9, 4), (4, 4, 1))


class PrefixStateTests(unittest.TestCase):
    def test_cached_decode_matches_full_prefix_exactly(self) -> None:
        result = REFERENCE.incremental_equivalence()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["exact"])
        self.assertEqual(result["prefill_endpoint"], 10)
        self.assertEqual(result["cached_endpoint"], 11)

    def test_rejected_proposal_does_not_commit(self) -> None:
        prefill = REFERENCE.running_prefill()
        trace, state = REFERENCE.propose_token(
            prefill.prefix_state,
            REFERENCE.decode_token(),
            accept=False,
        )
        self.assertEqual(trace.prefix_state.endpoint, 11)
        self.assertEqual(state, prefill.prefix_state)
        self.assertEqual(state.endpoint, 10)

    def test_stale_endpoint_is_rejected(self) -> None:
        prefill = REFERENCE.running_prefill()
        stale = replace(prefill.prefix_state.kda[0], endpoint=9)
        broken = replace(
            prefill.prefix_state,
            kda=(stale,) + prefill.prefix_state.kda[1:],
        )
        with self.assertRaises(ValueError):
            REFERENCE.validate_prefix_state(broken)

    def test_foreign_prefix_identity_is_rejected(self) -> None:
        prefill = REFERENCE.running_prefill()
        foreign = replace(
            prefill.prefix_state,
            accepted_labels=prefill.prefix_state.accepted_labels[:-1]
            + ("text:foreign",),
        )
        with self.assertRaises(ValueError):
            REFERENCE.validate_prefix_state(foreign)
        with self.assertRaises(ValueError):
            REFERENCE.run_forward(
                (REFERENCE.decode_token(),),
                state=foreign,
            )

    def test_all_lifetime_faults_are_detected(self) -> None:
        faults = REFERENCE.state_lifetime_faults()
        self.assertEqual(len(faults), 4)
        self.assertTrue(all(faults.values()))


class LedgerAndPrecisionTests(unittest.TestCase):
    def test_context_and_logical_state_ledgers(self) -> None:
        context = REFERENCE.context_ledger(text=6, visual=4, accepted=1)
        self.assertEqual(context["used"], 11)
        self.assertEqual(context["remaining"], 1_048_565)
        state_10 = REFERENCE.logical_state_ledger(10)
        state_11 = REFERENCE.logical_state_ledger(11)
        self.assertEqual(state_10["kda_fixed_elements"], 48)
        self.assertEqual(state_10["global_growing_elements"], 60)
        self.assertEqual(
            state_11["total_logical_elements"] - state_10["total_logical_elements"],
            6,
        )

    def test_precision_range_contract(self) -> None:
        result = REFERENCE.precision_contract()
        self.assertEqual(result["cumulative_log_decay_open_interval"], (-80, 0))
        self.assertTrue(result["within_bf16_dynamic_range"])

    def test_operator_local_precision_sensitivity_is_conditioned(self) -> None:
        result = REFERENCE.precision_sensitivity(64)
        self.assertEqual(result["steps"], 64)
        self.assertTrue(result["all_finite"])
        self.assertGreaterEqual(result["maximum_absolute_error"], 0.0)
        self.assertTrue(result["range_contract"]["within_bf16_dynamic_range"])

    def test_every_correctness_check_passes(self) -> None:
        checks = REFERENCE.all_correctness_checks()
        self.assertGreaterEqual(len(checks), 14)
        self.assertTrue(all(checks.values()), checks)


if __name__ == "__main__":
    unittest.main()
