"""Deterministic tests for the Chapter 17 companion."""

from __future__ import annotations

import math
import unittest

from . import reference


class K3PostTrainingDataSftReferenceTests(unittest.TestCase):
    def test_valid_record_and_geometry(self) -> None:
        events = reference.valid_events()
        result = reference.validate_record(events, reference.certificate_for(events))
        self.assertTrue(result.accepted)
        fixture = reference.record_fixture()
        self.assertEqual(fixture["red_center_y"], 40)
        self.assertEqual(fixture["blue_center_y"], 100)
        self.assertEqual(fixture["center_delta"], 60)
        self.assertEqual(fixture["call_count"], 2)

    def test_all_eight_frozen_variants_have_exact_outcomes(self) -> None:
        fixture = reference.record_fixture()
        self.assertEqual(
            set(fixture["variants"]), {f"C{index}" for index in range(1, 9)}
        )
        for identifier, expected in reference.EXPECTED_VARIANTS.items():
            realized = fixture["variants"][identifier]
            self.assertEqual(realized["accepted"], expected["accepted"])
            self.assertEqual(realized["code"], expected["code"])
            self.assertEqual(realized["expected_code"], expected["code"])
            self.assertEqual(realized["terminal_success"], expected["terminal_success"])
            self.assertEqual(
                realized["trajectory_validity"], expected["trajectory_validity"]
            )
        self.assertTrue(fixture["variants"]["C7"]["simulator_accepted"])
        self.assertEqual(fixture["variants"]["C8"]["quality_flag"], "LOW_EFFICIENCY")
        self.assertEqual(fixture["variants"]["C8"]["call_count"], 3)
        self.assertEqual(
            set(fixture["recovery_records"]), {"executor_error", "state_refresh"}
        )
        self.assertTrue(
            all(
                record["accepted"] and record["code"] == "ACCEPT"
                for record in fixture["recovery_records"].values()
            )
        )

    def test_serialization_round_trip_and_mask(self) -> None:
        fixture = reference.serialization_fixture()
        self.assertTrue(fixture["round_trip_equal"])
        self.assertTrue(fixture["canonical_stable"])
        self.assertTrue(fixture["mask_valid"])
        self.assertTrue(fixture["user_target_mutation_rejected"])
        self.assertEqual(fixture["event_count"], 6)
        self.assertEqual(fixture["span_count"], 6)
        self.assertEqual(fixture["token_count"], 30)
        self.assertEqual(fixture["supervised_token_count"], 15)
        self.assertEqual(
            fixture["malformed_cases"],
            {
                "duplicate_call_index": "CALL_RESULT_CONSERVATION",
                "unmatched_tool_result": "ORPHAN_RESULT",
                "missing_message_boundary_rejected": True,
            },
        )

    def test_noncanonical_serialization_is_rejected(self) -> None:
        serialized = reference.serialize_events(reference.valid_events())
        with self.assertRaises(ValueError):
            reference.parse_events(serialized + " ")

    def test_weighted_masked_loss(self) -> None:
        fixture = reference.masked_loss_fixture()
        self.assertTrue(
            math.isclose(
                fixture["record_a_normalized"], 1.151292546497023, rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                fixture["record_b_normalized"], 0.8573992140459634, rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                fixture["global_weighted_token_loss"], 1.0925138800068108, rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                fixture["mean_record_normalized_loss"],
                1.0043458802714932,
                rel_tol=1e-12,
            )
        )
        self.assertEqual(
            fixture["weighted_target_units"], {"direct_a": 8, "agent_b": 2}
        )
        self.assertEqual(fixture["record_a_mask"], [0, 1, 1, 0, 1, 1])
        self.assertTrue(fixture["zero_denominator_rejected"])
        with self.assertRaises(ValueError):
            reference.weighted_masked_loss((((0.8, 0.7), (0, 0), 1),))

    def test_six_mixture_ledgers(self) -> None:
        fixture = reference.mixture_fixture()
        self.assertEqual(fixture["record_share"], {"direct": "1/2", "agentic": "1/2"})
        self.assertEqual(
            fixture["serialized_token_share"], {"direct": "1/11", "agentic": "10/11"}
        )
        self.assertEqual(
            fixture["supervised_token_share"], {"direct": "1/6", "agentic": "5/6"}
        )
        self.assertEqual(fixture["action_share"], {"direct": "0/1", "agentic": "1/1"})
        self.assertEqual(
            fixture["approximate_compute_share"], {"direct": "1/17", "agentic": "16/17"}
        )
        self.assertFalse(fixture["source_lineage"]["family_shares_defined"])
        self.assertEqual(
            fixture["source_lineage"]["source_reuse"], {"studio-seed-1": 2}
        )
        self.assertEqual(fixture["raw_totals"]["supervised_tokens"], 60)
        self.assertEqual(fixture["raw_totals"]["actions"], 8)
        self.assertTrue(fixture["empty_denominator_rejected"])

    def test_sequential_error_terms_are_qualified(self) -> None:
        fixture = reference.sequential_error_fixture()
        self.assertEqual(fixture["quadratic_term"], 6.25)
        self.assertEqual(fixture["linear_term"], 0.25)
        self.assertFalse(fixture["terms_are_probabilities"])
        self.assertEqual(len(fixture["recovery_trajectories"]), 2)
        self.assertEqual(
            {record["record_id"] for record in fixture["recovery_trajectories"]},
            {"studio-recovery-schema", "studio-recovery-stale-state"},
        )
        with self.assertRaises(ValueError):
            reference.sequential_error_fixture(0, 0.01)


if __name__ == "__main__":
    unittest.main()
