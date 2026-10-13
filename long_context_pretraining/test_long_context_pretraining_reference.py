"""Deterministic tests for the Chapter 15 companion."""

from __future__ import annotations

from fractions import Fraction
import unittest

from . import reference


class LongContextPretrainingReferenceTests(unittest.TestCase):
    def test_orbit_answer_only_support_and_coverage(self) -> None:
        record = reference.orbit_record(64)
        self.assertEqual(
            reference.transitive_support(record, 64), frozenset({4, 23, 45})
        )
        self.assertEqual(
            [reference.coverage(record, threshold) for threshold in (16, 32, 60, 61)],
            [Fraction(1), Fraction(1), Fraction(1), Fraction(0)],
        )

    def test_nested_orbit_stage_spans(self) -> None:
        spans = {
            length: reference.dependency_span(
                reference.orbit_record(length), reference.ORBIT_POSITIONS[length][-1]
            )
            for length in reference.ORBIT_POSITIONS
        }
        self.assertEqual(spans, {8: 7, 16: 15, 32: 30, 64: 60})

    def test_dependency_faults_are_detected(self) -> None:
        fixture = reference.dependency_fixture()
        self.assertEqual(fixture["stipulated_graph_spans"], [19, 41, 60])
        self.assertEqual(fixture["deleted_graph_edge_spans"], [19, 0, 19])
        self.assertTrue(fixture["missing_fact_rejected"])
        self.assertTrue(fixture["missing_annotation_rejected"])
        self.assertTrue(fixture["wrong_supplied_label_rejected"])
        self.assertTrue(fixture["future_support_rejected"])

    def test_packing_contracts_remain_semantically_separate(self) -> None:
        fixture = reference.packing_fixture()
        self.assertEqual(len(fixture["policies"]), 3)
        self.assertFalse(fixture["semantic_bundle_created"])
        self.assertTrue(fixture["provenance_fault_rejected"])
        self.assertTrue(fixture["support_dropping_truncation_rejected"])
        self.assertEqual(fixture["physical_atoms"], 16)
        self.assertEqual(
            fixture["first_v_target"]["position_context_reset"], ("v2", "v1")
        )
        self.assertFalse(fixture["v1_loss"]["position_context_reset"])

    def test_mixture_fractions_are_exact(self) -> None:
        fixture = reference.mixture_fixture()
        self.assertEqual(fixture["three_short_one_long"], ["3/11", "8/11"])
        self.assertEqual(fixture["long_pooled_target_share_at_32"], "1/22")
        self.assertEqual(fixture["active_targets"], 22)
        self.assertEqual(fixture["exercise_token_masses"], ["7/87", "16/87", "64/87"])
        self.assertEqual(fixture["long_probability_for_40_percent_tokens"], "23/887")

    def test_curriculum_ledgers_reconcile(self) -> None:
        fixture = reference.curriculum_fixture()
        self.assertEqual(fixture["steps"], [16, 16, 16, 16])
        self.assertEqual(fixture["records_per_step"], [32, 16, 8, 4])
        self.assertEqual(fixture["total_records"], [512, 256, 128, 64])
        self.assertEqual(fixture["answer_only_targets"], [512, 256, 128, 64])
        self.assertEqual(fixture["pairs_per_record"], [36, 136, 528, 2080])
        self.assertEqual(fixture["pairs_per_step"], [1152, 2176, 4224, 8320])
        self.assertEqual(fixture["total_pairs"], [18432, 34816, 67584, 133120])
        self.assertEqual(
            fixture["reduced_final"],
            {
                "steps": 4,
                "total_records": 16,
                "answer_only_targets": 16,
                "total_pairs": 33280,
            },
        )

    def test_nondivisible_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reference.stage_ledger(64, 250, 1000)


if __name__ == "__main__":
    unittest.main()
