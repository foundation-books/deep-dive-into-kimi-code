"""Deterministic tests for the Chapter 16 companion."""

from __future__ import annotations

from fractions import Fraction
import unittest

from . import reference
from .reference_joint_stages import (
    accepted_family_shares,
    joint_stage_fixture,
    two_loss_reductions,
)


class NativeMultimodalPretrainingReferenceTests(unittest.TestCase):
    def test_caption_record_has_exact_target_accounting(self) -> None:
        record = reference.caption_record()
        reference.validate_serialized_record(record)
        self.assertEqual(len(record.modality), 48)
        self.assertEqual(record.modality.count("visual"), 16)
        self.assertEqual(sum(item is not None for item in record.coordinate_state), 16)
        self.assertEqual(sum(record.loss_mask), 20)

    def test_all_five_studio_records_have_exact_serializations(self) -> None:
        fixture = reference.serialization_fixture()
        self.assertEqual(fixture["record_names"], ["e_T", "e_C", "e_G", "e_R", "e_V"])
        expected_counts = {
            "e_T": (64, 0, 0, 64),
            "e_C": (48, 16, 12, 20),
            "e_G": (40, 16, 16, 8),
            "e_R": (72, 16, 32, 24),
            "e_V": (64, 16, 36, 12),
        }
        for name, expected in expected_counts.items():
            realized = fixture["records"][name]
            counts = realized["counts"]
            self.assertEqual(
                (
                    counts["length"],
                    counts["visual_positions"],
                    counts["prompt_tokens"],
                    counts["target_positions"],
                ),
                expected,
            )
            self.assertTrue(realized["causal_prefix_valid"])
            serialization = realized["serialization"]
            self.assertEqual(
                {len(values) for values in serialization.values()}, {expected[0]}
            )
        self.assertIsNone(fixture["records"]["e_T"]["realized_grid"])
        self.assertEqual(fixture["records"]["e_V"]["realized_grid"]["frames"], 4)
        self.assertEqual(
            fixture["records"]["e_G"]["source_coordinate_frame"]["width"], 400
        )
        self.assertEqual(fixture["records"]["e_G"]["coordinate_frame"]["width"], 112)
        self.assertEqual(len(fixture["input_checksum"]), 64)
        self.assertEqual(len(fixture["output_checksum"]), 64)
        rerun = reference.serialization_fixture()
        self.assertEqual(fixture["input_checksum"], rerun["input_checksum"])
        self.assertEqual(fixture["output_checksum"], rerun["output_checksum"])

    def test_serialization_faults_are_detected(self) -> None:
        fixture = reference.serialization_fixture()
        self.assertTrue(fixture["visual_targets_rejected"])
        self.assertTrue(fixture["active_null_target_rejected"])
        self.assertTrue(fixture["missing_visual_coordinate_rejected"])
        self.assertTrue(fixture["media_identity_swap_detected"])
        self.assertTrue(fixture["stale_reflected_coordinates_detected"])
        self.assertTrue(fixture["leaked_answer_rejected"])
        self.assertTrue(fixture["evidence_removing_crop_rejected"])
        self.assertEqual(fixture["normalized_box"], ["1/4", "1/5", "9/20", "3/5"])
        self.assertEqual(fixture["reflected_box"], ["220", "40", "300", "120"])
        self.assertEqual(fixture["clipped_crop_local_box"], ["0", "40", "60", "120"])
        self.assertTrue(fixture["clip_and_retain_valid"])
        self.assertFalse(fixture["reject_partial_object_valid"])

    def test_two_family_mixture_uses_distinct_denominators(self) -> None:
        fixture = reference.mixture_fixture()["two_family"]
        self.assertEqual(fixture["record_share"], {"text": "1/2", "visual": "1/2"})
        self.assertEqual(fixture["token_share"], {"text": "1/4", "visual": "3/4"})
        self.assertEqual(fixture["target_share"], {"text": "2/3", "visual": "1/3"})
        self.assertEqual(fixture["compute_share"], {"text": "1/10", "visual": "9/10"})
        self.assertEqual(fixture["global_visual_token_share"], "1/2")

    def test_two_loss_reductions_and_accepted_shares(self) -> None:
        reductions = two_loss_reductions(Fraction(1), Fraction(4))
        self.assertEqual(reductions["equal_record_means"], Fraction(5, 2))
        self.assertEqual(reductions["pooled_active_targets"], Fraction(2))
        self.assertEqual(
            accepted_family_shares(),
            {"first": Fraction(2, 3), "second": Fraction(1, 3)},
        )

    def test_mixture_rejects_unnormalized_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            reference.mixture_ledgers(
                (
                    reference.FamilySpec("a", Fraction(1, 3), 8, 1, 0, 1),
                    reference.FamilySpec("b", Fraction(1, 3), 8, 1, 4, 1),
                )
            )

    def test_mixture_tracks_rejections_and_source_reuse(self) -> None:
        fixture = reference.mixture_fixture()
        self.assertEqual(
            fixture["source_reuse"],
            {
                "counts": {"studio-semantic-source": 5},
                "caps": {"studio-semantic-source": 5},
                "within_caps": True,
            },
        )
        self.assertTrue(fixture["source_cap_violation_rejected"])
        self.assertEqual(
            fixture["rejection_ledger"]["rejected_by_reason"],
            {"decode": 1, "geometry": 1},
        )
        self.assertTrue(fixture["rejection_ledger"]["conserved"])
        with self.assertRaises(ValueError):
            reference.source_reuse_ledger(("undeclared",), {})

    def test_resolution_counts_and_tail_policy(self) -> None:
        fixture = reference.resolution_fixture()
        self.assertEqual(fixture["low"]["encoder_positions"], 64)
        self.assertEqual(fixture["low"]["backbone_positions"], 16)
        self.assertEqual(fixture["high"]["encoder_positions"], 256)
        self.assertEqual(fixture["high"]["backbone_positions"], 64)
        self.assertEqual(fixture["four_frame"]["encoder_positions"], 256)
        self.assertEqual(fixture["four_frame"]["post_temporal_positions"], 64)
        self.assertEqual(fixture["four_frame"]["backbone_positions"], 16)
        self.assertEqual(fixture["five_frame_padded_tail"]["temporal_padding"], 3)
        self.assertFalse(
            fixture["five_frame_padded_tail"]["padding_positions_supervised"]
        )
        self.assertTrue(fixture["four_frame"]["target_valid"])
        self.assertEqual(
            fixture["wide_geometry_to_low_grid"]["realized_dimensions"],
            [112, 112],
        )
        self.assertEqual(fixture["wide_geometry_to_low_grid"]["backbone_positions"], 16)
        self.assertEqual(
            set(fixture["studio_records"]), {"e_T", "e_C", "e_G", "e_R", "e_V"}
        )
        self.assertTrue(fixture["all_studio_realizations_declared"])
        self.assertTrue(fixture["all_studio_targets_valid"])
        self.assertEqual(
            fixture["temporal_negative_cases"]["reversed_target"],
            "square moves left",
        )
        self.assertTrue(
            fixture["temporal_negative_cases"]["repeated_still_target_rejected"]
        )
        self.assertTrue(
            fixture["temporal_negative_cases"]["missing_event_frames_rejected"]
        )
        with self.assertRaises(ValueError):
            reference.grid_ledger(5, 8, 8, temporal_group=4)

    def test_four_stage_budgets_reconcile(self) -> None:
        fixture = reference.stage_fixture()
        self.assertEqual(
            fixture["stage_names"],
            [
                "vision_preparation",
                "joint_acquisition",
                "quality_cooldown",
                "multimodal_long_continuation",
            ],
        )
        self.assertEqual(
            [stage["budgets"] for stage in fixture["stages"]],
            [
                {
                    "records": 20,
                    "backbone_tokens": 960,
                    "targets": 400,
                    "visual_positions": 320,
                    "compute_units": 60,
                },
                {
                    "records": 40,
                    "backbone_tokens": 2160,
                    "targets": 1560,
                    "visual_positions": 320,
                    "compute_units": 80,
                },
                {
                    "records": 40,
                    "backbone_tokens": 2400,
                    "targets": 1600,
                    "visual_positions": 320,
                    "compute_units": 90,
                },
                {
                    "records": 40,
                    "backbone_tokens": 2640,
                    "targets": 1640,
                    "visual_positions": 320,
                    "compute_units": 120,
                },
            ],
        )
        self.assertTrue(fixture["all_intended_observed_budgets_reconciled"])
        self.assertTrue(fixture["all_gate_classes_declared"])
        self.assertTrue(fixture["all_precision_states_declared"])
        self.assertTrue(fixture["all_transition_fields_accounted"])
        self.assertTrue(fixture["all_handoffs_valid"])
        tuple_fields = {
            "pi_s",
            "R_s",
            "F_s",
            "L_s",
            "M_s",
            "Theta_s",
            "Omega_s",
            "B_s",
            "G_s",
        }
        gate_classes = {"entry", "integrity", "stability", "retention", "exit"}
        for stage in fixture["stages"]:
            self.assertEqual(set(stage["tuple_fields"]), tuple_fields)
            self.assertEqual(set(stage["gates"]), gate_classes)
            self.assertEqual(stage["budgets"], stage["observed_budgets"])
            self.assertTrue(stage["budget_reconciled"])
            self.assertTrue(stage["handoff_from_previous_valid"])
            self.assertEqual(
                set(stage["changed_fields_from_previous"])
                | set(stage["fixed_fields_from_previous"]),
                tuple_fields,
            )

    def test_stage_budget_requires_exact_realization(self) -> None:
        with self.assertRaises(ValueError):
            reference.stage_budget(
                {"text": Fraction(1, 3), "video": Fraction(2, 3)}, 10
            )

    def test_joint_from_start_fixture_matches_printed_transition(self) -> None:
        fixture = joint_stage_fixture()
        stages = fixture["stages"]
        self.assertEqual(
            [stage["name"] for stage in stages],
            [
                "initial_joint",
                "joint_acquisition",
                "quality_cooldown",
                "long_cap_continuation",
            ],
        )
        self.assertEqual(
            [stage["family_counts"] for stage in stages],
            [
                {"text": 10, "caption": 5, "grounding": 5},
                {"text": 20, "caption": 10, "grounding": 10},
                {"text": 20, "grounding": 10, "code_render": 10},
                {"text": 20, "code_render": 10, "video": 10},
            ],
        )
        self.assertEqual(
            [stage["budgets"] for stage in stages],
            [
                {
                    "records": 20,
                    "backbone_tokens": 1080,
                    "targets": 780,
                    "visual_positions": 160,
                    "compute_units": 40,
                },
                {
                    "records": 40,
                    "backbone_tokens": 2160,
                    "targets": 1560,
                    "visual_positions": 320,
                    "compute_units": 80,
                },
                {
                    "records": 40,
                    "backbone_tokens": 2400,
                    "targets": 1600,
                    "visual_positions": 320,
                    "compute_units": 90,
                },
                {
                    "records": 40,
                    "backbone_tokens": 2640,
                    "targets": 1640,
                    "visual_positions": 320,
                    "compute_units": 120,
                },
            ],
        )
        self.assertEqual(set(stages[1]["changed_fields"]), {"L_s", "B_s", "G_s"})
        self.assertEqual(
            set(stages[1]["fixed_fields"]),
            {"Pi_s", "R_s", "F_s", "M_s", "Theta_s", "Omega_s"},
        )
        self.assertEqual(stages[0]["retention_status"], "not_applicable")
        self.assertTrue(
            all(stage["observed_training_budgets"] is None for stage in stages)
        )
        self.assertTrue(all(stage["training_gate_verdict"] is None for stage in stages))
        self.assertEqual(stages[3]["media_choices"]["video"], "four_frame_low")


if __name__ == "__main__":
    unittest.main()
