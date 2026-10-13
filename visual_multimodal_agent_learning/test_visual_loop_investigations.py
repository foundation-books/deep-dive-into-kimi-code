"""Finite Chapter 25 investigations T01--T12."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import os
import unittest

from . import reference
from .crop_source_checks import (
    source_and_preview_details,
    viewport_edges_to_root,
    xywh_to_edges,
)
from .k3_media_binding_checks import (
    automatic_media_from_messages,
    explicit_media_binding,
    probe_pinned_extractor,
)
from .visual_action_mask_checks import probe_pinned_verl, studio_mask_check
from .visual_observation_trace import (
    Event,
    Media,
    action_mask,
    check_policy_request,
    eligible_for_policy_batch,
    terminal_success,
    validate_binding,
)


class VisualLoopInvestigations(unittest.TestCase):
    def setUp(self) -> None:
        fixtures = Path(__file__).with_name("fixtures")
        self.root = Media("i0", (fixtures / "studio-root.png").read_bytes(), 160, 160)
        self.crop = Media("i1", (fixtures / "studio-crop.png").read_bytes(), 300, 600)
        self.other_crop = Media(
            "i4", (fixtures / "studio-distractor-crop.png").read_bytes(), 300, 600
        )

    def test_t01_studio_geometry(self) -> None:
        result = reference.coordinate_round_trip_check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["blue_viewport_center"], ["150", "400"])

    def test_t02_xywh_to_edge_rectangle(self) -> None:
        self.assertEqual(xywh_to_edges((10, 20, 60, 120)), (10, 20, 70, 140))

    def test_t03_path_only_does_not_supply_pixels(self) -> None:
        with self.assertRaisesRegex(ValueError, "without supplied media"):
            check_policy_request(
                "crop saved to /tmp/i1.png", [], protected_terms=["blue-box"]
            )

    def test_t04_correct_media_and_position(self) -> None:
        report = explicit_media_binding(
            "<|kimi_image_placeholder|>",
            [self.crop],
            [self.crop],
            prepared_grid_count=1,
        )
        self.assertEqual(report["image_ids"], ["i1"])
        self.assertEqual(report["dimensions"], [[300, 600]])
        validate_binding([self.crop], [self.crop], prepared_grids=1)
        with self.assertRaisesRegex(ValueError, "grid count"):
            explicit_media_binding(
                "<|kimi_image_placeholder|>",
                [self.crop],
                [self.crop],
                prepared_grid_count=0,
            )

    def test_t05_same_shape_swap_is_detected(self) -> None:
        self.assertEqual(
            (self.crop.width, self.crop.height),
            (self.other_crop.width, self.other_crop.height),
        )
        with self.assertRaisesRegex(ValueError, "identity/order mismatch"):
            validate_binding(
                [self.crop, self.other_crop],
                [self.other_crop, self.crop],
                prepared_grids=2,
            )

    def test_t06_tool_role_requires_declared_adaptation(self) -> None:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": self.root}]},
            {"role": "tool", "content": [{"type": "image", "image": self.crop}]},
        ]
        self.assertEqual(automatic_media_from_messages(messages), [self.root])
        self.assertEqual(
            explicit_media_binding(
                "<|kimi_image_placeholder|>",
                [self.crop],
                [self.crop],
                prepared_grid_count=1,
            )["image_ids"],
            ["i1"],
        )

    def test_t07_action_and_observation_ownership(self) -> None:
        result = studio_mask_check()
        self.assertEqual(result["response_mask"], [1, 1, 1, 1, 0, 0, 0, 1, 1])
        self.assertEqual(result["sampled_positions"], 6)
        self.assertTrue(result["image_conditions_later_answer"])
        with self.assertRaisesRegex(ValueError, "mislabeled"):
            action_mask([Event("i1", "assistant", "image", 3)])

    def test_t08_root_and_viewport_sources_differ(self) -> None:
        edges = (10, 20, 70, 140)
        self.assertEqual(edges, xywh_to_edges((10, 20, 60, 120)))
        self.assertEqual(
            viewport_edges_to_root(
                edges, source_crop=(10, 20, 60, 120), viewport=(300, 600)
            ),
            (12, 24, 24, 48),
        )
        high, preview, width, height = source_and_preview_details()
        self.assertEqual((width, height), (80, 40))
        self.assertNotEqual(high, preview)
        self.assertIn(b"\x0f\x0f\x0f", high)
        self.assertNotIn(b"\x0f\x0f\x0f", preview)

    def test_t09_full_response_budget_does_not_undo_tool(self) -> None:
        result = studio_mask_check()
        self.assertTrue(result["full_budget_blocks_next_decision"])
        self.assertTrue(terminal_success(tool_completed=True, verifier_passed=True))
        self.assertFalse(terminal_success(tool_completed=True, verifier_passed=False))
        self.assertTrue(terminal_success(tool_completed=False, verifier_passed=True))

    def test_t10_oracle_replay_is_not_ordinary_collection(self) -> None:
        ordinary = [Event("call", "assistant", "call", 2)]
        replay = ordinary + [Event("oracle", "assistant", "answer", 2, True)]
        self.assertTrue(eligible_for_policy_batch(ordinary))
        self.assertFalse(eligible_for_policy_batch(replay))
        self.assertEqual(action_mask(replay), [1, 1, 0, 0])

    def test_t11_freshness_and_effect_are_separate(self) -> None:
        result = reference.gui_staleness_check()
        self.assertTrue(result["stale_action_rejected"])
        self.assertTrue(result["effect_verified"])

    def test_t12_protected_truth_and_filename_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected"):
            check_policy_request(
                "blue-box=(20,80,40,40)",
                [self.crop],
                protected_terms=["blue-box"],
            )
        check_policy_request(
            "inspect image i1", [self.crop], protected_terms=["blue-box"]
        )

    def test_frozen_chart_assets_match_generator(self) -> None:
        fixtures = Path(__file__).with_name("fixtures")
        expected = {
            "chart-source.png": "f93abf10bfa325b36def54335501a8932de968a3b74977bbef7cd238e26da53c",
            "chart-source-detail.png": "2b8e846c7675e4ff507bcac47d3fc5232ae46f39fbf4ff8b5fde375729b1bc06",
            "chart-preview-enlarged.png": "a7da40134e446956418b1a486582cf0857d0eefb2f8c8c259dd8603378a8071e",
            "studio-root.png": "4c8a38f6f29172c18379f6e3bb219c101ef4251245a022a9f5980a0ac06b9444",
            "studio-crop.png": "c7f30ee9616e5907e1f2d23714fef76822bca664a2155d4afbeee2e0679270ce",
            "studio-distractor-crop.png": "6e88de59545d14cc3aa65076687da4e9bf48010378c4b254611079308bf4003f",
        }
        for filename, digest in expected.items():
            self.assertEqual(
                sha256((fixtures / filename).read_bytes()).hexdigest(), digest
            )

    @unittest.skipUnless(
        os.environ.get("KIMI_K3_PROCESSOR_FILE"), "pinned processor source unavailable"
    )
    def test_optional_pinned_processor_extractor(self) -> None:
        result = probe_pinned_extractor()
        self.assertTrue(result["user_image_extracted"])
        self.assertFalse(result["tool_image_automatic_extraction"])

    @unittest.skipUnless(
        os.environ.get("KIMI_VERL_CHECKOUT"), "registered veRL checkout unavailable"
    )
    def test_optional_pinned_verl_trainer_mask_handoff(self) -> None:
        result = probe_pinned_verl()
        self.assertEqual(result["revision"], "ea53291385ce764019a2b40733605f21d8317583")
        self.assertIn("ray_trainer", result)


if __name__ == "__main__":
    unittest.main()
