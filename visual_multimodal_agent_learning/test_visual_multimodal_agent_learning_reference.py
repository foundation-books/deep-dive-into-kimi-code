"""Dependency-free tests for the Chapter 25 semantic companion."""

from __future__ import annotations

from fractions import Fraction
import unittest

from . import reference
from .test_visual_loop_investigations import VisualLoopInvestigations
from .test_inspection_task import InspectionTaskTests


class VisualMultimodalAgentReferenceTests(unittest.TestCase):
    def test_inspection_task_boundary(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(InspectionTaskTests)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])

    def test_observation_lineage(self) -> None:
        result = reference.observation_lineage_check()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["duplicate_pixels_remain_distinct"])
        self.assertTrue(result["missing_parent_rejected"])

    def test_coordinate_round_trip(self) -> None:
        result = reference.coordinate_round_trip_check()
        self.assertEqual(result["red_viewport_center"], ["150", "100"])
        self.assertEqual(result["blue_viewport_box_xywh"], ["50", "300", "200", "200"])
        self.assertTrue(result["round_trip_exact"])
        self.assertTrue(result["out_of_support_rejected"])

    def test_visual_diagnostics(self) -> None:
        result = reference.visual_diagnostic_check()
        self.assertEqual(result["box_iou"], "1/3")
        self.assertEqual(result["ocr_score"], "3/4")
        self.assertEqual(result["count_error"], 2)
        self.assertEqual(result["zero_denominator_cases_rejected"], 2)

    def test_zero_vision_denominators(self) -> None:
        result = reference.zero_vision_denominator_check()
        self.assertEqual(result["record_fraction"], "82/83")
        self.assertEqual(result["token_fraction"], "63/64")
        self.assertFalse(result["exact_record_zero_vision"])
        self.assertFalse(result["exact_token_zero_vision"])

    def test_active_perception(self) -> None:
        result = reference.active_perception_check()
        self.assertEqual(result["total_information_cost"], 3)
        self.assertTrue(result["complete_lineage"])
        self.assertTrue(result["state_version_bound"])
        self.assertTrue(result["missing_tool_version_rejected"])

    def test_gui_staleness(self) -> None:
        result = reference.gui_staleness_check()
        self.assertTrue(result["stale_action_rejected"])
        self.assertEqual(result["fresh_action_point"], [90, 80])
        self.assertTrue(result["effect_verified"])

    def test_diagnostic_certificate(self) -> None:
        result = reference.diagnostic_certificate_check()
        self.assertEqual(
            result["remaining_candidates"], ["capture", "grounding", "perception"]
        )
        self.assertFalse(result["unique_cause_claimed"])
        self.assertTrue(result["residual_uncertainty_retained"])

    def test_evidence_handoff(self) -> None:
        result = reference.handoff_check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["invariant_count"], 18)
        self.assertEqual(result["evidence_status_count"], 7)
        self.assertEqual(result["pending_protocol_count"], 8)

    def test_undefined_diagnostic_denominators_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            reference._box_iou((0, 0, 0, 0), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            reference._ocr_score("", "")

    def test_fraction_fixture_is_exact(self) -> None:
        self.assertEqual(Fraction(100, 8_300), Fraction(1, 83))

    def test_visual_loop_investigations(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            VisualLoopInvestigations
        )
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])


if __name__ == "__main__":
    unittest.main()
