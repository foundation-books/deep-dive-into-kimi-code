"""Tests for the Chapter 21 dependency-free semantic companion."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import unittest

from . import reference


class MultiDomainReasoningEffortReferenceTests(unittest.TestCase):
    def test_grid_identity_and_expert_type(self) -> None:
        cells = reference.domain_effort_cells()
        self.assertEqual(len(cells), 9)
        self.assertEqual(len(set(cells)), 9)
        self.assertTrue(reference.validate_cell("a", "H"))
        with self.assertRaises(ValueError):
            reference.validate_cell("a", "H", expert_kind="moe")
        with self.assertRaises(ValueError):
            reference.validate_cell("missing", "L")

    def test_resource_projection_and_units(self) -> None:
        general = reference.ResourceVector(12, 5, 0, 1, 0)
        agent = reference.ResourceVector(14, 8, 14, 3, 2)
        self.assertEqual(reference.token_cost("g", general), 12)
        self.assertEqual(reference.token_cost("a", agent), 36)
        self.assertEqual(agent.assistant_output, 36)
        self.assertEqual(reference.add_like_units(12, "tokens", 5, "tokens"), 17)
        with self.assertRaises(ValueError):
            reference.add_like_units(30, "tokens", 2, "turns")
        with self.assertRaises(ValueError):
            replace(agent, tools=-1).validate()

    def test_budget_boundary_reward_and_provenance(self) -> None:
        baseline = reference.BaselineRecord(
            "studio-general-001", "r1", "cold-g-v1", "teaching", Fraction(12)
        )
        multiplier = reference.MultiplierRecord(
            "g", "L", "stage-L", Fraction(2, 3), "fixture"
        )
        equal = reference.budget_state(
            "studio-general-001",
            "r1",
            "g",
            "L",
            reference.ResourceVector(8, 2, 0, 1, 0),
            baseline,
            multiplier,
        )
        self.assertEqual(equal.relation, "equal")
        self.assertEqual(reference.effective_reward(1, equal), 1)
        over = reference.budget_state(
            "studio-general-001",
            "r1",
            "g",
            "L",
            reference.ResourceVector(9, 2, 0, 1, 0),
            baseline,
            multiplier,
        )
        self.assertEqual(over.overrun, 1)
        self.assertEqual(reference.effective_reward(1, over), -1)
        with self.assertRaises(ValueError):
            reference.budget_state(
                "wrong-task",
                "r1",
                "g",
                "L",
                reference.ResourceVector(8, 2, 0, 1, 0),
                baseline,
                multiplier,
            )
        with self.assertRaises(ValueError):
            replace(baseline, value=Fraction(0)).validate()

    def test_curriculum_and_allocation(self) -> None:
        records = tuple(
            reference.MultiplierRecord(
                "g", effort, f"stage-{effort}", reference.MULTIPLIERS[effort], "fixture"
            )
            for effort in reference.CURRICULUM_ORDER
        )
        self.assertTrue(reference.validate_curriculum(records, 64))
        with self.assertRaises(ValueError):
            reference.validate_curriculum(tuple(reversed(records)), 64)
        rows = (
            reference.AllocationRow(1, 1, 1, 1, 7, 10, 1, 0),
            reference.AllocationRow(1, 1, 0, 1, 8, 18, 2, 1),
        )
        totals = reference.allocation_totals(rows)
        self.assertEqual(totals.rollouts, 2)
        self.assertEqual(totals.output, 28)
        with self.assertRaises(ValueError):
            reference.AllocationRow(0, 1, 0, 0, 0, 0, 0, 0).validate()

    def test_pareto_handoffs_and_six_protocols(self) -> None:
        points = (
            reference.ParetoPoint("short-success", (Fraction(1),), (Fraction(12),)),
            reference.ParetoPoint("long-success", (Fraction(1),), (Fraction(20),)),
            reference.ParetoPoint("short-failure", (Fraction(0),), (Fraction(8),)),
        )
        self.assertEqual(
            reference.pareto_frontier(points), ("short-success", "short-failure")
        )
        trace = reference.studio_trace()
        self.assertEqual(trace["allocation_totals"]["output"], 395)
        self.assertEqual(trace["handoff_count"], 9)
        self.assertEqual(trace["frontiers"]["a"], ("a-L", "a-H"))
        self.assertEqual(len(trace["content_sha256"]), 64)
        checks = reference.all_checks()
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(record["status"] == "pass" for record in checks.values()))


if __name__ == "__main__":
    unittest.main()
