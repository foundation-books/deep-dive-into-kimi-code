"""Tests for the finite Chapter 30 evaluation oracles."""

from __future__ import annotations

from fractions import Fraction
import unittest

from evaluating_reasoning_agentic_systems.reference import (
    all_checks,
    claim_admissible,
    compare_protocols,
    disposition_rates,
    judge_order_consistency,
    paired_effect,
    pareto_relation,
    pass_at_k,
    swarm_accounting,
    validate_record,
)


class Chapter30ReferenceTests(unittest.TestCase):
    def test_all_registered_checks_pass(self) -> None:
        checks = all_checks()
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(check["status"] == "pass" for check in checks.values()))

    def test_pass_at_k_boundaries(self) -> None:
        self.assertEqual(pass_at_k(4, 1, 2), Fraction(1, 2))
        self.assertEqual(pass_at_k(6, 2, 4), Fraction(14, 15))
        self.assertEqual(pass_at_k(5, 5, 3), 1)
        self.assertEqual(pass_at_k(5, 0, 3), 0)
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)

    def test_protocol_comparison_returns_witness(self) -> None:
        left = {"effort": "low", "harness": "H0", "tools": "T0"}
        right = {"effort": "max", "harness": "H0", "tools": "T0"}
        self.assertEqual(
            compare_protocols(left, right, set(left), {"effort"}),
            ("controlled_difference", ["effort"]),
        )

    def test_paired_effect_retains_discordance(self) -> None:
        self.assertEqual(
            paired_effect((1, 0, 1, 0), (1, 1, 0, 0)),
            (Fraction(0), [2, 3]),
        )

    def test_disposition_rates_keep_two_estimands(self) -> None:
        semantic, dispatch, counts = disposition_rates(
            ["pass"] * 7 + ["fail"] * 2 + ["invalid", "timeout", "infrastructure_error"]
        )
        self.assertEqual((semantic, dispatch), (Fraction(7, 9), Fraction(7, 12)))
        self.assertEqual(counts["timeout"], 1)

    def test_judge_tie_policy_is_explicit(self) -> None:
        self.assertEqual(
            judge_order_consistency(("A", "tie", "B", "A"), ("A", "tie", "B", "B")),
            Fraction(3, 4),
        )

    def test_pareto_is_partial(self) -> None:
        self.assertEqual(pareto_relation((1, -10), (1, -12)), "left_dominates")
        self.assertEqual(
            pareto_relation((1, -12), (Fraction(9, 10), -10)), "incomparable"
        )

    def test_swarm_accounting_is_not_wall_time(self) -> None:
        self.assertEqual(
            swarm_accounting(32, 32, 36, 18),
            {"critical_step_speedup": Fraction(16, 9), "work_factor": Fraction(9, 8)},
        )

    def test_claim_gate_rejects_pending_mechanism(self) -> None:
        claim = {
            "system": "S",
            "population": "D",
            "protocol": "P",
            "metric": "m",
            "kind": "mechanism",
        }
        evidence = {**claim, "status": "pending"}
        admitted, reasons = claim_admissible(claim, evidence, True, False)
        self.assertFalse(admitted)
        self.assertEqual(reasons, ["missing_intervention", "evidence_pending"])

    def test_record_validation_reports_fields(self) -> None:
        valid, missing = validate_record({}, {}, {})
        self.assertFalse(valid)
        self.assertIn("system.harness", missing)
        self.assertIn("task.verifier", missing)
        self.assertIn("protocol.evaluator", missing)


if __name__ == "__main__":
    unittest.main()
