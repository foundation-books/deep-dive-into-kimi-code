"""Tests for the Chapter 23 dependency-free semantic companion."""

from __future__ import annotations

from fractions import Fraction
import math
import unittest

from . import reference


class CapabilityConsolidationReferenceTests(unittest.TestCase):
    def test_nine_routes_are_total_and_unique(self) -> None:
        portfolio = reference.studio_teacher_portfolio()
        self.assertEqual(len(portfolio), 9)
        self.assertEqual(
            reference.route_teacher("studio-agent-001", "high").domain, "agent"
        )
        with self.assertRaises(ValueError):
            reference.route_teacher("missing", "high")

    def test_reverse_kl_reward_identity(self) -> None:
        p = (Fraction(4, 5), Fraction(1, 5))
        q = (Fraction(11, 20), Fraction(9, 20))
        rewards = reference.opd_rewards(p, q)
        expected = sum(
            float(weight) * reward for weight, reward in zip(q, rewards, strict=True)
        )
        self.assertTrue(
            math.isclose(expected, -reference.reverse_kl(q, p), abs_tol=1e-15)
        )
        self.assertTrue(
            math.isinf(reference.reverse_kl((Fraction(1, 2), Fraction(1, 2)), (1, 0)))
        )

    def test_clipping_and_top_k_support(self) -> None:
        rewards = reference.opd_rewards(
            (Fraction(4, 5), Fraction(1, 5)), (Fraction(11, 20), Fraction(9, 20))
        )
        self.assertAlmostEqual(
            reference.clipped_reward(rewards[1], Fraction(7, 10)), -0.7
        )
        projected = reference.top_k_projection(
            (Fraction(3, 5), Fraction(1, 4), Fraction(1, 10), Fraction(1, 20)), 2
        )
        self.assertEqual(projected, (Fraction(12, 17), Fraction(5, 17), 0, 0))
        with self.assertRaises(ValueError):
            reference.clipped_reward(-math.inf, Fraction(7, 10))

    def test_strict_opd_attempt_refuses_late_support_failure(self) -> None:
        token = reference.ScoredOPDToken
        first = token("teacher-a", "student-v1", "prefix-1", "a", 1, Fraction(1, 2))
        second = token("teacher-a", "student-v1", "prefix-2", "b", 0, Fraction(1, 2))
        state = (Fraction(5), Fraction(3), 7, Fraction(1, 10))
        calls: list[tuple[float, ...]] = []

        def updater(
            old: tuple[Fraction, Fraction, int, Fraction],
            coefficients: tuple[float, ...],
        ) -> tuple[Fraction, Fraction, int, Fraction]:
            calls.append(coefficients)
            return (old[0] - 1, old[1] + 1, old[2] + 1, old[3] + 1)

        refused = reference.strict_opd_attempt((first, second), Fraction(4, 5))
        self.assertEqual(refused.status, "refused")
        self.assertEqual(refused.examined_positions, 2)
        self.assertEqual(refused.clipped_rewards, ())
        self.assertEqual(
            refused.failure,
            ("teacher-a", "student-v1", "prefix-2", "b", "structural_teacher_zero"),
        )
        self.assertEqual(
            reference.commit_opd_attempt(refused, state, updater), (state, 0)
        )
        self.assertEqual(calls, [])

        unresolved = reference.strict_opd_attempt(
            (
                first,
                token("teacher-a", "student-v1", "prefix-2", "b", None, Fraction(1, 2)),
            ),
            Fraction(4, 5),
        )
        self.assertEqual(unresolved.failure[-1], "unresolved_score")
        self.assertEqual(
            reference.commit_opd_attempt(unresolved, state, updater), (state, 0)
        )
        self.assertEqual(calls, [])

        admitted = reference.strict_opd_attempt((first,), Fraction(4, 5))
        self.assertEqual(admitted.status, "admitted")
        updated_state, committed_uses = reference.commit_opd_attempt(
            admitted, state, updater
        )
        self.assertEqual(updated_state[2], 8)
        self.assertEqual(committed_uses, 1)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(
            reference.reverse_kl((1, 0), (Fraction(1, 2), Fraction(1, 2))), math.log(2)
        )

    def test_token_use_units_distinguish_reuse_from_unique_samples(self) -> None:
        sampled = tuple(("trajectory-1", index) for index in range(100))
        admitted = sampled[:80]
        single = reference.token_use_counts(sampled, admitted, admitted)
        self.assertEqual((single.sampled, single.admitted, single.uses), (100, 80, 80))
        repeated = reference.token_use_counts(
            sampled + sampled[:1], admitted, admitted * 3
        )
        self.assertEqual(
            (repeated.sampled, repeated.admitted, repeated.uses), (100, 80, 240)
        )
        self.assertEqual(reference.token_use_counts(sampled, admitted, ()).uses, 0)
        self.assertEqual(
            reference.token_use_counts(
                (("a", 0), ("b", 0)), (("a", 0), ("b", 0)), ()
            ).sampled,
            2,
        )
        with self.assertRaises(ValueError):
            reference.token_use_counts(sampled, admitted, (sampled[90],))

    def test_mixture_and_counts_conserve(self) -> None:
        weights = {
            (domain, effort): Fraction(1, 9)
            for domain in reference.DOMAINS
            for effort in reference.EFFORTS
        }
        self.assertTrue(reference.validate_mixture(weights))
        self.assertEqual(reference.CellCounts(12, 1, 2, 9).available, 9)
        with self.assertRaises(ValueError):
            reference.CellCounts(2, 1, 1, 1).validate()

    def test_quantization_contract_and_parity(self) -> None:
        contract = reference.studio_quantization_contract()
        self.assertTrue(contract.validate())
        check = reference.all_checks()["RESULT_CH21_PRECISION_PARITY_ORACLE"]
        self.assertTrue(check["matched"]["eligible"])
        self.assertEqual(check["contract_failure"]["status"], "contract_mismatch")
        self.assertEqual(check["decision_failure"]["status"], "decision_mismatch")

    def test_acceptance_tv_lk_and_recurrence(self) -> None:
        p = (Fraction(3, 5), Fraction(3, 10), Fraction(1, 10))
        q = (Fraction(1, 2), Fraction(2, 5), Fraction(1, 10))
        self.assertEqual(reference.acceptance_mass(p, q), Fraction(9, 10))
        self.assertEqual(reference.total_variation(p, q), Fraction(1, 10))
        self.assertAlmostEqual(reference.lk_loss(p, q), -math.log(0.9))
        outputs = ("blue", "square", "is", "below", "the", "red", "circle")
        self.assertEqual(
            reference.recurrent_inputs("<start>", outputs)[1:], outputs[:-1]
        )

    def test_complete_handoff_and_check_registry(self) -> None:
        handoff = reference.studio_handoff()
        self.assertEqual(handoff["portfolio_cells"], 9)
        self.assertEqual(handoff["measured_model_outcomes"], "pending")
        checks = reference.all_checks()
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(record["status"] == "pass" for record in checks.values()))


if __name__ == "__main__":
    unittest.main()
