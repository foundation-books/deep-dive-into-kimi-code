"""Counterexamples and boundary tests for the Chapter 32 finite models."""

from dataclasses import replace
import unittest

from agent_correctness_failure_safety_frontiers.reference import (
    Claim,
    Event,
    Grant,
    History,
    Service,
    all_checks,
    claim_admissible,
    classify_run,
    compaction_collision,
    disclosure,
    fold,
    lossy_compaction,
    non_amplifying,
    recovery_decision,
    reliability,
    sufficient_compaction,
    race_schedules,
    verdict,
    zero_harm_upper,
)


class Chapter31ReferenceTests(unittest.TestCase):
    def test_ten_registered_checks(self) -> None:
        checks = all_checks()
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(item["status"] == "pass" for item in checks.values()))

    def test_goal_does_not_erase_earlier_violation(self) -> None:
        vector = classify_run((0, 1, 2, 1), (True, False, True), True, (), True)
        self.assertTrue(vector["goal"])
        self.assertFalse(vector["invariant"])
        self.assertFalse(vector["authority"])
        with self.assertRaises(ValueError):
            classify_run((0, 1), (), True, (), True)

    def test_four_evidence_verdicts(self) -> None:
        self.assertEqual(verdict((0, 1), lambda n: n == 1), "unknown")
        self.assertEqual(verdict((0, 1), lambda n: n <= 1), "supported")
        self.assertEqual(verdict((2, 3), lambda n: n <= 1), "refuted")
        self.assertEqual(verdict((), lambda n: True), "inconsistent")
        self.assertEqual(verdict((), lambda n: False), "inconsistent")
        with self.assertRaises(ValueError):
            verdict((-1,), lambda n: n <= 1)

    def test_replay_is_pure_and_retains_timeout(self) -> None:
        call = Event("call", "c", "allocate", "p", "q")
        events = [call, replace(call, kind="timeout")]
        original = tuple(events)
        service = Service()
        for _ in range(3):
            self.assertEqual(fold(events), (("c",), ()))
        self.assertEqual(tuple(events), original)
        self.assertEqual(service.attempts, 0)
        self.assertEqual(service.jobs, {})

    def test_protocol_rejects_identity_order_and_duplicates(self) -> None:
        call = Event("call", "c", "allocate", "p", "q")
        result = replace(call, kind="result")
        self.assertEqual(fold((call, result)), ((), ("c",)))
        invalid = (
            (result,),
            (call, call),
            (call, result, result),
            (call, replace(result, payload="other")),
            (call, replace(result, request="other")),
            (call, replace(result, tool="tests")),
            (replace(call, call=""),),
            (replace(call, kind="invented"),),
        )
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                fold(events)
        self.assertEqual(fold((call, replace(call, kind="error"))), ((), ("c",)))
        malformed = classify_run((0,), (), False, (replace(call, call=""),), False)
        self.assertFalse(malformed["syntax"])
        self.assertFalse(malformed["protocol"])

    def test_revocation_order_and_expiry_at_commitment(self) -> None:
        service = Service()
        grant = Grant()
        self.assertTrue(grant.valid(service.epoch, 1, "worker", "allocate"))
        service.revoke()
        with self.assertRaises(PermissionError):
            service.commit("q", "p", grant)
        self.assertEqual(service.jobs, {})
        fresh = replace(grant, epoch=5)
        self.assertEqual(service.commit("q", "p", fresh), 1)
        with self.assertRaises(PermissionError):
            service.commit("q2", "p", fresh, now=10)
        child = replace(fresh, actions=frozenset({"query"}), expires=9)
        self.assertTrue(non_amplifying(fresh, child))
        self.assertFalse(non_amplifying(fresh, replace(child, expires=11)))
        self.assertFalse(
            non_amplifying(fresh, replace(child, actions=frozenset({"publish"})))
        )
        other_worker = replace(child, principal="worker-b")
        self.assertFalse(non_amplifying(fresh, other_worker))
        delegated = replace(fresh, delegates=frozenset({"worker-b"}))
        self.assertTrue(non_amplifying(delegated, other_worker))
        self.assertFalse(
            non_amplifying(
                delegated, replace(other_worker, delegates=frozenset({"worker-c"}))
            )
        )

    def test_confidentiality_requires_pair(self) -> None:
        self.assertNotEqual(disclosure(0, True), disclosure(1, True))
        self.assertEqual(disclosure(0, False), disclosure(1, False))
        with self.assertRaises(ValueError):
            disclosure(2, True)

    def test_compaction_counterexample_and_repair(self) -> None:
        histories = (History("q", "timeout"), History(None, "initial"))
        self.assertEqual(compaction_collision(histories, lossy_compaction), (0, 1))
        self.assertIsNone(compaction_collision(histories, sufficient_compaction))
        self.assertEqual(sufficient_compaction(histories[0]), ("q", "timeout"))

    def test_unknown_effect_and_idempotency_scope(self) -> None:
        service = Service()
        first = service.commit("q", "payload1", Grant())
        self.assertEqual(service.commit("q", "payload1", Grant()), first)
        self.assertEqual(len(service.jobs), 1)
        self.assertEqual(service.attempts, 2)
        with self.assertRaises(ValueError):
            service.commit("q", "payload2", Grant())
        self.assertEqual(recovery_decision("not_found_yet", 5, 0), "query")
        self.assertEqual(recovery_decision("unknown", 7, 2), "stop")
        self.assertEqual(recovery_decision("unknown", 12, 1), "stop")
        self.assertEqual(
            recovery_decision("absent_final", 7, 2), "consider_fresh_admission"
        )
        self.assertEqual(recovery_decision("committed", 12, 2), "use_receipt")
        with self.assertRaises(ValueError):
            recovery_decision("unknown", 1, 2)

    def test_all_declared_interleavings(self) -> None:
        split = race_schedules()
        self.assertEqual(len(split), 6)
        self.assertEqual(split[("read_a", "read_b", "act_a", "act_b")], 2)
        self.assertEqual(split[("read_a", "act_a", "read_b", "act_b")], 1)
        self.assertEqual(sum(n == 2 for n in split.values()), 4)
        self.assertEqual(set(race_schedules(True).values()), {1})

    def test_reliability_and_exact_binomial_endpoint(self) -> None:
        chain, bound = reliability(100, 0.99)
        self.assertAlmostEqual(chain, 0.3660323412732292)
        self.assertEqual(bound, 1)
        upper = zero_harm_upper(20)
        self.assertAlmostEqual((1 - upper) ** 20, 0.05)
        self.assertAlmostEqual(upper, 0.13910834066826516)
        self.assertEqual(reliability(1, 1), (1, 0))
        self.assertEqual(reliability(1, 0), (0, 1))
        for cases, alpha in ((0, 0.05), (20, 0), (20, 1)):
            with self.assertRaises(ValueError):
                zero_harm_upper(cases, alpha)

    def test_claim_admission_preserves_quantifier_and_unknowns(self) -> None:
        claim = Claim(
            "prefix",
            "trace",
            ("task", "service"),
            ("task", "service"),
            ("complete covered-effect model",),
            ("q",),
            ("q",),
        )
        self.assertTrue(claim_admissible(claim))
        self.assertFalse(
            claim_admissible(replace(claim, quantifier="all_future_attacks"))
        )
        self.assertFalse(claim_admissible(replace(claim, quantifier="finite_model")))
        self.assertFalse(claim_admissible(replace(claim, versions=("task",))))
        self.assertFalse(claim_admissible(replace(claim, retained_unknowns=())))
        self.assertFalse(claim_admissible(replace(claim, assumptions=())))

    def test_combined_incident_stops_without_certifying_goal(self) -> None:
        service = Service()
        stale = Grant()
        service.revoke()
        with self.assertRaises(PermissionError):
            service.commit("q31", "r1-preview", stale)
        fresh = replace(stale, epoch=service.epoch)
        service.commit("q31", "r1-preview", fresh)
        call = Event("call", "c5", "allocate", "r1-preview", "q31")
        observed = (call, replace(call, kind="timeout"))
        durable = sufficient_compaction(History("q31", "service-timeout"))
        self.assertEqual(durable, ("q31", "service-timeout"))
        self.assertEqual(fold(observed), (("c5",), ()))
        actions, recoveries = 5, 0
        while recovery_decision("unknown", actions, recoveries) == "query":
            actions += 1
            recoveries += 1
        self.assertEqual((actions, recoveries), (7, 2))
        self.assertEqual(len(service.jobs), 1)
        self.assertEqual(service.attempts, 2)  # one denied, one committed
        self.assertEqual(verdict((0, 1), lambda n: n == 1), "unknown")
        self.assertEqual(verdict((0, 1), lambda n: n <= 1), "supported")
        vector = classify_run((0, 1), (True,), False, observed, True)
        self.assertEqual(
            vector,
            dict(
                syntax=True,
                protocol=True,
                authority=True,
                goal=False,
                invariant=True,
                progress=True,
            ),
        )
        # A terminal error changes the exchange ledger, not the possible effects.
        errored = (call, replace(call, kind="error"))
        self.assertEqual(fold(errored), ((), ("c5",)))
        self.assertEqual(verdict((0, 1), lambda n: n == 0), "unknown")
        claim = Claim(
            "prefix",
            "trace",
            ("q31.1",),
            ("q31.1",),
            ("complete covered-effect model",),
            ("q31",),
            ("q31",),
        )
        self.assertTrue(claim_admissible(claim))
        self.assertFalse(claim_admissible(replace(claim, retained_unknowns=())))


if __name__ == "__main__":
    unittest.main()
