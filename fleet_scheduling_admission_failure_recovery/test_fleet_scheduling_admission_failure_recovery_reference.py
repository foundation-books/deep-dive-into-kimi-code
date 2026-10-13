"""Independent finite arithmetic and adversarial ownership checks."""

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path
import unittest
from fractions import Fraction

from .ledger import Contract, Owner
from .control import AdmissionQueue, Membership, Pending
from .reference import (
    WEIGHTS,
    Candidate,
    assignment,
    drain_ready,
    failover_work,
    moments,
    placement,
    recovery_route,
    replay_bound,
    service_bound,
    usable_capacity,
)
from .simulation import fifo_slots, fleet_trace


class FleetReferenceTests(unittest.TestCase):
    def test_fixed_hash_and_membership(self) -> None:
        clusters = ("C0", "C1", "C2", "C3")
        expected = (
            ("C1", "C2"),
            ("C3", "C0"),
            ("C0", "C3"),
            ("C2", "C3"),
            ("C0", "C3"),
            ("C3", "C0"),
        )
        self.assertEqual(tuple(assignment(s, clusters) for s in "ABCDEF"), expected)
        for i in range(200):
            session = f"session-{i}"
            pair = assignment(session, clusters)
            self.assertNotEqual(*pair)
            self.assertEqual(pair, assignment(session, clusters[::-1]))
            added = assignment(session, clusters + ("C4",))
            self.assertIn(added[0], (pair[0], "C4"))
            for removed in clusters:
                reduced = tuple(c for c in clusters if c != removed)
                if pair[0] != removed:
                    self.assertEqual(assignment(session, reduced)[0], pair[0])
        for candidates in ((), ("C0",), ("C0", "C0")):
            with self.assertRaises(ValueError):
                assignment("A", candidates)
        with self.assertRaises(ValueError):
            assignment("A\0B", clusters)

    def test_placement_revalidates_acquired_prefix(self) -> None:
        cold = Candidate("C1")
        pair = ("C0", "C1")
        warm = Candidate("C0", acquired_prefix=90, queue_ms=3)
        self.assertEqual((warm.estimate(), cold.estimate()), (5, 11))
        self.assertEqual(placement((warm, cold), pair, 12, "cost"), "C0")
        busy = Candidate("C0", acquired_prefix=90, queue_ms=12)
        self.assertIsNone(placement((busy, cold), pair, 12))
        self.assertEqual(placement((busy, cold), pair, 12, "cost"), "C1")
        stale_hit = Candidate("C0", acquired_prefix=0, queue_ms=3)
        self.assertEqual(stale_hit.estimate(), 14)
        self.assertEqual(placement((stale_hit, cold), pair, 12, "cost"), "C1")
        self.assertIsNone(placement((Candidate("C0", context="v8"),), pair, 100))
        self.assertIsNone(placement((), pair, 100))

    def test_weighted_oracle_and_enumeration(self) -> None:
        self.assertEqual(failover_work(), {"C1": 110, "C2": 70, "C3": 50})
        self.assertEqual(Fraction(110 * 3, 230), Fraction(33, 23))
        values = [
            sum(
                w
                for w, destination in zip(WEIGHTS, choices, strict=True)
                if destination == 0
            )
            for choices in itertools.product(range(3), repeat=6)
        ]
        mean = Fraction(sum(values), 729)
        variance = sum((value - mean) ** 2 for value in values) / 729
        self.assertEqual((mean, variance), (Fraction(230, 3), Fraction(27800, 9)))
        self.assertEqual(moments(WEIGHTS, 3), (mean, variance))
        self.assertEqual(moments(WEIGHTS, 1), (230, 0))

    def test_replay_endpoints_and_fairness(self) -> None:
        work = (110, 70, 50)
        self.assertEqual(replay_bound(work, (20, 20, 20)), 60)
        self.assertEqual(replay_bound(work, (30, 20, 10)), 50)
        self.assertEqual(replay_bound(work, (20, 20, 20), False), 55)
        self.assertEqual(fifo_slots((90, 20), 20), (5, 6))
        self.assertEqual(fifo_slots((60, 10), 20), (3, 4))
        self.assertEqual(fifo_slots((40, 10), 20), (2, 3))
        self.assertEqual(fifo_slots((50, 50), 50), (1, 2))
        self.assertEqual(service_bound(6, 3, 3, 2), 5)
        for start in range(20):
            for duration in range(40):
                endpoints = sum(
                    start < t <= start + duration for t in range(10, 70, 10)
                )
                self.assertGreaterEqual(30 * endpoints, 3 * max(0, duration - 10))

    def test_memory_race_and_retirement(self) -> None:
        owner = Owner()
        a, b = ("A", 1, 5, "g"), ("B", 1, 5, "g")
        self.assertTrue(owner.prepare(a, 6))
        self.assertFalse(owner.prepare(b, 7))
        self.assertEqual(owner.occupancy(), (4, 6))
        self.assertTrue(owner.prepare(a, 6))
        with self.assertRaises(ValueError):
            owner.prepare(a, 7)
        self.assertFalse(owner.allocate(a, "allocation-A", 2))
        self.assertTrue(owner.commit(a))
        self.assertTrue(owner.allocate(a, "allocation-A", 2))
        self.assertTrue(owner.allocate(a, "allocation-A", 2))
        self.assertEqual(owner.occupancy(), (6, 4))
        self.assertFalse(owner.resize(9))
        owner.retire(a)
        self.assertFalse(owner.release(a, quiescent=False))
        self.assertEqual(owner.occupancy(), (6, 4))
        self.assertTrue(owner.release(a, quiescent=True))
        self.assertTrue(owner.release(a, quiescent=True))
        self.assertEqual(owner.occupancy(), (4, 0))
        self.assertTrue(owner.prepare(b, 7))

    def test_prepared_timeout_and_unique_attempt(self) -> None:
        owner = Owner()
        a, b = ("A", 1, 5, "g"), ("A", 2, 5, "g")
        owner.prepare(a, 1)
        owner.retire(a)
        owner.release(a, quiescent=True)
        self.assertFalse(owner.commit(a))
        owner.prepare(b, 1)
        self.assertTrue(owner.commit(b))
        c = ("A", 3, 5, "g")
        owner.prepare(c, 1)
        self.assertFalse(owner.commit(c))

    def test_budget_boundary_and_duplicate_service(self) -> None:
        owner = Owner()
        key = ("A", 1, 5, "g")
        owner.prepare(key, 1)
        owner.commit(key)
        for slot in (0, 1):
            for quantum in range(5):
                self.assertTrue(
                    owner.dispatch(key, f"{slot}-{quantum}", slot, "L", 0, 10)
                )
            self.assertFalse(owner.dispatch(key, f"extra-{slot}", slot, "L", 0, 1))
        self.assertTrue(owner.dispatch(key, "0-0", 0, "L", 0, 10))
        self.assertFalse(owner.dispatch(key, "backdated", 0, "S", 0, 1))
        self.assertEqual(owner.spent[(0, "L", 0)], 50)
        self.assertEqual(owner.occupancy(), (4, 1))
        with self.assertRaises(ValueError):
            owner.dispatch(key, "0-0", 1, "L", 0, 10)
        owner.advance_epoch(6)
        self.assertFalse(owner.dispatch(key, "new", 2, "L", 0, 1))
        self.assertEqual(owner.spent[(1, "L", 0)], 50)

    def test_output_uncertainty_and_restart(self) -> None:
        owner = Owner()
        old = ("A", 1, 5, "g")
        owner.prepare(old, 1, contract=Contract(output_cap=8))
        owner.commit(old)
        owner.allocate(old, "old-buffer", 1)
        for sequence in range(6):
            self.assertTrue(owner.publish(old, sequence, str(sequence), now=20))
        for sequence in range(5):
            owner.deliver("A", sequence)
        owner = copy.deepcopy(owner)  # restart with complete authoritative state
        owner.advance_epoch(6)
        new = ("A", 2, 6, "g")
        owner.prepare(new, 1, contract=Contract(output_cap=8))
        owner.commit(new)
        self.assertFalse(owner.publish(old, 6, "obsolete", now=20))
        self.assertEqual(owner.deliver("A", 5), "5")
        self.assertEqual(owner.deliver("A", 5), "5")
        self.assertEqual(len(owner.visible), 6)
        with self.assertRaises(ValueError):
            owner.publish(new, 5, "different", now=20)
        self.assertFalse(owner.release(old, quiescent=False))
        self.assertEqual(owner.occupancy(), (5, 1))
        self.assertTrue(owner.publish(new, 6, "continued", now=20))

    def test_immutable_contract_survives_slots_and_retries(self) -> None:
        owner = Owner()
        key = ("A", 1, 5, "g")
        contract = Contract((10, 4, 0), ("L",), 1, 30)
        self.assertTrue(owner.prepare(key, 1, contract=contract))
        self.assertTrue(owner.commit(key))
        self.assertTrue(owner.dispatch(key, "p", 0, "L", 0, 10))
        self.assertFalse(owner.dispatch(key, "extra", 1, "L", 0, 1))
        self.assertFalse(owner.dispatch(key, "class", 1, "S", 1, 1))
        self.assertFalse(owner.dispatch(key, "late", 2, "L", 1, 1))
        self.assertTrue(owner.dispatch(key, "decode", 1, "L", 1, 4))
        self.assertFalse(owner.dispatch(key, "over-cap", 1, "L", 1, 1))
        self.assertTrue(owner.dispatch(key, "decode", 1, "L", 1, 4))
        self.assertEqual(owner.grants[key].served, (10, 4, 0))
        self.assertTrue(owner.publish(key, 0, "It", now=20))
        self.assertFalse(owner.publish(key, 1, "extra", now=20))
        with self.assertRaises(ValueError):
            owner.prepare(key, 1, contract=Contract())
        owner.advance_epoch(6)
        retry = ("A", 2, 6, "g")
        with self.assertRaises(ValueError):
            owner.prepare(retry, 1, contract=Contract(output_cap=2, deadline=30))
        self.assertTrue(owner.prepare(retry, 1, contract=contract))
        self.assertTrue(owner.commit(retry))
        self.assertFalse(owner.dispatch(retry, "late", 2, "L", 0, 1))
        self.assertEqual(owner.deliver("A", 0), "It")
        empty = ("B", 1, 6, "g")
        owner.prepare(empty, 1, contract=contract)
        owner.commit(empty)
        self.assertFalse(owner.publish(empty, 0, "late", now=30))

    def test_trace_also_executes_with_python_optimization(self) -> None:
        program = (
            "import json; "
            "from fleet_scheduling_admission_failure_recovery.simulation "
            "import fleet_trace; "
            "events, owners = fleet_trace(); "
            "print(json.dumps([events, [list(o.occupancy()) "
            "+ [len(o.visible), sum(o.spent.values())] for o in owners.values()]]))"
        )
        outputs = []
        for flags in ([], ["-O"]):
            result = subprocess.run(
                [sys.executable, *flags, "-c", program],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(json.loads(result.stdout))
        self.assertEqual(*outputs)
        self.assertEqual([row[:3] for row in outputs[0][1]], [[4, 0, 3]] * 3)

    def test_recovery_and_membership_gates(self) -> None:
        candidates = (Candidate("C1"), Candidate("C2"))
        args = dict(reconstructable=True, fenced=True)
        self.assertEqual(recovery_route("A", "C1", candidates, **args), "C1")
        self.assertEqual(recovery_route("A", "C1", candidates[1:], **args), "C2")
        self.assertIsNone(recovery_route("A", "C1", (), **args))
        self.assertIsNone(
            recovery_route("A", "C1", candidates, reconstructable=False, fenced=True)
        )
        self.assertIsNone(
            recovery_route("A", "C1", candidates, reconstructable=True, fenced=False)
        )
        self.assertIsNone(recovery_route("A", "C1", candidates, attempt=4, **args))
        self.assertEqual([usable_capacity(t) for t in (20, 40, 60)], [0, 0, 100])
        self.assertFalse(drain_ready(True, 1, True))
        self.assertFalse(drain_ready(True, 0, False))
        self.assertTrue(drain_ready(True, 0, True))

    def test_main_trace_matches_independent_totals(self) -> None:
        events, owners = fleet_trace()
        complete = {
            e["request"]: e["time"] for e in events if e["status"] == "complete"
        }
        self.assertEqual(
            complete,
            {
                "A.r1": 70,
                "B.r1": 50,
                "C.r1": 40,
                "D.r1": 80,
                "E.r1": 60,
                "F.r1": 50,
                "SC1.r1": 20,
                "SC2.r1": 20,
                "SC3.r1": 20,
            },
        )
        for cls, expected in (("R", (230, 0, 0)), ("L", (60, 24, 0)), ("S", (6, 3, 0))):
            work = [e["work"] for e in events if e["cls"] == cls]
            self.assertEqual(tuple(sum(w[i] for w in work) for i in range(3)), expected)
        for owner in owners.values():
            self.assertEqual(owner.occupancy(), (4, 0))
            self.assertEqual(len(owner.visible), 3)


class ControlTests(unittest.TestCase):
    def test_pending_capacity_deadline_and_retry(self) -> None:
        queue = AdmissionQueue(limit=2)
        a = Pending("A", 0, 6)
        self.assertEqual(queue.offer(a, 0), "deferred")
        self.assertEqual(queue.offer(a, 0), "deferred")
        self.assertEqual(queue.offer(Pending("B", 1, 7), 1), "deferred")
        self.assertEqual(queue.offer(Pending("C", 2, 1), 2), "rejected")
        self.assertIsNone(queue.next_ready(2, 5))
        self.assertEqual(queue.next_ready(2, 6), a)
        self.assertEqual(queue.retry(a, 3, False), "waiting_authority")
        self.assertEqual(queue.retry(a, 3, True), "deferred")
        self.assertEqual(queue.retry(a, 4, True), "deferred")
        self.assertIsNone(queue.next_ready(12, 12))
        replacement = queue.next_ready(13, 12)
        assert replacement is not None
        self.assertEqual(replacement.attempt, 2)
        self.assertEqual(queue.retry(a, 14, True), "deferred")
        self.assertEqual(queue.admitted["A"], replacement)
        self.assertEqual(queue.retry(Pending("A", 0, 5), 14, True), "obsolete_attempt")
        self.assertEqual(queue.retry(replacement, 15, True), "deferred")
        self.assertIsNone(queue.next_ready(34, 12))
        third = queue.next_ready(35, 12)
        assert third is not None
        self.assertEqual(queue.retry(third, 36, True), "failed")
        self.assertIsNone(queue.next_ready(201, 16))
        self.assertEqual(queue.outcomes["B"], "expired")
        self.assertEqual(queue.offer(Pending("huge", 201, 13), 201), "rejected")
        self.assertEqual(
            queue.offer(Pending("uncapped", 201, 1, cap=5), 201), "rejected"
        )

    def test_join_and_drain_preserve_existing_owners(self) -> None:
        membership = Membership()
        for name in ("C2", "C4"):
            membership.request(name)
            self.assertFalse(membership.assign(name, "A"))
            membership.install(name, "v7")
        self.assertFalse(membership.activate("C4", False))
        self.assertEqual(membership.version, 1)
        self.assertTrue(membership.activate("C2", True))
        self.assertTrue(membership.assign("C2", "B"))
        membership.drain("C2")
        self.assertFalse(membership.assign("C2", "new"))
        self.assertFalse(membership.retire("C2"))
        self.assertFalse(membership.handoff("C2", "C4", "B", True, True))
        self.assertTrue(membership.activate("C4", True))
        self.assertFalse(membership.handoff("C2", "C4", "B", True, False))
        self.assertTrue(membership.handoff("C2", "C4", "B", True, True))
        self.assertTrue(membership.retire("C2"))
        self.assertEqual(membership.clusters["C4"].owners, {"B"})
        self.assertEqual(membership.version, 5)

    def test_draining_owner_requires_quiescence_before_retirement(self) -> None:
        membership = Membership()
        membership.request("C2")
        membership.install("C2", "v7")
        self.assertTrue(membership.activate("C2", True))
        self.assertTrue(membership.assign("C2", "A"))
        membership.drain("C2")
        self.assertFalse(membership.finish_owner("C2", "A", quiescent=False))
        self.assertEqual(membership.clusters["C2"].owners, {"A"})
        self.assertFalse(membership.retire("C2"))
        self.assertTrue(membership.finish_owner("C2", "A", quiescent=True))
        self.assertTrue(membership.retire("C2"))


if __name__ == "__main__":
    unittest.main()
