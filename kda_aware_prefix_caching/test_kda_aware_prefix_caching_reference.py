"""Independent expected values and deliberately unsafe event sequences."""

from dataclasses import replace
from itertools import combinations, permutations
import unittest

from kda_aware_prefix_caching.reference import (
    Job,
    LocalScheduler,
    Page,
    Pool,
    ScalarState,
    advance,
    eligible_boundaries,
)
from kda_aware_prefix_caching.protocol import Bundle, BundleOwner


def fixture() -> tuple[Pool, BundleOwner, Bundle]:
    pool = Pool(640)
    pool.reserve("s8", 192)
    for page in (
        Page("m0", 64, "MLA", 8, initialized=(0, 8)),
        Page("g1", 64, "KDA", 8, "G1"),
        Page("g2", 64, "KDA", 8, "G2"),
    ):
        pool.materialize("s8", page)
    bundle = Bundle(
        "s8",
        ("v7", "N"),
        tuple(map(str, range(8))),
        8,
        1,
        ("G1", "G2"),
        ("m0", "g1", "g2"),
    )
    return pool, BundleOwner(pool, ("G1", "G2")), bundle


class PrefixTests(unittest.TestCase):
    def test_all_small_subsets(self) -> None:
        universe = (2, 4, 6)
        subsets = [set(c) for n in range(4) for c in combinations(universe, n)]
        for mla in subsets:
            for g1 in subsets:
                for g2 in subsets:
                    expected = tuple(
                        b
                        for b in (0, 2, 4, 6)
                        if b == 0 or (b in mla and b in g1 and b in g2)
                    )
                    self.assertEqual(eligible_boundaries(6, 2, mla, (g1, g2)), expected)

    def test_running_scenarios_and_domains(self) -> None:
        mla = set(range(2, 13, 2))
        for g2, expected in (({4, 8}, 8), ({4, 8, 10}, 10), ({4}, 4)):
            self.assertEqual(
                eligible_boundaries(10, 2, mla, ({4, 8, 10}, g2))[-1], expected
            )
        self.assertEqual(eligible_boundaries(0, 2, set(), (set(),)), (0,))
        self.assertEqual(eligible_boundaries(5, 2, {3, 4, 6}, ({3, 4, 6},)), (0, 4))
        with self.assertRaises(ValueError):
            eligible_boundaries(1, 0, set(), (set(),))

    def test_resume_every_split_and_omitted_auxiliary(self) -> None:
        values = (1, 2, 4, 5)
        expected = ((1, 1), (3, 3), (7, 6), (12, 9))
        for boundary in range(5):
            saved, first = advance(ScalarState(), values[:boundary])
            restored, last = advance(saved, values[boundary:])
            self.assertEqual(first + last, expected)
            self.assertEqual(restored, ScalarState(12, 5))
        saved, _ = advance(ScalarState(), (1, 2))
        private, _ = advance(saved, (4,))
        self.assertEqual(saved, ScalarState(3, 2))
        self.assertNotEqual(saved, private)
        _, wrong = advance(ScalarState(3, 0), (4,))
        self.assertEqual(wrong, ((7, 4),))


class ProtocolTests(unittest.TestCase):
    def test_self_declared_group_subset_is_not_complete(self) -> None:
        pool = Pool(128)
        pool.reserve("partial", 128)
        pool.materialize("partial", Page("m", 64, "MLA", 8, initialized=(0, 8)))
        pool.materialize("partial", Page("g", 64, "KDA", 8, "G1"))
        owner = BundleOwner(pool, ("G1", "G2"))
        bundle = Bundle(
            "partial", ("v7", "N"), tuple(map(str, range(8))), 8, 1, ("G1",), ("m", "g")
        )
        with self.assertRaises(ValueError):
            owner.prepare(bundle)
        self.assertEqual(owner.bundles, {})
        self.assertEqual(pool.writers, set())
        self.assertEqual(pool.retained, set())
        self.assertEqual(sum(pool.pins.values()), 0)

    def test_completed_pending_storage_cannot_be_recycled(self) -> None:
        pool, owner, bundle = fixture()
        owner.prepare(bundle)
        for handle in bundle.handles:
            owner.complete(bundle.ident, handle)
        self.assertNotIn(bundle.ident, owner.visible)
        self.assertFalse(pool.release(bundle.ident))
        self.assertEqual(pool.live, 192)
        self.assertTrue(owner.publish(bundle.ident))
        owner.invalidate(bundle.ident)
        self.assertTrue(owner.retire(bundle.ident))
        self.assertEqual(pool.live, 0)

    def test_model_group_contract_domain(self) -> None:
        for groups in ((), ("G1", "G1")):
            with self.assertRaises(ValueError):
                BundleOwner(Pool(0), groups)

    def test_every_copy_completion_order_and_prefix_identity(self) -> None:
        for order in permutations(("m0", "g1", "g2")):
            pool, owner, bundle = fixture()
            owner.prepare(bundle)
            for handle in order[:-1]:
                owner.complete("s8", handle)
                self.assertFalse(owner.publish("s8"))
                self.assertIsNone(owner.acquire("s8", 1, bundle.context, bundle.prefix))
            owner.complete("s8", order[-1])
            self.assertTrue(owner.publish("s8"))
            self.assertIsNone(
                owner.acquire("s8", 1, bundle.context, bundle.prefix[::-1])
            )
            self.assertIsNone(owner.acquire("s8", 1, ("v7", "other"), bundle.prefix))
            self.assertEqual(sum(pool.pins.values()), 0)

    def test_acquired_resume_preserves_snapshot(self) -> None:
        pool, owner, bundle = fixture()
        owner.prepare(bundle)
        for handle in bundle.handles:
            owner.complete("s8", handle)
        owner.publish("s8")
        lease = owner.acquire("s8", 1, bundle.context, bundle.prefix)
        self.assertIsNotNone(lease)
        saved, _ = advance(ScalarState(), tuple(range(1, 9)))
        self.assertEqual(saved, ScalarState(36, 8))
        pool.reserve("B", 128)
        for group in ("G1", "G2"):
            pool.materialize("B", Page("b" + group, 64, "KDA", 8, group))
        result, outputs = advance(saved, (9, 10, 11, 12))
        self.assertEqual(outputs, ((45, 17), (55, 19), (66, 21), (78, 23)))
        self.assertEqual(result, ScalarState(78, 12))
        self.assertEqual(saved, ScalarState(36, 8))
        owner.release_lease(lease)
        self.assertTrue(pool.release("B"))
        self.assertEqual(pool.live, 192)

    def test_readiness_acquisition_invalidation_and_reclamation(self) -> None:
        pool, owner, bundle = fixture()
        owner.prepare(bundle)
        self.assertFalse(owner.publish("s8"))
        owner.complete("s8", "g1")
        owner.complete("s8", "g2")
        self.assertFalse(owner.publish("s8"))
        owner.complete("s8", "m0")
        self.assertTrue(owner.publish("s8"))
        self.assertFalse(pool.release("s8"))
        self.assertIsNone(owner.acquire("s8", 0, bundle.context, bundle.prefix))
        self.assertIsNone(owner.acquire("s8", 1, ("v8", "N"), bundle.prefix))
        self.assertEqual(sum(pool.pins.values()), 0)
        lease = owner.acquire("s8", 1, bundle.context, bundle.prefix)
        self.assertIsNotNone(lease)
        self.assertEqual(sum(pool.pins.values()), 3)
        owner.invalidate("s8")
        self.assertIsNone(owner.acquire("s8", 1, bundle.context, bundle.prefix))
        self.assertFalse(owner.retire("s8"))
        owner.release_lease(lease)
        owner.release_lease(lease)
        self.assertTrue(owner.retire("s8"))
        self.assertEqual(pool.live, 0)

    def test_late_writer_prevents_recycle(self) -> None:
        pool, owner, bundle = fixture()
        owner.prepare(bundle)
        owner.invalidate("s8")
        self.assertFalse(owner.retire("s8"))
        self.assertFalse(pool.release("s8"))
        for handle in bundle.handles:
            owner.complete("s8", handle)
        self.assertFalse(owner.publish("s8"))
        self.assertTrue(owner.retire("s8"))

    def test_bad_group_and_coverage_fail_without_pins(self) -> None:
        pool, owner, bundle = fixture()
        pool.pages["g2"] = replace(pool.pages["g2"], boundary=10)
        with self.assertRaises(ValueError):
            owner.prepare(bundle)
        pool.pages["g2"] = replace(pool.pages["g2"], boundary=8)
        pool.pages["m0"] = replace(pool.pages["m0"], initialized=(0, 6))
        with self.assertRaises(ValueError):
            owner.prepare(bundle)
        self.assertEqual(sum(pool.pins.values()), 0)
        self.assertEqual(pool.writers, set())


class CapacityTests(unittest.TestCase):
    def test_full_timeline_occupancy(self) -> None:
        pool = Pool(832)
        pool.reserve("cache", 448)
        pool.materialize("cache", Page("m0", 64, "MLA", 8, initialized=(0, 8)))
        pool.materialize("cache", Page("m1", 64, "MLA", 12, initialized=(8, 12)))
        for group, boundaries in (("G1", (4, 8, 10)), ("G2", (4, 8))):
            for boundary in boundaries:
                ident = group + str(boundary)
                pool.materialize("cache", Page(ident, 64, "KDA", boundary, group))
        self.assertEqual(pool.live, 448)
        for ident in ("B", "D"):
            self.assertTrue(pool.reserve(ident, 192))
            boundary = 8 if ident == "B" else 0
            for group in ("G1", "G2"):
                pool.materialize(ident, Page(ident + group, 64, "KDA", boundary, group))
            coverage = (8, 12) if ident == "B" else (0, 2)
            pool.materialize(
                ident, Page(ident + "M", 64, "MLA", coverage[1], initialized=coverage)
            )
        self.assertEqual(pool.live, 832)
        pool.release("D")
        self.assertEqual(pool.live, 640)
        pool.release("B")
        self.assertEqual(pool.live, 448)

    def test_foreign_finish_and_pending_cancellation(self) -> None:
        pool = Pool(256)
        pool.reserve("cache", 64)
        scheduler = LocalScheduler(pool)
        with self.assertRaises(ValueError):
            scheduler.finish("cache")
        self.assertEqual(pool.committed, 64)
        scheduler.submit(Job("B", 128, 3))
        scheduler.admit()
        pool.materialize("B", Page("b", 64, "KDA", 8, "G1"))
        pool.writers.add("b")
        self.assertFalse(scheduler.cancel("B"))
        self.assertIsNone(scheduler.step())
        self.assertEqual(pool.committed, 192)
        pool.writers.remove("b")
        self.assertTrue(scheduler.cancel("B"))
        self.assertTrue(scheduler.cancel("B"))
        self.assertEqual(pool.committed, 64)

    def test_fixed_membership_lag(self) -> None:
        scheduler = LocalScheduler(Pool(4))
        for ident in ("A", "B", "C", "D"):
            scheduler.submit(Job(ident, 1, 5))
            scheduler.admit()
        served = dict.fromkeys(("A", "B", "C", "D"), 0)
        for k in range(1, 21):
            ident = scheduler.step()
            served[ident] += 1
            self.assertTrue(all(k / 4 - count < 1 for count in served.values()))
        self.assertEqual(set(served.values()), {5})

    def test_double_admission_materialization_and_release(self) -> None:
        pool = Pool(640)
        self.assertTrue(pool.reserve("cache", 448))
        pool.materialize("cache", Page("cache", 448, "MLA", 12, initialized=(0, 12)))
        self.assertTrue(pool.reserve("B", 128))
        self.assertFalse(pool.reserve("D", 128))
        self.assertEqual(pool.committed, 576)
        pool.materialize("B", Page("b1", 64, "KDA", 8, "G1"))
        self.assertEqual((pool.live, pool.committed), (512, 576))
        self.assertTrue(pool.release("B"))
        self.assertTrue(pool.release("B"))
        self.assertEqual(pool.committed, 448)
        self.assertTrue(pool.reserve("D", 128))

    def test_fifo_and_conditional_cyclic_service(self) -> None:
        scheduler = LocalScheduler(Pool(256))
        scheduler.submit(Job("B", 128, 3))
        scheduler.submit(Job("D", 128, 1))
        self.assertEqual(scheduler.admit(), "admitted")
        self.assertEqual(scheduler.admit(), "admitted")
        self.assertEqual(scheduler.step(), "B")
        self.assertEqual(scheduler.step(), "D")
        self.assertTrue(scheduler.finish("D"))
        self.assertEqual([scheduler.step(), scheduler.step()], ["B", "B"])
        self.assertTrue(scheduler.finish("B"))
        self.assertTrue(scheduler.finish("B"))
        self.assertIsNone(scheduler.step())
        self.assertEqual(scheduler.pool.committed, 0)

    def test_burst_wait_and_oversize_rejection(self) -> None:
        scheduler = LocalScheduler(Pool(128))
        for job in (Job("B", 128, 1), Job("D", 64, 1), Job("X", 129, 1)):
            scheduler.submit(job)
        self.assertEqual(scheduler.admit(), "admitted")
        self.assertEqual(scheduler.admit(), "waiting")
        scheduler.step()
        scheduler.finish("B")
        self.assertEqual(scheduler.admit(), "admitted")
        self.assertEqual(scheduler.admit(), "rejected")


if __name__ == "__main__":
    unittest.main()
