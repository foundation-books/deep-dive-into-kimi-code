"""Independent expected traces and adversarial publication cases."""

from dataclasses import replace
import unittest

from . import endpoint_cache, prefetch_stopping, resource_pooling
from . import ssd_pipeline, transfer_qos, transfer_scheduling
from .reference import (
    Context,
    ReceivedObject,
    admit,
    allocated_bytes,
    can_publish,
    can_retire,
    choose_route,
    chunk_schedule,
    forecast,
    goodput,
    layer_schedule,
    occupancy,
    prefix_keys,
    retention_value,
    reusable_boundary,
    route_costs,
    transfer_bound,
)


class ServingReferenceTest(unittest.TestCase):
    def test_prefix_continuity_and_context(self):
        context = Context("repo-demo", "v1", "layout1", "position-zero")
        a = tuple("p1 p2 p3 p4 p5 p6 p7 p8 a1 a2 a3 a4".split())
        b = a[:8] + ("b1", "b2", "b3", "b4")
        c = a[:4] + tuple("c1 c2 c3 c4 c5 c6 c7 c8".split())
        ka, kb, kc = (prefix_keys(tokens, context) for tokens in (a, b, c))
        self.assertEqual(reusable_boundary(kb, set(ka)), 8)
        self.assertEqual(reusable_boundary(kc, set(ka)), 4)
        self.assertEqual(reusable_boundary(kb, {ka[1]}), 0)
        self.assertEqual(prefix_keys(a[:10], context), ka[:2])
        self.assertEqual(prefix_keys((), context), ())
        self.assertNotEqual(ka, prefix_keys(a, replace(context, version="v2")))
        self.assertNotEqual(ka, prefix_keys(a, replace(context, layout="layout2")))
        self.assertNotEqual(ka[1], prefix_keys(("z",) + a[1:], context)[1])

    def test_allocation_and_peak(self):
        self.assertEqual(
            [allocated_bytes(n) for n in (0, 4, 8, 12, 13, 16)],
            [0, 1024, 2048, 3072, 4096, 4096],
        )
        self.assertEqual(occupancy({"live": 3072, "receive": 2048}), 5120)
        self.assertEqual(occupancy({"remaining": 2048, "receive": 2048}), 4096)

    def test_independent_pipeline_timelines(self):
        self.assertEqual(
            layer_schedule(),
            (
                ((0, 3), (3, 5)),
                ((3, 6), (6, 8)),
                ((6, 9), (9, 11)),
                ((9, 12), (12, 14)),
            ),
        )
        self.assertEqual(
            layer_schedule(7),
            (
                ((0, 3), (3, 5)),
                ((3, 6), (6, 8)),
                ((7, 10), (10, 12)),
                ((10, 13), (13, 15)),
            ),
        )
        self.assertEqual(chunk_schedule(), 16)
        self.assertEqual(chunk_schedule(1, 2), 8)

    def test_routes_empty_ties_and_fallback(self):
        self.assertEqual(route_costs(1), {"P1": 18, "P2-local": 16, "P2-remote": 12})
        slow = route_costs(0.1)
        self.assertEqual(slow["P2-remote"], 21)
        all_routes = set(slow)
        self.assertEqual(choose_route(route_costs(1), all_routes), "P2-remote")
        self.assertIsNone(choose_route(slow, all_routes))
        self.assertIsNone(choose_route(route_costs(1), {"P1", "P2-local"}))
        self.assertIsNone(choose_route({}, set()))
        self.assertEqual(choose_route({"b": 12, "a": 12}, {"a", "b"}), "a")

    def test_forecast_and_growth_reservation(self):
        events = ((1, -1), (2, -1), (3, 2), (5, 2))
        self.assertEqual(forecast(3, events, 6), 5)
        self.assertEqual(forecast(3, events, 2), 1)
        self.assertFalse(admit(5, 5, 10, 5, 1, 4))
        self.assertTrue(admit(3, 5, 10, 5, 1, 4))
        self.assertFalse(admit(3, 5, 10, 9, 1, 4))
        self.assertGreater(allocated_bytes(13), 3072)

    def test_complete_protected_publication(self):
        first = ReceivedObject("B1", 7, 7, True, True, 10, True)
        second = replace(first, identity="B2")
        self.assertTrue(can_publish({"B1", "B2"}, (first, second), 5))
        self.assertFalse(can_publish({"B1", "B2"}, (first,), 5))
        self.assertTrue(can_publish({"B1"}, (first,), 5))
        for broken in (
            replace(second, current_generation=8),
            replace(second, complete=False),
            replace(second, compatible=False),
            replace(second, protected_until=5),
            replace(second, quiescent=False),
        ):
            self.assertFalse(can_publish({"B1", "B2"}, (first, broken), 5))
        self.assertFalse(can_publish({"B1"}, (first, first), 5))
        self.assertFalse(can_retire(1, 0, False))
        self.assertFalse(can_retire(0, 1, False))
        self.assertFalse(can_retire(0, 0, True))
        self.assertTrue(can_retire(0, 0, False))

    def test_value_and_populations(self):
        self.assertEqual(transfer_bound(3, 1, (1, 1), 0.75), 5)
        self.assertEqual(transfer_bound(3, 1, (1,), 4), 4)
        self.assertEqual(retention_value(2, 10, 3, 2, 1), 11)
        self.assertEqual(retention_value(2, 10, 10, 2, 1), -3)
        self.assertEqual(
            goodput(((10, 14, 19), (12, 18, 22), ()), (True, True, False), 25), 40
        )
        self.assertEqual(goodput(((12, 16, 20, 24),), (True,), 25), 40)
        self.assertEqual(goodput(((12, 16, 20, 24),), (True,), 23), 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            allocated_bytes(-1)
        with self.assertRaises(ValueError):
            route_costs(0)
        with self.assertRaises(ValueError):
            forecast(0, ((1, -1),), 2)
        with self.assertRaises(ValueError):
            goodput(((2, 1),), (True,), 25)


class ImplementationMechanismTest(unittest.TestCase):
    def test_scores_and_estimate_response(self):
        local = transfer_scheduling.Device("local", 8, 4)
        remote = transfer_scheduling.Device("remote", 0, 1, 2)
        self.assertEqual((local.score(4), remote.score(4)), (3, 8))
        self.assertEqual(replace(local, outstanding=40).score(4), 11)
        rate = transfer_scheduling.update_estimate(1, 3, 0.5)
        self.assertEqual(rate, 2)
        self.assertEqual(replace(remote, bandwidth=rate).score(4), 4)
        self.assertEqual(transfer_scheduling.update_estimate(1, 3, 0), 3)
        self.assertEqual(transfer_scheduling.update_estimate(1, 3, 1), 1)
        with self.assertRaises(ValueError):
            replace(local, bandwidth=0).score(4)

    def test_inverse_shares_probe_and_folded_tail(self):
        devices = (
            transfer_scheduling.Device("local", 8, 4),
            transfer_scheduling.Device("remote", 0, 1, 2),
        )
        allocate = transfer_scheduling.allocate_slices
        self.assertEqual(
            allocate(devices, 17, 4, 4),
            (("local", 4), ("local", 4), ("remote", 4), ("local", 5)),
        )
        self.assertEqual(
            allocate(devices, 17, 4, 4, call_index=100),
            (("local", 4), ("remote", 4), ("local", 4), ("remote", 5)),
        )
        # Conservation must hold for every legal small plan, including folds.
        for count in range(1, 8):
            for tail in range(1, 8):
                total = (count - 1) * 4 + tail
                for call in (1, 100):
                    result = allocate(devices, total, count, 4, call)
                    self.assertEqual(len(result), count)
                    self.assertEqual(sum(n for _, n in result), total)
                    self.assertTrue(all(n > 0 for _, n in result))
        # A probe is a sampling opportunity, not a claim that every NIC is used.
        self.assertEqual(allocate(devices, 4, 1, 4, 100), (("local", 4),))

    def test_nonpreemptive_priority_and_finite_arrivals(self):
        background = tuple(transfer_qos.Slice(f"b{i}", 0, 2) for i in range(10))
        jobs = background + (transfer_qos.Slice("urgent", 1.5, 0),)
        fifo = transfer_qos.schedule(jobs, False)
        priority = transfer_qos.schedule(jobs, True)
        self.assertEqual(fifo[-1], ("urgent", 10, 11))
        self.assertEqual(priority[2], ("urgent", 2, 3))
        self.assertEqual(priority[-1][2], 11)
        self.assertEqual({name for name, _, _ in priority}, {s.name for s in jobs})
        posted = (transfer_qos.Slice("bulk", 0, 2, 10), jobs[-1])
        self.assertEqual(transfer_qos.schedule(posted, True)[-1], ("urgent", 10, 11))
        finite = (transfer_qos.Slice("background", 0, 2),) + tuple(
            transfer_qos.Slice(f"h{i}", i, 0) for i in range(10)
        )
        self.assertEqual(
            transfer_qos.schedule(finite, True)[-1], ("background", 10, 11)
        )

    def test_promotion_selection_and_work(self):
        choose = transfer_qos.promotion
        self.assertEqual(choose((1, 19), 21, 10, False), ((0, 1), 1, 2))
        self.assertEqual(choose((1, 19), 21, 10, True), ((0,), 2, 2))
        self.assertEqual(choose((19, 1), 21, 10, False), ((), 1, 2))
        self.assertEqual(choose((19, 1), 21, 10, True), ((1,), 2, 2))
        self.assertEqual(choose((0, 30), 21, 10, True), ((), 2, 2))
        self.assertEqual(choose((), 21, 10, False), ((), 0, 0))

    def test_prefetch_policy_order_and_reversal(self):
        arrivals = ((2, 8), (5, 12), (10, 16))
        policies = ("best_effort", "timeout", "wait_complete")
        costs = {8: 10, 12: 4, 16: 1}
        results = [
            prefetch_stopping.stopping_trace(p, arrivals, costs) for p in policies
        ]
        self.assertEqual([r.stop for r in results], [3, 6, 10])
        self.assertEqual([r.boundary for r in results], [8, 12, 16])
        self.assertEqual([r.installed for r in results], [4, 7, 11])
        self.assertEqual([r.prefill_ready for r in results], [14, 11, 12])
        self.assertEqual([r.first_token for r in results], [18, 15, 16])
        reverse = {8: 20, 12: 12, 16: 1}
        self.assertEqual(
            [
                prefetch_stopping.stopping_trace(p, arrivals, reverse).first_token
                for p in policies
            ],
            [28, 23, 16],
        )
        early = prefetch_stopping.stopping_trace(
            "timeout", ((1, 8), (2, 12), (4, 16)), costs
        )
        self.assertEqual(early.stop, 4)

    def test_prefetch_holes_ranks_and_deadline(self):
        self.assertEqual(prefetch_stopping.continuous_boundary(4, {8, 16}), 8)
        self.assertEqual(
            prefetch_stopping.joint_boundary(4, ({8, 12, 16}, {8, 12})), 12
        )
        hit = prefetch_stopping.timeout_reached
        self.assertFalse(hit(2.006, 2, 2048, 0.004, 0.002, 0.006))
        self.assertTrue(hit(2.007, 2, 2048, 0.004, 0.002, 0.006))
        # Cap at six milliseconds although the uncapped budget is eight.
        self.assertFalse(hit(0.008, 0, 2048, 0.004, 0.002, 1))
        self.assertTrue(hit(0.009, 0, 2048, 0.004, 0.002, 1))
        hole = prefetch_stopping.stopping_trace("timeout", ((2, 8), (5, 16)), {8: 10})
        self.assertEqual(hole.boundary, 8)
        with self.assertRaises(ValueError):
            prefetch_stopping.stopping_trace(
                "wait_complete", ((2, 8), (5, 16)), {8: 10}
            )

    def test_pool_registration_pressure_and_release(self):
        self.assertEqual(8 * 64, 512)
        pool = resource_pooling.StagingPool(128, 160)
        first, second, overflow = (pool.acquire(n) for n in (64, 64, 32))
        self.assertEqual((first, second, overflow), (0, 1, 2))
        self.assertEqual(
            (pool.arena_used, pool.registered, pool.registrations), (128, 160, 2)
        )
        self.assertIsNone(pool.acquire(1))
        pool.release(2)
        self.assertEqual((pool.registered, pool.unregistrations), (128, 1))
        pool.release(0)
        pool.release(1)
        self.assertEqual(pool.peak_registered, 160)
        bounded = resource_pooling.StagingPool(128, 128)
        bounded.acquire(64)
        bounded.acquire(64)
        self.assertIsNone(bounded.acquire(32))
        bounded.release(0)
        self.assertIsNotNone(bounded.acquire(32))
        self.assertEqual(bounded.registrations, 1)
        internal = resource_pooling.StagingPool(128, 160, internal=96)
        self.assertIsNone(internal.acquire(64, allow_overflow=False))
        self.assertIsNotNone(internal.acquire(64))
        self.assertEqual(internal.registered, 192)
        self.assertEqual(resource_pooling.registration_costs(8, 1, 10, 1), (80, 28))

    def test_bucket_delay_and_restore_dependencies(self):
        buckets = ssd_pipeline.Buckets(8, 3)
        self.assertEqual(buckets.heartbeat((("a", 3), ("b", 3))), ())
        self.assertEqual(buckets.heartbeat(()), ())
        self.assertEqual(buckets.pending, [("a", 3), ("b", 3)])
        self.assertEqual(buckets.heartbeat((("c", 2),)), (("a", "b", "c"),))
        self.assertEqual(buckets.pending, [])
        self.assertEqual(buckets.heartbeat((("d", 5), ("e", 5))), (("d",),))
        self.assertEqual(buckets.pending, [("e", 5)])
        self.assertEqual(buckets.heartbeat((("too-large", 9),)), ())
        count_bound = ssd_pipeline.Buckets(8, 2)
        self.assertEqual(count_bound.heartbeat((("x", 1), ("y", 1))), (("x", "y"),))
        self.assertEqual(
            ssd_pipeline.restore_timeline(),
            (
                ("lookup", 8, 9),
                ("wait for staging", 9, 11),
                ("disk read", 11, 15),
                ("network", 15, 17),
                ("installation", 17, 18),
            ),
        )
        self.assertEqual(ssd_pipeline.restore_timeline(buffer_ready=20)[-1][2], 27)

    def test_preferred_sampled_and_fallback_placement(self):
        a = ssd_pipeline.Segment("A", 64, 0.2)
        b = ssd_pipeline.Segment("B", 64, 0.8)
        choose = ssd_pipeline.choose_segment
        self.assertEqual(choose((a, b), 8, (), ("A", "B"), ()), ("B", "sampled"))
        self.assertEqual(choose((a, b), 8, ("A",), ("A", "B"), ()), ("A", "preferred"))
        full = replace(b, memory_free=0)
        self.assertEqual(choose((a, full), 8, (), ("B",), ("A",)), ("A", "fallback"))
        unknown = replace(a, ssd_free_ratio=None)
        self.assertEqual(choose((unknown, b), 8, (), ("A", "B"), ()), ("A", "sampled"))
        self.assertIsNone(choose((full,), 8, (), ("B",), ("B",)))

    def test_sieve_trace_and_separate_reclamation(self):
        cache = endpoint_cache.Sieve(4)
        self.assertIsNone(cache.evict())
        for name in "ABCD":
            cache.insert(name)
        cache.get("A")
        cache.get("C")
        cache.insert("E")
        self.assertEqual([e.name for e in cache.entries], list("ACDE"))
        self.assertEqual(
            [e.visited for e in cache.entries], [False, True, False, False]
        )
        self.assertEqual(cache.last_scans, 2)
        self.assertEqual([e.name for e in cache.pending], ["B"])
        self.assertIsNone(cache.get("B"))
        cache.get("D")
        cache.insert("F")
        self.assertEqual([e.name for e in cache.entries], list("ACDF"))
        self.assertEqual([e.name for e in cache.pending], list("BE"))
        self.assertEqual(cache.last_scans, 3)
        self.assertEqual(cache.reclaim(set()), ())
        self.assertEqual(cache.reclaim({1}), (1,))
        self.assertEqual([e.name for e in cache.pending], ["E"])

    def test_sieve_singleton_and_reused_peer_name(self):
        cache = endpoint_cache.Sieve(1)
        first = cache.insert("A")
        cache.get("A")
        cache.insert("B")
        self.assertEqual(cache.last_scans, 2)
        new = cache.insert("A")
        self.assertNotEqual(first.serial, new.serial)
        self.assertEqual(cache.reclaim({first.serial}), (first.serial,))
        self.assertIs(cache.get("A"), new)
        with self.assertRaises(ValueError):
            endpoint_cache.Sieve(0)


if __name__ == "__main__":
    unittest.main()
