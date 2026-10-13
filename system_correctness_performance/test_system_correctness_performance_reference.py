import unittest
from dataclasses import replace
from fractions import Fraction as F
from itertools import pairwise, permutations

from system_correctness_performance.reference import (
    cohort_metrics,
    directional_difference,
    emitted_law,
    error_bound,
    latency,
    peak_live,
    quantile,
    recovery_start,
    resource_bounds,
    routed_outputs,
    tolerance_accepts,
    total_variation,
)
from system_correctness_performance.state import (
    PROMPT,
    DurableFixture,
    compatible,
    crash_fixture,
    materialize,
    prepare_suffix,
)
from system_correctness_performance.trace import (
    critical_path,
    reduce_witness,
    violations,
)


class NumericalTests(unittest.TestCase):
    def test_near_zero_and_directed_relation(self):
        self.assertTrue(tolerance_accepts((0, 1), (5e-7, 1.0005), 1e-6, 1e-3))
        self.assertFalse(tolerance_accepts((0, 1), (5e-7, 1.0005), 0, 1e-3))
        self.assertFalse(tolerance_accepts((2,), (3.5,), 0, 0.5))
        self.assertTrue(tolerance_accepts((3.5,), (2,), 0, 0.5))
        self.assertTrue(tolerance_accepts((0,), (0.75,), 1, 0))
        self.assertTrue(tolerance_accepts((0.75,), (1.5,), 1, 0))
        self.assertFalse(tolerance_accepts((0,), (1.5,), 1, 0))

    def test_exceptional_values_shape_and_empty(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.assertFalse(tolerance_accepts((value,), (value,), 1, 1))
        self.assertFalse(tolerance_accepts((1,), (), 1, 1))
        self.assertTrue(tolerance_accepts((), (), 0, 0))
        self.assertFalse(tolerance_accepts((-1e308,), (1e308,), 0, 1.9))
        with self.assertRaises(ValueError):
            tolerance_accepts((1,), (1,), -1, 0)

    def test_accumulation_and_margin_counterexample(self):
        self.assertAlmostEqual(error_bound([0.5] * 4, [0.001] * 4), 0.001875)
        self.assertAlmostEqual(error_bound([1] * 4, [0.001] * 4), 0.004)
        self.assertEqual(error_bound([], [], 2), 2)
        self.assertEqual(error_bound([2, 3], [1, 4], 5), 37)
        z, candidate = (0.001, 0), (0, 0.001)
        self.assertTrue(tolerance_accepts(z, candidate, 0.001, 0))
        self.assertNotEqual(z.index(max(z)), candidate.index(max(candidate)))

    def test_routes_permutation_shared_bug_and_gradient(self):
        xs, routes = (2, 3), ((0.25, 0.75), (0.5, 0.5))
        self.assertEqual(routed_outputs(xs, routes), (5.5, 7.5))
        self.assertEqual(routed_outputs(xs[::-1], routes[::-1])[::-1], (5.5, 7.5))
        shared_bug = (4, 6)
        self.assertFalse(tolerance_accepts((5.5, 7.5), shared_bug, 1e-6, 0))
        for step in (1e-1, 1e-2, 1e-3):
            self.assertAlmostEqual(
                directional_difference(xs, (1, -1), routes, step), -3.625, places=8
            )
        self.assertEqual(routed_outputs((), ()), ())

    def test_finite_laws_and_zero_proposal_support(self):
        p, q = (F(3, 5), F(2, 5)), (F(1, 2), F(1, 2))
        self.assertEqual(total_variation(p, q), F(1, 10))
        for draft in (q, p, (F(1), F(0)), (F(0), F(1))):
            self.assertEqual(emitted_law(p, draft), p)
        # Two exact fair samplers use different couplings on the same tape.
        tape = [F(1, 8), F(3, 8), F(5, 8), F(7, 8)]
        first = [u < F(1, 2) for u in tape]
        second = [1 - u < F(1, 2) for u in tape]
        self.assertEqual(sum(first), sum(second))
        self.assertNotEqual(first, second)


class StateAndTraceTests(unittest.TestCase):
    def test_commit_and_both_crash_branches(self):
        clean = DurableFixture()
        self.assertTrue(clean.commit(prepare_suffix(clean.record), 8))
        for after_commit in (False, True):
            fixture = crash_fixture(after_commit)
            self.assertEqual(fixture.record.history, clean.record.history)
            self.assertEqual(fixture.record.history[-2:], (1, 4))
            self.assertEqual(len(fixture.record.history), 4098)
            self.assertEqual(fixture.record.cursor, 3)
            self.assertEqual(fixture.record.epoch, 9)
            self.assertTrue(compatible(fixture.record))
            self.assertEqual(fixture.visible, {4096: 1, 4097: 4})
            self.assertFalse(fixture.deliver(8))

    def test_identity_corruptions_are_rejected(self):
        fixture = DurableFixture()
        good = prepare_suffix(fixture.record)
        for field in (
            {"history": PROMPT},
            {"version": "v6"},
            {"preprocessing": "wrong"},
        ):
            bad_object = replace(good.objects[0], **field)
            bad = replace(good, objects=(bad_object, *good.objects[1:]))
            self.assertFalse(compatible(bad))
            with self.assertRaises(ValueError):
                fixture.commit(bad, 8)
        with self.assertRaises(ValueError):
            fixture.commit(replace(good, cache_generation=13), 8)
        self.assertFalse(compatible(replace(good, objects=good.objects[:-1])))

    def test_unknown_outcome_queries_and_stale_effect(self):
        fixture = DurableFixture()
        prepared = prepare_suffix(fixture.record)
        self.assertEqual(fixture.record.history, PROMPT)  # before durable commit
        fixture.commit(prepared, 8)
        self.assertEqual(fixture.record.history, PROMPT + (1, 4))
        recovered = fixture.takeover()
        self.assertFalse(fixture.commit(prepared, 8))
        self.assertTrue(fixture.commit(recovered, 9))
        with self.assertRaises(ValueError):
            fixture.commit(materialize(PROMPT, 0, 9), 9)

    def test_all_independent_completion_interleavings(self):
        chain = ("start", "ready", "consume", "commit")
        edges = tuple(pairwise(chain))
        valid_count = 0
        for order in permutations((*chain, "C")):
            if not violations(order, edges):
                valid_count += 1
                self.assertEqual(tuple(x for x in order if x != "C"), chain)
        self.assertEqual(valid_count, 5)
        broken = ("start", "consume", "ready", "commit", "C")
        self.assertEqual(violations(broken, edges), (("ready", "consume"),))
        reduced = reduce_witness(broken, edges, ("ready", "consume"))
        self.assertEqual(reduced, ("start", "consume", "ready"))
        with self.assertRaises(ValueError):
            violations(("start", "consume", "commit"), edges)

    def test_training_reduction_semantics(self):
        shard_gradients = [1 * x * x for x in (1, 2)]
        self.assertEqual(shard_gradients, [1, 4])
        self.assertEqual(1 - 0.1 * sum(shard_gradients), 0.5)
        self.assertEqual(1 - 0.1 * sum(shard_gradients) / 2, 0.75)


class MeasurementTests(unittest.TestCase):
    def test_timing_populations_and_quantiles(self):
        self.assertEqual(
            latency(0, (50, 70, 95), 100),
            {"ttft": 50, "gaps": (20, 25), "mean_gap": 22.5, "e2e": 100},
        )
        self.assertIsNone(latency(0, (), 10)["ttft"])
        self.assertIsNone(latency(0, (5,), 10)["mean_gap"])
        self.assertEqual(quantile((10, 20, 25, 100), 0.5), 20)
        self.assertEqual(quantile((10, 20, 25, 100), 0.95), 100)
        self.assertIsNone(quantile((), 0.95))
        with self.assertRaises(ValueError):
            latency(0, (5, 4), 10)

    def test_complete_cohort_and_unfinished_outcomes(self):
        rows = [
            (
                str(i),
                "success" if i < 80 else "rejected" if i < 90 else "failed",
                i < 60,
            )
            for i in range(100)
        ]
        expected = {
            "success_rate": 8,
            "goodput": 6,
            "offered_attainment": 0.6,
            "conditional_attainment": 0.75,
        }
        self.assertEqual(cohort_metrics(rows, 10), expected)
        rows[-5:] = [(str(i), "unfinished", False) for i in range(95, 100)]
        self.assertEqual(cohort_metrics(rows, 10), expected)
        with self.assertRaises(ValueError):
            cohort_metrics(rows + [rows[0]], 10)
        with self.assertRaises(ValueError):
            cohort_metrics([("A", "failed", True)], 10)
        self.assertIsNone(cohort_metrics([], 10)["offered_attainment"])

    def test_lifetimes_roofline_and_dependency_bound(self):
        mib = 2**20
        self.assertEqual(peak_live(((0, 2, 4 * mib), (2, 5, 6 * mib))), 6 * mib)
        self.assertEqual(peak_live(((0, 2, 4 * mib), (1, 5, 6 * mib))), 10 * mib)
        self.assertEqual(peak_live(((1, 1, 100),)), 0)
        self.assertEqual(resource_bounds(1e12, 1e10, 2e14, 1e12), (100, 1e14, 0.01))
        self.assertEqual(critical_path({"compute": 3, "transfer": 2}, []), 3)
        self.assertEqual(
            critical_path({"compute": 3, "transfer": 2}, [("compute", "transfer")]), 5
        )
        with self.assertRaises(ValueError):
            critical_path({"a": 1}, [("a", "a")])

    def test_recovery_requires_a_full_observed_window(self):
        samples = [(20, False), (30, False), (40, True), (45, True), (50, True)]
        self.assertEqual(recovery_start(samples, 3), 40)
        self.assertIsNone(recovery_start(samples[:-1], 3))
        self.assertEqual(24 - 20, 4)  # detection is a separate event
        self.assertEqual(40 - 20, 20)


if __name__ == "__main__":
    unittest.main()
