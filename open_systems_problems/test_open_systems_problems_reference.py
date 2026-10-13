"""Independent expected values and adversarial variants of the chapter fixtures."""

import unittest
from dataclasses import replace
from fractions import Fraction as F
from itertools import permutations, product

from open_systems_problems.reference import (
    Observation,
    TraceCase,
    amplification,
    cohort_metrics,
    migration_cost,
    online_choice,
    pareto,
    placement,
    readout,
    recurrence,
    resource_floor,
    retention_value,
)
from open_systems_problems.state import EffectStore, Identity, Package, joint_boundary


class SystemsTests(unittest.TestCase):
    def test_bounds_and_feasible_frontier(self):
        self.assertEqual(resource_floor(F(12), F(8), F(6), F(2), F(3)), 4)
        self.assertEqual(resource_floor(F(12), F(4), F(6), F(2), F(3)), 3)
        self.assertEqual(F(4 + 6, 2 + 6), F(5, 4))
        self.assertEqual(pareto(((F(2), F(20)), (F(3), F(10)), (F(4), F(25)))), (0, 1))
        self.assertEqual(pareto(((F(1),), (F(1),))), (0, 1))
        self.assertEqual(pareto(()), ())
        with self.assertRaises(ValueError):
            resource_floor(F(1), F(1), F(0), F(1), F(0))

    def test_joint_boundary_and_identity_variants(self):
        package = Package()
        self.assertEqual(joint_boundary(package, Identity()), 8)
        self.assertEqual(joint_boundary(package, Identity(), 7), 4)
        for changed in (
            replace(Identity(), weights="v1"),
            replace(Identity(), numerical="n1"),
            replace(Identity(), diagram="d1"),
            replace(Identity(), preprocessing="p1"),
            replace(Identity(), prefix="different-text"),
        ):
            self.assertIsNone(joint_boundary(package, changed))
        for broken in (
            replace(package, ready_groups=(frozenset({4, 8}),)),
            replace(package, ready_groups=(frozenset({4}), frozenset({8}))),
            replace(package, ready_groups=(frozenset(), frozenset({8}))),
            replace(package, auxiliary_ready=False),
            replace(package, mla_through=3),
        ):
            self.assertIsNone(joint_boundary(broken, Identity()))

    def test_conditional_value_and_duplicate_work(self):
        args = (F(2), F(10), F(1, 2), F(6, 5), F(1, 100))
        self.assertEqual(retention_value(*args), F(2, 5))
        self.assertEqual(retention_value(F(2), F(10), F(1, 6), F(6, 5), F(1, 100)), 0)
        self.assertEqual(
            retention_value(F(2), F(10), F(0), F(6, 5), F(1, 100)), -F(1, 5)
        )
        # Enumerate one mutually exclusive hit/miss event, not two independent hits.
        outcomes = ((F(1, 2), F(6, 5)), (F(1, 2), F(0)))
        joint_saving = sum(prob * saved for prob, saved in outcomes)
        self.assertEqual(joint_saving - F(2, 5), F(1, 5))
        self.assertNotEqual(joint_saving, 2 * F(3, 5))

    def test_online_information_and_queue_reversal(self):
        obs = Observation((F(3, 2), F(3, 10)), (F(1, 5), F(6, 5)))
        self.assertEqual(placement(obs.queues, obs.service_estimates), (1, F(3, 2)))
        changed = replace(obs, queues=(F(3, 2), F(7, 10)))
        self.assertEqual(
            placement(changed.queues, changed.service_estimates), (0, F(17, 10))
        )
        traces = [TraceCase(obs, F(t), length) for t in (0, 100) for length in (1, 100)]
        self.assertEqual({online_choice(t.observed) for t in traces}, {1})
        self.assertEqual(placement((F(0), F(0)), (F(1), F(1)))[0], 0)

    def test_numerical_counterexample_and_migration_cost(self):
        exact = recurrence((F(2), F(0)))
        rounded = recurrence((F(2), F(0)), F(4))
        self.assertEqual(exact, (2, 1))
        self.assertEqual(rounded, (0, 0))
        self.assertEqual((readout(exact[-1]), readout(rounded[-1])), ("A", "B"))
        self.assertEqual(recurrence((F(6),), F(4)), (8,))
        self.assertEqual(recurrence((F(-2),), F(4)), (0,))
        move = migration_cost(F(2), F(4), F(1, 10), F(0), F(1, 5))
        self.assertEqual(move, F(4, 5))
        self.assertGreater(move + F(1, 2), F(6, 5))

    def test_exhaustive_scalar_domain(self):
        # E46.4's entire 341-input domain, including the empty continuation.
        # The oracle uses a closed-form exact sum and distance minimization
        # on a grid, independently of recurrence()'s update-and-round method.
        witnesses = []
        checked = 0
        grid = tuple(F(value) for value in range(-32, 33, 4))
        for length in range(5):
            for inputs in product((F(-2), F(0), F(2), F(6)), repeat=length):
                checked += 1
                exact = recurrence(inputs)
                rounded = recurrence(inputs, F(4))
                previous = F(0)
                for index, (value, actual) in enumerate(
                    zip(inputs, rounded, strict=True)
                ):
                    expected_exact = sum(
                        inputs[j] / 2 ** (index - j) for j in range(index + 1)
                    )
                    self.assertEqual(exact[index], expected_exact)
                    target = previous / 2 + value
                    self.assertLess(abs(target), 32)  # Oracle grid covers the domain.
                    nearest = min(grid, key=lambda q: (abs(q - target), int(q / 4) % 2))
                    self.assertEqual(actual, nearest)
                    previous = actual
                exact_last = exact[-1] if exact else F(0)
                rounded_last = rounded[-1] if rounded else F(0)
                if readout(exact_last) != readout(rounded_last):
                    witnesses.append(inputs)
        self.assertEqual(checked, 341)
        self.assertIn((F(2), F(0)), witnesses)
        self.assertEqual(recurrence(()), ())
        self.assertEqual(recurrence((), F(4)), ())

    def test_cohort_and_amplification_edges(self):
        self.assertEqual(cohort_metrics(20, 16, 12, F(10)), (F(6, 5), F(3, 5), F(3, 4)))
        self.assertEqual(cohort_metrics(12, 12, 12, F(10))[1], 1)
        self.assertEqual(cohort_metrics(8, 4, 0, F(10))[1], 0)
        self.assertEqual(cohort_metrics(0, 0, 0, F(10)), (0, None, None))
        self.assertEqual(amplification(F(35), F(20)), F(7, 4))
        self.assertIsNone(amplification(F(1), F(0)))
        with self.assertRaises(ValueError):
            cohort_metrics(1, 1, 2, F(10))

    def test_unfenced_effect_witness(self):
        store = EffectStore()
        store.takeover()
        self.assertTrue(store.device_write(8, "prefix15"))
        self.assertTrue(store.device_write(7, "obsolete", fenced=False))
        self.assertFalse(store.commit(7, 15, ready=True))
        self.assertEqual(store.payload, "obsolete")
        self.assertFalse(store.commit(8, 15, ready=True))

    def test_fenced_late_writes_and_commit_readiness(self):
        # Both possible write orders after takeover preserve the new payload.
        for owners in permutations((7, 8)):
            store = EffectStore()
            store.takeover()
            for owner in owners:
                payload = "prefix15" if owner == 8 else "obsolete"
                self.assertEqual(store.device_write(owner, payload), owner == 8)
            self.assertFalse(store.commit(8, 15, ready=False))
            self.assertFalse(store.commit(7, 15, ready=True))
            self.assertTrue(store.commit(8, 15, ready=True))
            self.assertFalse(store.commit(8, 15, ready=True))
            self.assertEqual(store.published, [15])


if __name__ == "__main__":
    unittest.main()
