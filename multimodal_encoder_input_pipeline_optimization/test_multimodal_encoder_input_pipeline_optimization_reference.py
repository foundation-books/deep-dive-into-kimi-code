"""Independent finite differences and invalid-trace witnesses for Chapter 39."""

import unittest
from . import reference as r
from . import schedule as s


class Chapter39Tests(unittest.TestCase):
    def test_geometry_and_assignment(self):
        self.assertEqual([x.positions for x in r.SAMPLES], [0, 16, 64, 64])
        self.assertEqual(sum(x.length for x in r.SAMPLES), 196)
        self.assertEqual([x.spatial_cost for x in r.SAMPLES], [0, 256, 4096, 1024])
        groups, loads = r.greedy({x.name: x.spatial_cost for x in r.SAMPLES})
        self.assertEqual(loads, [4096, 1280])
        self.assertEqual(sorted(sum(groups, [])), list("ABCD"))

    def test_packing_and_masks(self):
        m = r.packing(r.SAMPLES[1:3])
        self.assertEqual(m[36], ("C", 0))
        self.assertEqual(len(set(m)), 84)
        self.assertFalse(r.allowed(m, 36, 35))
        self.assertTrue(r.allowed(m, 37, 36))
        self.assertFalse(r.allowed(m, 36, 37))
        with self.assertRaises(ValueError):
            r.packing([r.SAMPLES[1]] * 2)

    def test_attention_partition_and_finite_difference(self):
        q, k, v, g = [0.2, -0.4, 0.8], [0.1, -0.3], [0.5, 1.5], [1, 2, -1]
        full = r.attention(q, k, v, g)
        split = r.partitioned_attention(q, k, v, g, [[2], [0, 1]])
        for a, b in zip(full, split, strict=True):
            for x, y in zip(a, b, strict=True):
                self.assertAlmostEqual(x, y, places=12)
        args = [q, k, v]
        for index in range(3):
            for j in range(len(args[index])):
                plus, minus = [x[:] for x in args], [x[:] for x in args]
                plus[index][j] += 1e-6
                minus[index][j] -= 1e-6
                fp = sum(
                    x * y for x, y in zip(r.attention(*plus, g)[0], g, strict=True)
                )
                fm = sum(
                    x * y for x, y in zip(r.attention(*minus, g)[0], g, strict=True)
                )
                self.assertAlmostEqual((fp - fm) / 2e-6, full[index + 1][j], places=7)
        self.assertEqual(r.attention([0], [0, 0], [0, 2], [1])[0], [1])
        self.assertEqual(r.attention([0], [0], [0], [1])[0], [0])
        with self.assertRaises(ValueError):
            r.partitioned_attention(q, k, v, g, [[0, 1], [1, 2]])

    def test_queues_lifetimes_and_versions(self):
        trace = r.admit([(0, 6, 3), (1, 9, 1)], 12)
        self.assertEqual(trace, [(0, 3, 6), (3, 4, 9)])
        self.assertEqual(r.peak(trace), 9)
        self.assertEqual(r.peak([(0, 5, 6), (2, 6, 9)]), 15)
        with self.assertRaises(ValueError):
            r.admit([(0, 13, 1)], 12)
        for actual, outstanding in [(("B", 2), 0), (("B", 1), 1)]:
            with self.assertRaises(ValueError):
                r.require_version(("B", 1), actual, outstanding)

    def test_replicated_projection_weight_gradient(self):
        x, w, g = [0.2, -0.4, 0.8], 0.7, [1, 2, -1]

        def forward(weight):
            z = [xi * weight for xi in x]
            return sum(
                a * b for a, b in zip(r.attention(z, z, z, g)[0], g, strict=True)
            )

        z = [xi * w for xi in x]
        _, dq, dk, dv = r.partitioned_attention(z, z, z, g, [[2], [0, 1]])
        dw = sum(xi * (a + b + c) for xi, a, b, c in zip(x, dq, dk, dv, strict=True))
        numerical = (forward(w + 1e-6) - forward(w - 1e-6)) / 2e-6
        self.assertAlmostEqual(dw, numerical, places=7)
        self.assertNotAlmostEqual(2 * dw, numerical, places=5)

    def test_objective(self):
        self.assertEqual(r.weighted_loss([[0], [2, 2, 2]], [1, 1]), 1.5)
        self.assertNotEqual((0 + 2) / 2, 1.5)
        self.assertEqual(r.weighted_loss([[0], [2, 2, 2]], [3, 1]), 1)

    def test_bubble_deadline_and_shared_resource(self):
        self.assertEqual(s.bubble_start(10, 14, 3, [(10, 14)]), 10)
        self.assertIsNone(s.bubble_start(12, 14, 3, [(10, 14)]))
        self.assertIsNone(s.bubble_start(10, 14, 3, [(10, 14), (12, 14)]))
        self.assertIsNone(s.bubble_start(10, 13, 2, [(10, 14), (12, 15)]))

    def test_schedule_and_negative_mutants(self):
        for mode, expected in [("exposed", 34), ("overlap", 33), ("cp", 34)]:
            work = s.jobs(mode)
            events = s.run(work)
            self.assertEqual(s.check(work, events), expected)
            with self.assertRaises(ValueError):
                s.check(work, events[:-1])
            bad = [
                (j, 0, j.duration) if j.name == "U" else (j, a, b) for j, a, b in events
            ]
            with self.assertRaises(ValueError):
                s.check(work, bad)
        j1 = s.Job("x", 2, (0,), (), (0,))
        j2 = s.Job("y", 2, (0,), (), (1,))
        with self.assertRaises(ValueError):
            s.check([j1, j2], [(j1, 0, 2), (j2, 1, 3)])
        work = s.jobs("cp", late=30)
        self.assertEqual(s.check(work, s.run(work)), 44)


if __name__ == "__main__":
    unittest.main()
