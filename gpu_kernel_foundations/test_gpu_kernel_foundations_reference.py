"""Golden traces, operand pairing, every supported length, and input bounds."""

import unittest

from gpu_kernel_foundations.reference import (
    butterfly_stages,
    dot_trace,
    lane_indices,
    tile_copy_rows,
    tile_slice,
    two_slot_schedule,
    warp_and_lane,
    xor_storage_offset,
)


class WarpDotTests(unittest.TestCase):
    def test_block_identities(self):
        self.assertEqual(
            [warp_and_lane(t) for t in (31, 32, 70, 127)],
            [(0, 31), (1, 0), (2, 6), (3, 31)],
        )

    def test_golden_trace(self):
        stages = dot_trace(list(range(1, 129)), [1] * 128)
        self.assertEqual([s[0] for s in stages], [196, 396, 808, 1680, 3616, 8256])
        self.assertEqual(stages[0], tuple(196 + 4 * lane for lane in range(32)))
        self.assertEqual(stages[-1], (8256,) * 32)
        self.assertEqual(stages[0][:4], (196, 200, 204, 208))
        self.assertEqual(stages[1][:4], (396, 396, 412, 412))
        self.assertEqual(stages[2][:4], (808, 808, 808, 808))

    def test_all_supported_lengths_and_signed_weights(self):
        for length in range(129):
            x = list(range(1, length + 1))
            for weights in ([1] * length, [(i % 7) - 3 for i in range(length)]):
                with self.subTest(length=length, signed=weights != [1] * length):
                    expected = sum(a * b for a, b in zip(x, weights, strict=True))
                    self.assertEqual(dot_trace(x, weights)[-1], (expected,) * 32)
            assigned = [i for lane in range(32) for i in lane_indices(length, lane)]
            self.assertEqual(sorted(assigned), list(range(length)))
        self.assertEqual(dot_trace(list(range(1, 101)), [1] * 100)[-1][0], 5050)
        self.assertEqual(dot_trace([1, 2, 3], [1, 1, 1])[0], (1, 2, 3) + (0,) * 29)
        self.assertEqual(dot_trace([1, 2, 3], [1, 1, 1])[-1], (6,) * 32)
        self.assertEqual(
            dot_trace(list(range(1, 129)), [(i % 7) - 3 for i in range(128)])[-1],
            (-133,) * 32,
        )

    def test_each_coordinate_keeps_its_operand_pair(self):
        weights = [3 * i - 191 for i in range(128)]
        for coordinate in range(128):
            x = [0] * 128
            x[coordinate] = coordinate + 1
            expected = (coordinate + 1) * weights[coordinate]
            self.assertEqual(dot_trace(x, weights)[-1], (expected,) * 32)

    def test_stages_combine_disjoint_groups(self):
        # One set bit tags each original lane. A duplicate addition would carry
        # into another bit and fail the expected group-membership masks.
        stages = butterfly_stages([1 << lane for lane in range(32)])
        for stage, values in enumerate(stages):
            width = 1 << stage
            for lane, value in enumerate(values):
                first = (lane // width) * width
                expected = ((1 << width) - 1) << first
                self.assertEqual(value, expected)

    def test_rejects_unsupported_domains(self):
        for thread in (-1, 128):
            with self.assertRaises(ValueError):
                warp_and_lane(thread)
        for length, lane in ((-1, 0), (129, 0), (3, -1), (3, 32)):
            with self.assertRaises(ValueError):
                lane_indices(length, lane)
        for length in (0, 3, 31, 33):
            with self.assertRaises(ValueError):
                butterfly_stages([0] * length)
        with self.assertRaises(ValueError):
            dot_trace([1], [])


class TileAndLifetimeTests(unittest.TestCase):
    def test_two_slices_and_copy_placement(self):
        self.assertEqual(tile_slice(0), ((7, 10), (5, 8)))
        self.assertEqual(tile_slice(2), ((43, 50), (7, 8)))
        rows = tile_copy_rows()
        self.assertEqual(
            rows,
            (
                (0, 0, (0, 0), 8, 0, 50),
                (1, 0, (0, 1), 9, 1, 60),
                (2, 0, (1, 0), 12, 2, 12),
                (3, 0, (1, 1), 13, 3, 16),
            ),
        )
        # Independent partial products reconstruct each copied output.
        for row in rows:
            u, v = row.coordinate
            self.assertEqual(row.value, tile_slice(0)[u][v] + tile_slice(2)[u][v])
        relative_offsets = [row.source_offset - 8 for row in rows]
        self.assertEqual(relative_offsets, [0, 1, 4, 5])
        self.assertEqual(len(relative_offsets), 4)
        self.assertEqual(max(relative_offsets) + 1, 6)
        self.assertEqual(8 + rows[2].destination_offset, 10)
        self.assertNotEqual(8 + rows[2].destination_offset, rows[2].source_offset)

    def test_swizzle_bijection_banks_and_wrong_consumer(self):
        offsets = [xor_storage_offset(r, c) for r in range(32) for c in range(32)]
        self.assertEqual(sorted(offsets), list(range(1024)))
        self.assertEqual(xor_storage_offset(1, 2), 35)
        self.assertEqual(xor_storage_offset(1, 3), 34)
        self.assertEqual([(32 * r) % 32 for r in range(32)], [0] * 32)
        self.assertEqual(
            [xor_storage_offset(r, 0) % 32 for r in range(32)], list(range(32))
        )
        for row in range(32):
            self.assertEqual(
                sorted(xor_storage_offset(row, c) % 32 for c in range(32)),
                list(range(32)),
            )

    def test_reuse_timeline_and_generation(self):
        self.assertEqual(
            two_slot_schedule(3),
            (
                (0, 0, 0, (0, 2), (2, 5), 5),
                (1, 1, 0, (2, 4), (5, 8), 8),
                (2, 0, 1, (5, 7), (8, 11), 11),
            ),
        )
        for durations in ((2, 3), (3, 2)):
            uses = two_slot_schedule(6, *durations)
            self.assertEqual(uses[-1].reusable_at, sum(durations) + 5 * max(durations))
            for earlier, later in zip(uses[:-2], uses[2:], strict=True):
                self.assertEqual(earlier.slot, later.slot)
                self.assertNotEqual(earlier.generation, later.generation)
                self.assertGreaterEqual(later.load_interval[0], earlier.reusable_at)
        self.assertEqual(two_slot_schedule(0), ())

    def test_rejects_unsupported_maps_and_durations(self):
        with self.assertRaises(ValueError):
            tile_slice(1)
        for row, column in ((-1, 0), (32, 0), (0, -1), (0, 32)):
            with self.assertRaises(ValueError):
                xor_storage_offset(row, column)
        for args in ((-1, 2, 3), (3, 0, 3), (3, 2, 0)):
            with self.assertRaises(ValueError):
                two_slot_schedule(*args)


if __name__ == "__main__":
    unittest.main()
