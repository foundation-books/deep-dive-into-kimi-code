"""Fault-sensitive exact checks for the Chapter 37 printed worksheets."""

from __future__ import annotations

from dataclasses import fields, replace
from fractions import Fraction
import unittest

from . import reader_walkthrough as walk
from . import reference as ref


class ReaderWalkthroughTests(unittest.TestCase):
    def test_outputs_and_unequal_coefficient_swap(self) -> None:
        routes = ref.canonical_routes()
        expected = dict(
            zip(
                (
                    "v1",
                    "v2",
                    "v3",
                    "v4",
                    "tau1",
                    "tau2",
                    "tau3",
                    "tau4",
                    "tau5",
                    "tau6",
                ),
                map(
                    Fraction,
                    (
                        "101.5",
                        "201.5",
                        "301.5",
                        "401.5",
                        "502",
                        "602",
                        "702.654629",
                        "802.5",
                        "903",
                        "1003.5",
                    ),
                ),
                strict=True,
            )
        )
        self.assertEqual(walk.diagnostic_outputs(routes), expected)
        self.assertEqual(sum(expected.values()), ref.EXPECTED_CHECKSUM)
        swapped = tuple(
            replace(route, weight=1 - route.weight) if route.origin == "tau3" else route
            for route in routes
        )
        self.assertEqual(
            walk.diagnostic_outputs(swapped)["tau3"], Fraction("702.345371")
        )

    def test_route_moves_are_not_extra_remote_payloads(self) -> None:
        home, balanced = ref.home_plan(), ref.balanced_plan()
        self.assertEqual(walk.source_execution_counts(home), ((9, 1), (4, 6)))
        self.assertEqual(walk.source_execution_counts(balanced), ((8, 2), (2, 8)))
        self.assertEqual(len(walk.remote_input_pairs(home)), 5)
        self.assertEqual(
            walk.remote_input_pairs(balanced),
            frozenset({("tau1", 1), ("tau4", 0), ("tau5", 0)}),
        )
        self.assertEqual(len(ref.MIGRATED), 3)

    def test_tau1_exactly_once_weighting_and_local_return(self) -> None:
        routes = [route for route in ref.canonical_routes() if route.origin == "tau1"]
        contributions = [
            route.weight * (100 * route.token + route.expert) for route in routes
        ]
        self.assertEqual(contributions, [Fraction("250.5"), Fraction("251.5")])
        self.assertEqual(sum(contributions), 502)
        self.assertEqual(2 * sum(contributions), 1004)
        self.assertNotEqual(
            sum(
                route.weight * value
                for route, value in zip(routes, contributions, strict=True)
            ),
            502,
        )

    def test_signed_duplicates_keep_a_real_slot_even_at_raw_zero(self) -> None:
        self.assertEqual(walk.encode_destination(1, 8, 32, False), 40)
        self.assertEqual(walk.encode_destination(1, 0, 32, True), -33)
        self.assertEqual(walk.decode_destination(-33, 32), (1, 0))
        self.assertEqual(walk.encode_destination(0, 0, 32, True), -1)
        self.assertEqual(walk.decode_destination(-1, 32), (0, 0))
        with self.assertRaises(ValueError):
            walk.encode_destination(1, 32, 32, False)

    def test_gradient_suffix_is_an_alias_not_extra_backing(self) -> None:
        self.assertEqual(
            walk.physical_backing_bytes(),
            {
                "weight_view": 288,
                "local_weights": 192,
                "gradient_view": 576,
                "temporary_gradient_alias": 192,
                "local_gradients": 384,
            },
        )

    def test_overwrite_invalidates_lease_but_not_full_plan_binding(self) -> None:
        binding = walk.PhysicalBinding(
            ref.balanced_plan().digest,
            1,
            ("hidden-A", "weights-A"),
            7,
            ref.HOME,
            (12, 12),
            8,
            "BF16/FP32",
        )
        buffer = walk.PayloadBuffer(binding)
        first = buffer.publish((2, 3))
        copied = buffer.read(first, binding)
        second = buffer.publish((7, 11))
        walk.validate_replay(binding, buffer.binding)
        self.assertEqual(buffer.read(second, binding), (7, 11))
        self.assertEqual(copied, (2, 3))
        with self.assertRaisesRegex(ValueError, "expired"):
            buffer.read(first, binding)

        replacements = {
            "plan_digest": "another-plan",
            "allocation_generation": 2,
            "address_tags": ("hidden-B", "weights-A"),
            "parameter_revision": 8,
            "homes": (1, 1, 0, 0),
            "capacities": (16, 16),
            "payload_width": 16,
            "dtype_policy": "FP32/FP32",
        }
        self.assertEqual(set(replacements), {item.name for item in fields(binding)})
        for name, value in replacements.items():
            with self.subTest(field=name), self.assertRaises(ValueError):
                buffer.read(second, replace(binding, **{name: value}))

    def test_reprefetch_requires_the_forward_version(self) -> None:
        available = {7: (2, 3), 8: (7, 11)}
        self.assertEqual(walk.restore_forward_value(7, available), (2, 3))
        with self.assertRaises(ValueError):
            walk.restore_forward_value(7, {8: (7, 11)})

    def test_165_counts_85_router_histograms_11_first_splits(self) -> None:
        mathematical = tuple(walk.weak_compositions(8, 4))
        candidates = {row for row in mathematical if max(row) <= 4}
        globally_realized = walk.distinct_pair_histograms(4)
        per_source = walk.distinct_pair_histograms(2)
        admitted = {
            row
            for row in candidates
            if all(split in per_source for split in walk.first_source_split(row))
        }
        self.assertEqual(
            (len(mathematical), len(candidates), len(admitted)), (165, 85, 11)
        )
        self.assertEqual(candidates, globally_realized)
        self.assertIn((2, 2, 2, 2), admitted)
        self.assertNotIn((0, 0, 4, 4), admitted)
        self.assertIn((0, 0, 4, 4), globally_realized)
        self.assertNotIn((8, 0, 0, 0), globally_realized)
        self.assertEqual(
            walk.first_source_split((0, 0, 4, 4)), ((0, 0, 0, 4), (0, 0, 4, 0))
        )

    def test_gpu_slot_domain_preserves_counts_and_allows_repeated_ids(self) -> None:
        larger = (
            (4, 4, 4, 4),
            (8, 0, 4, 4),
            (0, 8, 4, 4),
            (4, 0, 8, 4),
            (4, 4, 0, 8),
            (1, 7, 2, 6),
            (2, 6, 1, 7),
            (3, 5, 5, 3),
            (6, 6, 6, 6),
            (12, 0, 6, 6),
            (2, 10, 5, 7),
            (7, 5, 10, 2),
        )
        counts_grid = (*walk.weak_compositions(8, 4), *larger)
        self.assertEqual(len(counts_grid), 177)
        for counts in counts_grid:
            with self.subTest(counts=counts):
                arrays = walk.sorted_source_slots(counts, 2, 2)
                flat = [expert for rank in arrays for pair in rank for expert in pair]
                self.assertEqual(
                    tuple(flat.count(expert) for expert in range(1, 5)), counts
                )
                self.assertTrue(
                    any(len(set(pair)) != 2 for rank in arrays for pair in rank)
                )
                matrix = ref.constructive_balance(counts, ref.HOME, 2)
                ref.validate_allocation(matrix, counts, sum(counts) // 2)

    def test_four_rank_input_capacities_keys_and_owner_sums(self) -> None:
        routes, destinations, matrix = walk.four_rank_worksheet()
        homes = (0, 0, 1, 1, 2, 2, 3, 3)
        self.assertEqual(len({route.identity for route in routes}), 24)
        self.assertEqual(ref.column_sums(matrix), (6, 6, 6, 6))
        self.assertEqual(
            ref.redundant_sets(matrix, homes),
            (frozenset({3, 4}), frozenset({5}), frozenset({7}), frozenset()),
        )
        for rank in range(4):
            segments = ref.physical_segments(matrix, homes, rank, 4)
            self.assertEqual(tuple(segment.end for segment in segments), (4, 8))
            self.assertLessEqual(segments[-1].end, 8)
        partials: dict[tuple[int, int], int] = {}
        for route, destination in zip(routes, destinations, strict=True):
            key = (route.expert, destination)
            partials[key] = partials.get(key, 0) + 10 * route.token + 1
        totals = tuple(
            sum(value for (expert, _), value in partials.items() if expert == target)
            for target in range(1, 9)
        )
        self.assertEqual(totals, (0, 0, 104, 264, 424, 584, 744, 904))
        self.assertEqual((partials[(4, 1)], partials[(4, 0)]), (152, 112))
        self.assertEqual((partials[(7, 3)], partials[(7, 2)]), (392, 352))


if __name__ == "__main__":
    unittest.main()
