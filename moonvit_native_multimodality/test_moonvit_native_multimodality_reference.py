"""Tests for the Chapter 9 reference contracts."""

from __future__ import annotations

from math import inf
import unittest

from . import reference as REFERENCE


class GeometryTests(unittest.TestCase):
    def test_running_grid_and_padding(self) -> None:
        image = REFERENCE.realize_image(56, 56)
        self.assertEqual((image.patch_rows, image.patch_cols), (4, 4))
        self.assertEqual(image.premerge_tokens, 16)
        self.assertEqual(image.projected_tokens, 4)
        padded = REFERENCE.realize_image(61, 45)
        self.assertEqual((padded.pad_width, padded.pad_height), (23, 11))
        self.assertEqual((padded.patch_rows, padded.patch_cols), (4, 6))

    def test_pinned_limits_do_not_upscale(self) -> None:
        small = REFERENCE.realize_image(13, 9)
        self.assertEqual((small.resized_width, small.resized_height), (13, 9))
        large = REFERENCE.realize_image(10_000, 8_000)
        self.assertLessEqual(large.resized_width, 512 * 14)
        self.assertLessEqual(large.resized_height, 512 * 14)
        self.assertLessEqual(large.premerge_tokens, 65_536 + 1_024)

    def test_budget_rounding_and_alignment_are_not_a_hard_token_cap(self) -> None:
        pinned = REFERENCE.realize_image(3_584, 3_584)
        rounded = REFERENCE.realize_image(4_096, 4_096)
        larger = REFERENCE.realize_image(8_192, 8_192)

        self.assertEqual(pinned.projected_tokens, 16_384)
        self.assertEqual(
            (
                rounded.resized_width,
                rounded.pad_width,
                rounded.patch_cols,
                rounded.premerge_tokens,
                rounded.projected_tokens,
            ),
            (3_591, 21, 258, 66_564, 16_641),
        )
        self.assertGreater(rounded.projected_tokens, pinned.projected_tokens)
        self.assertEqual(larger.projected_tokens, pinned.projected_tokens)


class PackingAndMergeTests(unittest.TestCase):
    def test_offsets_coordinates_and_isolation(self) -> None:
        packed = REFERENCE.pack_grids(((1, 4, 4), (1, 4, 6)))
        self.assertEqual(packed.offsets, (0, 16, 40))
        self.assertEqual(packed.coordinates[16], (1, 0, 0, 0))
        mask = REFERENCE.isolation_mask(packed.sample_ids)
        self.assertEqual(mask[0][15], 0.0)
        self.assertEqual(mask[0][16], -inf)
        self.assertTrue(REFERENCE.mask_is_sample_isolated(packed.sample_ids, mask))

    def test_merge_is_a_four_source_rearrangement(self) -> None:
        groups = REFERENCE.merge_groups_2x2(4, 4)
        self.assertEqual(
            groups, ((0, 1, 4, 5), (2, 3, 6, 7), (8, 9, 12, 13), (10, 11, 14, 15))
        )
        merged = REFERENCE.merge_features_2x2(
            tuple((float(index),) for index in range(16)), 4, 4
        )
        self.assertEqual(merged[0], (0.0, 1.0, 4.0, 5.0))
        self.assertEqual(
            sorted(value for row in merged for value in row),
            list(map(float, range(16))),
        )


class VideoProjectionAndSpliceTests(unittest.TestCase):
    def test_factorized_interaction_ledger(self) -> None:
        counts = REFERENCE.attention_interactions(4, 16)
        self.assertEqual(counts["full"], 4096)
        self.assertEqual(counts["factorized_spatial"], 1024)
        self.assertEqual(counts["factorized_temporal"], 256)
        self.assertEqual(counts["factorized_total"], 1280)

    def test_temporal_pool_and_bias_free_projector(self) -> None:
        pooled = REFERENCE.temporal_pool(((1.0, 3.0), (3.0, 5.0), (9.0, 7.0)), 2)
        self.assertEqual(pooled, ((2.0, 4.0), (9.0, 7.0)))
        first = ((1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0))
        second = ((1.0, 0.0), (0.0, 1.0))
        projected = REFERENCE.bias_free_mlp_projector(
            (1.0, 2.0, 3.0, 4.0), first, second
        )
        self.assertEqual(len(projected), 2)
        self.assertAlmostEqual(
            sum(value * value for value in projected) / 2, 1.0, places=5
        )

    def test_splice_conserves_order_and_prompt_length(self) -> None:
        stream, spans = REFERENCE.splice_media(
            ("<image>", "Which", "shape", "is", "below", "red", "circle?"),
            (("v1", "v2", "v3", "v4"),),
        )
        self.assertEqual(spans, ((0, 4),))
        self.assertEqual(len(stream), 10)
        self.assertEqual(stream[-1], "circle?")
        with self.assertRaises(ValueError):
            REFERENCE.splice_media(("<image>",), ())


class IntegratedTraceTests(unittest.TestCase):
    def test_running_trace_and_faults(self) -> None:
        trace = REFERENCE.running_trace()
        self.assertEqual(trace["premerge_tokens"], 16)
        self.assertEqual(trace["projected_tokens"], 4)
        self.assertEqual(trace["backbone_positions"], 10)
        self.assertTrue(trace["next_token_not_in_prompt"])
        faults = REFERENCE.fault_detection_summary()
        self.assertEqual(len(faults), 7)
        self.assertTrue(all(faults.values()))


if __name__ == "__main__":
    unittest.main()
