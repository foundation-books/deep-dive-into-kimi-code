"""Tests for the Chapter 7 Attention Residuals reference oracle."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import unittest

from . import reference as REFERENCE


class AttentionResidualsReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dims = REFERENCE.RUNNING_DIMS

    def test_running_dimensions_and_partial_final_block(self) -> None:
        self.assertEqual(self.dims.transformations, 14)
        self.assertEqual(self.dims.block_lengths, (3, 3, 1))
        self.assertEqual(self.dims.blocks, 3)
        self.assertEqual(self.dims.terminal_sources, 4)

    def test_running_embedding_shape(self) -> None:
        embedding = REFERENCE.running_embedding()
        self.assertEqual(len(embedding), 10)
        self.assertEqual(tuple(map(len, embedding)), (8,) * 10)

    def test_exact_residual_unrolling(self) -> None:
        f = Fraction
        iterative, direct = REFERENCE.exact_residual_unroll(
            (f(1), f(-1)),
            ((f(2), f(3)), (f(-4), f(5)), (f(7), f(-2))),
        )
        self.assertEqual(iterative, direct)
        self.assertEqual(direct, (f(6), f(5)))

    def test_exact_three_source_example(self) -> None:
        mixture, residual = REFERENCE.exact_three_source_example()
        self.assertEqual(mixture, (Fraction(7, 6), Fraction(5, 3)))
        self.assertEqual(residual, (Fraction(3), Fraction(4)))

    def test_zero_query_is_uniform(self) -> None:
        dims = REFERENCE.Dimensions(
            tokens=1,
            model=2,
            decoder_layers=1,
            block_size=1,
        )
        sources = (
            ((1.0, 0.0),),
            ((0.0, 2.0),),
            ((2.0, 2.0),),
        )
        result = REFERENCE.mix_depth((0.0, 0.0), sources, dims)
        self.assertEqual(result.weights, ((1 / 3, 1 / 3, 1 / 3),))
        self.assertAlmostEqual(result.output[0][0], 1.0)
        self.assertAlmostEqual(result.output[0][1], 4 / 3)

    def test_full_probabilities_use_depth_axis(self) -> None:
        trace = REFERENCE.full_forward()
        for step in trace.steps:
            self.assertEqual(len(step.mixture.weights), self.dims.tokens)
            for row in step.mixture.weights:
                self.assertEqual(len(row), len(step.source_ids))
                self.assertAlmostEqual(sum(row), 1.0, places=13)

    def test_same_query_can_give_token_conditioned_weights(self) -> None:
        trace = REFERENCE.full_forward()
        step = trace.steps[1]
        self.assertNotEqual(step.mixture.weights[0], step.mixture.weights[9])

    def test_full_primary_and_independent_oracles_agree(self) -> None:
        direct = REFERENCE.full_forward()
        oracle = REFERENCE.full_forward_oracle()
        self.assertTrue(REFERENCE.traces_close(direct, oracle))

    def test_full_source_progression_and_terminal(self) -> None:
        trace = REFERENCE.full_forward()
        self.assertEqual(
            tuple(len(step.source_ids) for step in trace.steps),
            tuple(range(1, 15)),
        )
        self.assertEqual(len(trace.terminal_source_ids), 15)

    def test_block_primary_and_independent_oracles_agree(self) -> None:
        direct = REFERENCE.block_forward()
        oracle = REFERENCE.block_forward_oracle()
        self.assertTrue(REFERENCE.traces_close(direct, oracle))

    def test_block_boundary_source_counts(self) -> None:
        trace = REFERENCE.block_forward()
        expected = (
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
        )
        self.assertEqual(
            tuple(len(step.source_ids) for step in trace.steps),
            expected,
        )
        self.assertEqual(
            trace.terminal_source_ids,
            (
                "embedding",
                "block-1",
                "block-2",
                "partial-3",
            ),
        )

    def test_block_source_conservation_after_every_output(self) -> None:
        trace = REFERENCE.block_forward()
        for step in trace.steps:
            groups = REFERENCE.source_partition(step.state_after)
            represented = tuple(index for group in groups[1:] for index in group)
            self.assertEqual(represented, tuple(range(step.target + 1)))

    def test_early_commit_fails(self) -> None:
        state = REFERENCE.new_block_state()
        output = REFERENCE.transform_grid(state.embedding, 0)
        state = REFERENCE.append_partial(state, output, 0)
        with self.assertRaises(ValueError):
            REFERENCE.commit_partial(state)

    def test_duplicate_and_stale_indices_fail(self) -> None:
        trace = REFERENCE.block_forward()
        first_state = trace.steps[0].state_after
        duplicate = replace(first_state, partial_indices=(0, 0))
        with self.assertRaises(ValueError):
            REFERENCE.validate_block_state(duplicate)
        committed_state = trace.steps[6].state_after
        stale = replace(
            committed_state,
            partial=trace.steps[0].transformation_output,
            partial_indices=(0,),
        )
        with self.assertRaises(ValueError):
            REFERENCE.validate_block_state(stale)

    def test_missing_embedding_shape_fails(self) -> None:
        state = REFERENCE.new_block_state()
        broken = replace(state, embedding=state.embedding[1:])
        with self.assertRaises(ValueError):
            REFERENCE.validate_block_state(broken)

    def test_ephemeral_state_cannot_cross_forward_identity(self) -> None:
        state = REFERENCE.new_block_state()
        wrong = REFERENCE.DepthIdentity(
            "another-forward",
            state.identity.model_revision,
            state.identity.tokens,
            state.identity.width,
        )
        with self.assertRaises(ValueError):
            REFERENCE.validate_block_state(state, expected=wrong)

    def test_two_phase_merge_matches_direct_mix_at_every_state(self) -> None:
        trace = REFERENCE.block_forward()
        queries = REFERENCE.running_queries()
        for step in trace.steps:
            next_target = step.target + 1
            if next_target >= self.dims.transformations:
                continue
            sources, _ = REFERENCE.block_sources(step.state_after)
            direct = REFERENCE.mix_depth(
                queries[next_target],
                sources,
            ).output
            merged = REFERENCE.two_phase_block_mix(
                queries[next_target],
                step.state_after,
            )
            self.assertLess(
                REFERENCE.max_abs_difference(direct, merged),
                1e-12,
            )
        terminal_direct = trace.terminal.output
        terminal_merged = REFERENCE.two_phase_block_mix(
            queries[-1],
            trace.terminal_state,
        )
        self.assertLess(
            REFERENCE.max_abs_difference(terminal_direct, terminal_merged),
            1e-12,
        )

    def test_online_merge_handles_unequal_maxima(self) -> None:
        dims = REFERENCE.Dimensions(
            tokens=1,
            model=2,
            decoder_layers=1,
            block_size=1,
        )
        query = (40.0, -40.0)
        left_sources = (((1.0, 0.0),), ((0.0, 1.0),))
        right_sources = (((-1.0, 0.0),),)
        left = REFERENCE.online_summary(query, left_sources, dims)[0]
        right = REFERENCE.online_summary(query, right_sources, dims)[0]
        merged = REFERENCE.merge_online_state(left, right)
        output = tuple(value / merged.normalizer for value in merged.numerator)
        direct = REFERENCE.mix_depth(
            query,
            left_sources + right_sources,
            dims,
        ).output[0]
        self.assertLess(
            REFERENCE.max_abs_difference(output, direct),
            1e-12,
        )

    def test_wrong_token_source_axis_fails(self) -> None:
        embedding = REFERENCE.running_embedding()
        transposed = tuple(zip(*embedding, strict=True))
        with self.assertRaises(ValueError):
            REFERENCE.mix_depth(
                REFERENCE.query_vector(0),
                (transposed,),
            )

    def test_toy_resource_ledgers(self) -> None:
        full = REFERENCE.full_resource_ledger()
        block = REFERENCE.block_resource_ledger()
        self.assertEqual(full.source_interactions, 120)
        self.assertEqual(block.source_interactions, 39)
        self.assertEqual(full.maximum_live_sources, 15)
        self.assertEqual(block.maximum_live_sources, 4)
        self.assertEqual(full.live_source_scalars, 1200)
        self.assertEqual(block.live_source_scalars, 320)
        self.assertEqual(
            full.pseudo_query_parameters,
            block.pseudo_query_parameters,
        )

    def test_k3_boundary_ledger(self) -> None:
        ledger = REFERENCE.k3_boundary_ledger()
        self.assertEqual(ledger["block_lengths"], [12] * 7 + [9])
        self.assertEqual(ledger["non_embedding_blocks"], 8)
        self.assertEqual(ledger["terminal_sources"], 9)
        self.assertEqual(ledger["transformations"], 186)
        self.assertEqual(ledger["block_source_interactions"], 1003)
        self.assertEqual(ledger["full_source_interactions"], 17578)


if __name__ == "__main__":
    unittest.main()
