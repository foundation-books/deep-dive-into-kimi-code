from __future__ import annotations

from fractions import Fraction
import unittest

from .reference import (
    RUNNING_BLOCK_SIZE,
    RUNNING_REPRESENTATIVES,
    RUNNING_TOTAL_BLOCK_BUDGET,
    accessible_indices,
    all_selections,
    batch_attention,
    block_routes,
    dense_masked_attention,
    finite_difference_gradients,
    full_causal_attention,
    max_abs_tensor_difference,
    mean_key_representatives,
    merge_online_states,
    mutate_suffix,
    online_merge_microexample,
    online_state,
    pair_and_cost_ledger,
    partition_blocks,
    partitioned_attention,
    prefix_difference,
    route_loads,
    running_tensors,
    select_historical_blocks,
)


class BlockAndRoutingTests(unittest.TestCase):
    def test_partial_tail_contains_only_real_positions(self) -> None:
        self.assertEqual(
            partition_blocks(9, 4),
            ((0, 1, 2, 3), (4, 5, 6, 7), (8,)),
        )

    def test_running_representatives_and_final_routes(self) -> None:
        queries, keys, _ = running_tensors()
        representatives = mean_key_representatives(keys, RUNNING_BLOCK_SIZE)
        self.assertEqual(representatives, RUNNING_REPRESENTATIVES)
        selections = all_selections(
            queries,
            keys,
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        self.assertEqual(selections[9][0], (0, 1))
        self.assertEqual(selections[9][1], (2, 1))
        self.assertEqual(selections[0], ((), ()))

    def test_equal_scores_break_ties_by_lower_block_identifier(self) -> None:
        self.assertEqual(
            select_historical_blocks(
                scores=(1.0, 1.0, -3.0),
                query_position=4,
                block_size=2,
                total_budget=3,
            ),
            (0, 1),
        )

    def test_accessible_union_contains_history_and_current_prefix_once(self) -> None:
        self.assertEqual(
            accessible_indices(
                length=10,
                query_position=9,
                block_size=2,
                selected_history=(0, 1),
            ),
            (0, 1, 2, 3, 8, 9),
        )


class SemanticOracleTests(unittest.TestCase):
    def test_partitioned_forward_matches_dense_masked_oracle(self) -> None:
        queries, keys, values = running_tensors()
        dense = dense_masked_attention(
            queries,
            keys,
            values,
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        grouped = partitioned_attention(
            queries,
            keys,
            values,
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        self.assertLessEqual(max_abs_tensor_difference(dense, grouped), 1e-12)

    def test_full_budget_recovers_dense_causal_attention(self) -> None:
        queries, keys, values = running_tensors()
        full_budget = len(partition_blocks(len(queries), RUNNING_BLOCK_SIZE))
        sparse_limit = dense_masked_attention(
            queries, keys, values, RUNNING_BLOCK_SIZE, full_budget
        )
        dense = full_causal_attention(queries, keys, values)
        self.assertLessEqual(max_abs_tensor_difference(sparse_limit, dense), 1e-12)

    def test_future_suffix_cannot_change_an_earlier_prefix(self) -> None:
        queries, keys, values = running_tensors()
        baseline = dense_masked_attention(
            queries,
            keys,
            values,
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        perturbed = dense_masked_attention(
            queries,
            mutate_suffix(keys, 6, 7.0),
            mutate_suffix(values, 6, -9.0),
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        self.assertEqual(prefix_difference(baseline, perturbed, 5), 0.0)

    def test_batch_wrapper_is_sequence_isolated(self) -> None:
        queries, keys, values = running_tensors()
        short = (queries[:6], keys[:6], values[:6])
        altered = (
            queries,
            mutate_suffix(keys, 0, 3.0),
            mutate_suffix(values, 0, -4.0),
        )
        batched = batch_attention(
            (short, altered), RUNNING_BLOCK_SIZE, RUNNING_TOTAL_BLOCK_BUDGET
        )
        independent = dense_masked_attention(
            *short, RUNNING_BLOCK_SIZE, RUNNING_TOTAL_BLOCK_BUDGET
        )
        self.assertEqual(batched[0], independent)

    def test_small_backward_oracles_agree(self) -> None:
        queries, keys, values = running_tensors()
        small = (queries[:4], keys[:4], values[:4])
        dense_gradients = finite_difference_gradients(
            dense_masked_attention,
            *small,
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        grouped_gradients = finite_difference_gradients(
            partitioned_attention,
            *small,
            RUNNING_BLOCK_SIZE,
            RUNNING_TOTAL_BLOCK_BUDGET,
        )
        error = max(
            max_abs_tensor_difference(left, right)
            for left, right in zip(dense_gradients, grouped_gradients, strict=True)
        )
        self.assertLessEqual(error, 1e-8)


class MergeAndAccountingTests(unittest.TestCase):
    def test_online_merge_matches_direct_fraction(self) -> None:
        result = online_merge_microexample()
        self.assertAlmostEqual(result["output"], 27.0 / 7.0)
        local_average_weights = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
        global_weights = (Fraction(1, 7), Fraction(2, 7), Fraction(4, 7))
        self.assertEqual(sum(local_average_weights), 1)
        self.assertEqual(sum(global_weights), 1)
        values = (1, 3, 5)
        self.assertEqual(
            sum(w * v for w, v in zip(local_average_weights, values, strict=True)),
            Fraction(11, 3),
        )
        self.assertEqual(
            sum(w * v for w, v in zip(global_weights, values, strict=True)),
            Fraction(27, 7),
        )
        self.assertNotEqual(local_average_weights, global_weights)
        first = online_state((0.0,), ((1.0, 2.0),))
        second = online_state((0.0,), ((3.0, 4.0),))
        merged = merge_online_states(first, second)
        self.assertEqual(merged.numerator, (4.0, 6.0))

    def test_routes_and_pair_counts_are_conserved(self) -> None:
        queries, keys, _ = running_tensors()
        routes = block_routes(
            queries, keys, RUNNING_BLOCK_SIZE, RUNNING_TOTAL_BLOCK_BUDGET
        )
        self.assertEqual(len(routes), 28)
        self.assertEqual(sum(route_loads(routes, 5)), 28)
        ledger = pair_and_cost_ledger(
            queries, keys, RUNNING_BLOCK_SIZE, RUNNING_TOTAL_BLOCK_BUDGET
        )
        self.assertEqual(ledger["dense_causal_query_key_pairs"], 110)
        self.assertEqual(ledger["historical_route_records"], 28)
        self.assertEqual(ledger["mandatory_current_assignments"], 20)
        self.assertGreater(ledger["realized_causal_sparsity"], 0.0)


if __name__ == "__main__":
    unittest.main()
