from __future__ import annotations

import math
import unittest

from .reference import (
    ToyState,
    attnres,
    dispatch_and_combine,
    evaluate_tokens,
    kda_step,
    max_abs_difference,
    multimodal_permission_matrix,
    resource_ledger,
    route,
    state_vector,
)


class KDAReferenceTests(unittest.TestCase):
    def test_exercise_state_and_output(self) -> None:
        result = kda_step(
            ((1.0, 0.0), (0.0, 1.0)),
            key=(1.0, 0.0),
            value=(2.0, -1.0),
            query=(1.0, 1.0),
            beta=0.5,
            retention=(0.75, 0.5),
        )
        self.assertEqual(result.state, [[1.375, -0.5], [0.0, 0.5]])
        self.assertEqual(result.output, [1.375, 0.0])

    def test_update_decomposition(self) -> None:
        result = kda_step(
            ((1.0, 0.0), (0.0, 1.0)),
            key=(1.0, 0.0),
            value=(2.0, -1.0),
            query=(1.0, 1.0),
            beta=0.5,
            retention=(0.75, 0.5),
        )
        rebuilt = [
            [
                result.retained[i][j] - result.erased[i][j] + result.written[i][j]
                for j in range(2)
            ]
            for i in range(2)
        ]
        self.assertEqual(rebuilt, result.state)


class RouterReferenceTests(unittest.TestCase):
    def test_bias_affects_selection_not_weight_numerator(self) -> None:
        result = route((0.1, 0.7, 0.4, 0.6), (0.2, -0.1, 0.0, 0.1), 2)
        self.assertEqual(result.selected, (3, 0))
        self.assertAlmostEqual(result.biased_scores[0], result.scores[0] + 0.2)
        self.assertAlmostEqual(sum(result.weights), 1.0)
        expected = result.scores[3] / (result.scores[3] + result.scores[0])
        self.assertAlmostEqual(result.weights[0], expected)

    def test_extreme_negative_logits_keep_normalized_relative_weights(self) -> None:
        result = route((-1001.0, -1000.0), (0.0, 0.0), 2)
        self.assertEqual(result.scores, (0.0, 0.0))
        self.assertEqual(result.selected, (1, 0))
        self.assertAlmostEqual(result.weights[0], 1.0 / (1.0 + math.exp(-1.0)))
        self.assertAlmostEqual(sum(result.weights), 1.0)

    def test_dispatch_combine_preserves_contribution_count(self) -> None:
        decisions = (
            route((0.1, 0.7, 0.4, 0.6), (0.2, -0.1, 0.0, 0.1), 2),
            route((0.3, 0.2, 0.8, 0.1), (0.0, 0.1, -0.1, 0.0), 2),
        )
        combined, counts = dispatch_and_combine(((1.0, 2.0), (3.0, 4.0)), decisions)
        self.assertEqual(counts, (2, 2))
        self.assertEqual(len(combined), 2)
        self.assertTrue(all(len(token) == 2 for token in combined))


class AttentionResidualTests(unittest.TestCase):
    def test_exact_three_source_weights(self) -> None:
        weights, output = attnres(
            (0.0, math.log(2.0), math.log(3.0)),
            ((0.0, 0.0), (6.0, 0.0), (0.0, 6.0)),
        )
        self.assertLessEqual(max_abs_difference(weights, (1 / 6, 2 / 6, 3 / 6)), 1e-12)
        self.assertLessEqual(max_abs_difference(output, (2.0, 3.0)), 1e-12)


class StateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = (
            (0.2, 0.1, 0.3, -0.2, 0.4, 0.5),
            (0.1, -0.1, 0.2, 0.6, 0.3, -0.4),
            (0.5, 0.2, -0.3, 0.1, 0.7, 0.2),
            (-0.2, 0.4, 0.1, 0.3, -0.1, 0.8),
        )

    def test_full_prefix_matches_prefill_plus_step(self) -> None:
        full_logits, full_state = evaluate_tokens(self.tokens)
        _, prefill_state = evaluate_tokens(self.tokens[:-1])
        step_logits, step_state = evaluate_tokens((self.tokens[-1],), prefill_state)
        self.assertLessEqual(max_abs_difference(full_logits, step_logits), 1e-12)
        self.assertLessEqual(
            max_abs_difference(state_vector(full_state), state_vector(step_state)),
            1e-12,
        )

    def test_state_count_advances_once(self) -> None:
        _, state = evaluate_tokens((self.tokens[0],), ToyState.zero())
        self.assertEqual(state.token_count, 1)
        self.assertEqual(len(state.mla_cache), 1)


class ShapeAndEvidenceTests(unittest.TestCase):
    def test_multimodal_permission_relation(self) -> None:
        matrix = multimodal_permission_matrix(3, 4)
        self.assertEqual((len(matrix), len(matrix[0])), (7, 7))
        self.assertTrue(all(matrix[-1]))
        self.assertFalse(matrix[3][4])
        self.assertTrue(all(matrix[0][index] for index in range(3)))

    def test_resource_ledger_values(self) -> None:
        ledger = resource_ledger()
        self.assertEqual(ledger["prompt_activation_scalars"], 80)
        self.assertEqual(ledger["kda_state_scalars"], 24)
        self.assertEqual(ledger["mla_latent_cache_scalars"], 40)
        self.assertEqual(ledger["full_to_latent_cache_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
