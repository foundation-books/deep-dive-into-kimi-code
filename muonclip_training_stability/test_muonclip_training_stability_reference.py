"""Dependency-free tests for the Chapter 13 MuonClip companion."""

from __future__ import annotations

from fractions import Fraction
from math import isclose
import unittest

from . import reference

MLAHead = reference.MLAHead
absolute_extreme = reference.absolute_extreme
clip_factor = reference.clip_factor
distributed_consensus = reference.distributed_consensus
lag_counterexample = reference.lag_counterexample
large_logit_ratio = reference.large_logit_ratio
mha_allocation_factors = reference.mha_allocation_factors
mla_clip = reference.mla_clip
mla_score = reference.mla_score
running_trace = reference.running_trace
scaled_positive_max = reference.scaled_positive_max
score_matrix = reference.score_matrix
soft_cap = reference.soft_cap
trigger_statistics = reference.trigger_statistics
vector = reference.vector
ScoreMatrix = tuple[tuple[Fraction, ...], ...]


class MuonClipReferenceTests(unittest.TestCase):
    def test_positive_maximum_is_not_absolute_extreme(self) -> None:
        scores = ((Fraction(-500), Fraction(4)), (Fraction(3), Fraction(2)))
        self.assertEqual(scaled_positive_max(scores), 4)
        self.assertEqual(absolute_extreme(scores), 500)

    def test_admissible_mask_changes_the_declared_domain(self) -> None:
        scores = score_matrix(
            (vector(1, 0), vector(0, 1)),
            (vector(4, 0), vector(0, 3)),
            1,
        )
        causal = ((True, False), (True, True))
        self.assertEqual(scaled_positive_max(scores, causal), 4)
        invalid_masks = (
            ((True,), (True, True)),
            ((True, True),),
            ((True, True), (True, True), (False, False)),
        )
        for mask in invalid_masks:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                scaled_positive_max(scores, mask)

    def test_piecewise_factor_is_suppressive_and_safe(self) -> None:
        self.assertEqual(clip_factor(400, 100), Fraction(1, 4))
        self.assertEqual(clip_factor(80, 100), 1)
        self.assertEqual(clip_factor(-20, 100), 1)
        with self.assertRaises(ValueError):
            clip_factor(1, 0)

    def test_mha_allocation_conserves_one_gamma(self) -> None:
        for alpha in (Fraction(0), Fraction(1, 2), Fraction(1)):
            with self.subTest(alpha=alpha):
                query_factor, key_factor = mha_allocation_factors(Fraction(1, 4), alpha)
                self.assertEqual(query_factor * key_factor, Fraction(1, 4))

    def test_mla_paths_contract_and_shared_key_is_external(self) -> None:
        shared = vector(40, 0)
        head = MLAHead(vector(10, 0), vector(40, 0), vector(10, 0))
        clipped = mla_clip(head, Fraction(1, 4))
        self.assertEqual(mla_score(head, shared), 400)
        self.assertEqual(mla_score(clipped, shared), 100)
        self.assertEqual(shared, vector(40, 0))

    def test_complete_running_trace_separates_endpoints(self) -> None:
        trace = running_trace()
        self.assertEqual(trace["observed_scores"], (Fraction(400), Fraction(80)))
        self.assertEqual(trace["factors"], (Fraction(1, 4), Fraction(1)))
        self.assertEqual(trace["same_state_replay"], (Fraction(100), Fraction(80)))
        self.assertEqual(trace["candidate_scores"], (Fraction(384), Fraction(72)))
        self.assertEqual(trace["next_forward_scores"], (Fraction(96), Fraction(72)))
        self.assertEqual(trace["next_factors"], (Fraction(1), Fraction(1)))

    def test_complete_causal_fixture_matches_every_published_matrix(self) -> None:
        def projection_query(coefficient: Fraction | int) -> ScoreMatrix:
            return (vector(0, 0), vector(coefficient, 0))

        def projection_key(coefficient: Fraction | int) -> ScoreMatrix:
            return (vector(coefficient, 0), vector(0, 0))

        def branches(
            query: Fraction | int, key: Fraction | int, shared: Fraction | int
        ) -> tuple[ScoreMatrix, ScoreMatrix, ScoreMatrix]:
            content = score_matrix(projection_query(query), projection_key(key), 1)
            rotary = score_matrix(projection_query(query), projection_key(shared), 1)
            scaled = tuple(
                tuple((content[i][j] + rotary[i][j]) / 2 for j in range(2))
                for i in range(2)
            )
            return content, rotary, scaled

        mask = ((True, False), (True, True))
        observed_1 = branches(10, 40, 40)
        observed_2 = branches(2, 40, 40)
        self.assertEqual(observed_1, ((vector(0, 0), vector(400, 0)),) * 3)
        self.assertEqual(observed_2, ((vector(0, 0), vector(80, 0)),) * 3)
        maxima = tuple(
            scaled_positive_max(record[2], mask) for record in (observed_1, observed_2)
        )
        self.assertEqual(maxima, (400, 80))
        self.assertEqual(max(maxima), 400)
        self.assertEqual(
            Fraction(sum(value > 100 for value in maxima), 2), Fraction(1, 2)
        )
        self.assertEqual(large_logit_ratio(observed_1[2], 100, mask), Fraction(1, 3))
        self.assertEqual(large_logit_ratio(observed_2[2], 100, mask), 0)
        permitted = tuple(
            value
            for record in (observed_1, observed_2)
            for row_index, row in enumerate(record[2])
            for column_index, value in enumerate(row)
            if mask[row_index][column_index]
        )
        self.assertEqual(permitted, (0, 400, 0, 0, 80, 0))
        self.assertEqual(large_logit_ratio((permitted,), 100), Fraction(1, 6))

        replay_1 = branches(5, 20, 40)
        replay_1_rotary = score_matrix(
            projection_query(Fraction(5, 2)), projection_key(40), 1
        )
        replay_1_scores = tuple(
            tuple((replay_1[0][i][j] + replay_1_rotary[i][j]) / 2 for j in range(2))
            for i in range(2)
        )
        self.assertEqual(replay_1_scores, (vector(0, 0), vector(100, 0)))
        self.assertEqual(observed_2[2], (vector(0, 0), vector(80, 0)))

        candidate_1 = branches(12, 32, 32)
        candidate_2 = branches(Fraction(9, 4), 32, 32)
        committed_content = score_matrix(projection_query(6), projection_key(16), 1)
        committed_rotary = score_matrix(projection_query(3), projection_key(32), 1)
        committed_1 = tuple(
            tuple(
                (committed_content[i][j] + committed_rotary[i][j]) / 2 for j in range(2)
            )
            for i in range(2)
        )
        self.assertEqual(candidate_1[2], (vector(0, 0), vector(384, 0)))
        self.assertEqual(candidate_2[2], (vector(0, 0), vector(72, 0)))
        self.assertEqual(committed_1, (vector(0, 0), vector(96, 0)))
        self.assertEqual(candidate_2[2], (vector(0, 0), vector(72, 0)))
        self.assertEqual(large_logit_ratio(committed_1, 100, mask), 0)
        self.assertEqual(large_logit_ratio(candidate_2[2], 100, mask), 0)

        wrong_shared = score_matrix(
            projection_query(Fraction(5, 2)), projection_key(10), 1
        )
        wrong_1 = (replay_1[0][1][0] + wrong_shared[1][0]) / 2
        wrong_2_rotary = score_matrix(projection_query(2), projection_key(10), 1)
        wrong_2 = (observed_2[0][1][0] + wrong_2_rotary[1][0]) / 2
        self.assertEqual((wrong_1, wrong_2), (Fraction(125, 2), Fraction(50)))

        with self.assertRaises(ValueError):
            large_logit_ratio(observed_1[2], 100, ((True,), (True, True)))

    def test_lagged_factor_does_not_hard_bound_changed_candidate(self) -> None:
        record = lag_counterexample()
        self.assertEqual(record["observed"], 400)
        self.assertEqual(record["factor"], Fraction(1, 4))
        self.assertEqual(record["candidate"], 1600)
        self.assertEqual(record["committed"], 400)
        self.assertGreater(record["committed"], record["threshold"])

    def test_distributed_max_consensus_is_headwise(self) -> None:
        record = distributed_consensus(((400, 80), (64, 120)), 100)
        self.assertEqual(record["consensus_maxima"], (Fraction(400), Fraction(120)))
        self.assertEqual(record["common_factors"], (Fraction(1, 4), Fraction(5, 6)))
        self.assertNotEqual(record["local_factors"][0], record["local_factors"][1])

    def test_trigger_statistics_keep_metrics_separate(self) -> None:
        history = (
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
        )
        record = trigger_statistics(history)
        self.assertEqual(record["instantaneous_rates"], (Fraction(1, 4),) * 4)
        self.assertEqual(record["cumulative_ever_triggered"], 1)
        self.assertEqual(record["per_head_counts"], (1, 1, 1, 1))

    def test_maximum_and_tail_ratio_answer_different_questions(self) -> None:
        one_outlier = ((400,) + (0,) * 9999,)
        dense_tail = ((101,) * 1000 + (0,) * 9000,)
        self.assertGreater(
            scaled_positive_max(one_outlier), scaled_positive_max(dense_tail)
        )
        self.assertLess(
            large_logit_ratio(one_outlier, 100),
            large_logit_ratio(dense_tail, 100),
        )

    def test_soft_cap_changes_exposed_value_not_raw_input(self) -> None:
        raw = 400.0
        exposed = soft_cap(raw, 50.0)
        self.assertEqual(raw, 400.0)
        self.assertLess(exposed, 50.0)
        self.assertTrue(isclose(exposed, 50.0 * __import__("math").tanh(8.0)))


if __name__ == "__main__":
    unittest.main()
