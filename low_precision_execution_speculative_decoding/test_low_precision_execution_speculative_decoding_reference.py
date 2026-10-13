import unittest
from fractions import Fraction
from math import log
import struct

from low_precision_execution_speculative_decoding.dspark import (
    kda_step_record,
    markov_draft,
    markov_proposal_law,
    predicted_output_rate,
    prefix_survival,
    reconstruct_kda,
    recover_convolution_history,
    selected_proposal_output_law,
    window_positions,
)

from low_precision_execution_speculative_decoding.reference import (
    exact_one_step_output_law,
    greedy_commit,
    normalized_speedup,
    overlap_mass,
    positive_residual,
    state_action,
    quantize_mxfp4,
    decode_mxfp4,
    inverse_cdf,
    stochastic_commit,
    replay_scalar,
    StateIdentity,
    validate_state_bundle,
)


TARGET = {"a": 0.45, "b": 0.20, "c": 0.25, "eos": 0.10}
DRAFT = {"a": 0.30, "b": 0.40, "c": 0.20, "eos": 0.10}


class SpeculativeDecodingReferenceTests(unittest.TestCase):
    def test_overlap_and_residual_fixture(self) -> None:
        self.assertAlmostEqual(overlap_mass(TARGET, DRAFT), 0.80)
        residual = positive_residual(TARGET, DRAFT)
        expected = {"a": 0.75, "b": 0.0, "c": 0.25, "eos": 0.0}
        for token, probability in expected.items():
            self.assertAlmostEqual(residual[token], probability)

    def test_acceptance_and_correction_recover_target_law(self) -> None:
        recovered = exact_one_step_output_law(TARGET, DRAFT)
        for token, probability in TARGET.items():
            self.assertAlmostEqual(recovered[token], probability)

    def test_greedy_rejection_commits_prefix_and_target_choice(self) -> None:
        committed, accepted = greedy_commit(("a", "b", "c"), ("a", "x", "z", "bonus"))
        self.assertEqual(committed, ("a", "x"))
        self.assertEqual(accepted, 1)

    def test_all_accepted_chain_commits_bonus(self) -> None:
        committed, accepted = greedy_commit(("a", "b", "c"), ("a", "b", "c", "x"))
        self.assertEqual(committed, ("a", "b", "c", "x"))
        self.assertEqual(accepted, 3)

    def test_state_action_and_cost_fixture(self) -> None:
        action = state_action(proposal_length=3, accepted_length=1)
        self.assertEqual(action.accepted_draft_tokens, 1)
        self.assertEqual(action.final_token_kind, "correction")
        self.assertEqual(action.committed_tokens, 2)
        self.assertEqual(action.rejected_draft_tokens, 2)
        self.assertAlmostEqual(normalized_speedup(2.4, 0.18, 1.25, 0.07), 1.60)

    def test_quantized_block_dot_and_codes(self) -> None:
        values = (1.2, 2.7, 5.8, 11.4) + (0.0,) * 28
        scale, codes = quantize_mxfp4(values, 2.0)
        self.assertEqual((scale, codes[:4]), (128, (1, 3, 5, 7)))
        decoded = decode_mxfp4(scale, codes)
        self.assertEqual(decoded, (1, 3, 6, 12) + (0,) * 28)
        weights = (1.0, -1.0, 0.5, 0.0) + (0.0,) * 28
        self.assertEqual(quantize_mxfp4(weights, 1.0)[:1], (127,))
        self.assertEqual(sum(x * w for x, w in zip(decoded, weights, strict=True)), 1.0)
        real = sum(
            Fraction(str(x)) * Fraction(str(w))
            for x, w in zip(values, weights, strict=True)
        )
        self.assertEqual(real, Fraction(7, 5))
        self.assertEqual(32 * 4 + 8, 136)

    def test_quantization_ties_clipping_sign_and_exceptions(self) -> None:
        scale, codes = quantize_mxfp4((1.25, 1.75, -1.25, 15) + (0,) * 28, 1.0)
        self.assertEqual(decode_mxfp4(scale, codes)[:4], (1, 2, -1, 6))
        self.assertEqual(decode_mxfp4(*quantize_mxfp4((15,) * 32, 2))[0], 12)
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                quantize_mxfp4((invalid,) * 32, 1.0)
        with self.assertRaises(ValueError):
            quantize_mxfp4((0,) * 32, 3.0)

    def test_binary32_order_and_parity_counterexamples(self) -> None:
        def fp32(value: float) -> float:
            return struct.unpack("f", struct.pack("f", value))[0]

        self.assertEqual(fp32(fp32(2**24 + 1) - 2**24), 0)
        self.assertEqual(fp32(fp32(2**24 - 2**24) + 1), 1)
        reference, deployed = (0.51, 0.49), (0.49, 0.51)
        self.assertAlmostEqual(
            sum(abs(p - q) for p, q in zip(reference, deployed, strict=True)) / 2,
            0.02,
        )
        self.assertNotEqual(
            reference.index(max(reference)), deployed.index(max(deployed))
        )

    def test_stochastic_rejection_and_eager_state(self) -> None:
        proposal = tuple(inverse_cdf(DRAFT, draw) for draw in (0.1, 0.5, 0.8))
        self.assertEqual(proposal, ("a", "b", "c"))
        committed, accepted = stochastic_commit(
            proposal, [TARGET] * 4, [DRAFT] * 3, (0.2, 0.64, 0.2), 0.6, 10
        )
        self.assertEqual((committed, accepted), (("a", "a"), 1))
        embeddings = {"a": 1, "b": 2, "c": 3, "eos": 0}
        self.assertEqual(
            [replay_scalar(2, proposal[:i], embeddings) for i in (1, 2, 3)], [5, 12, 27]
        )
        self.assertEqual(replay_scalar(2, committed, embeddings), 11)
        self.assertEqual(replay_scalar(5, committed[1:], embeddings), 11)

    def test_all_accepted_bonus_eos_and_output_caps(self) -> None:
        bonus = {"a": 0.1, "b": 0.2, "c": 0.6, "eos": 0.1}
        args = (
            ("a", "b", "c"),
            [TARGET] * 3 + [bonus],
            [DRAFT] * 3,
            (0.2, 0.4, 0.2),
            0.95,
        )
        self.assertEqual(stochastic_commit(*args, 10), (("a", "b", "c", "eos"), 3))
        self.assertEqual(stochastic_commit(*args, 2), (("a", "b"), 2))
        self.assertEqual(stochastic_commit(*args, 0), ((), 0))
        self.assertEqual(
            stochastic_commit(("eos",), [TARGET] * 2, [DRAFT], (0.2,), 0.1, 9),
            (("eos",), 1),
        )
        self.assertEqual(
            stochastic_commit(("b",), [TARGET] * 2, [DRAFT], (0.5,), 0.6, 1),
            (("a",), 0),
        )
        self.assertEqual(stochastic_commit((), [bonus], [], (), 0.95, 3), (("eos",), 0))
        target_eos, draft_a = {"a": 0, "eos": 1}, {"a": 1, "eos": 0}
        self.assertEqual(
            stochastic_commit(("a",), [target_eos] * 2, [draft_a], (0.2,), 0.5, 8),
            (("eos",), 0),
        )
        for rejection_depth in range(3):
            targets = [DRAFT] * rejection_depth + [TARGET] * (4 - rejection_depth)
            suffix, accepted = stochastic_commit(
                ("b",) * 3, targets, [DRAFT] * 3, (0.64,) * 3, 0.6, 9
            )
            self.assertEqual(
                (suffix, accepted), (("b",) * rejection_depth + ("a",), rejection_depth)
            )

    def test_support_and_probability_validation(self) -> None:
        self.assertEqual(exact_one_step_output_law(TARGET, TARGET), TARGET)
        with self.assertRaises(ValueError):
            positive_residual(TARGET, TARGET)
        self.assertEqual(
            exact_one_step_output_law({"a": 0, "b": 1}, {"a": 1, "b": 0}),
            {"a": 0, "b": 1},
        )
        with self.assertRaises(ValueError):
            overlap_mass({"a": float("nan")}, {"a": 1})
        self.assertEqual(inverse_cdf({"a": 0.5, "b": 0.5}, 0.5), "b")
        self.assertEqual(inverse_cdf({"a": 0, "b": 1}, 0), "b")

    def test_exact_law_on_all_small_rational_pairs(self) -> None:
        laws = [
            {"a": a / 4, "b": b / 4, "c": (4 - a - b) / 4}
            for a in range(5)
            for b in range(5 - a)
        ]
        for target in laws:
            for draft in laws:
                for token, mass in exact_one_step_output_law(target, draft).items():
                    self.assertAlmostEqual(mass, target[token])

    def test_row_permutation_fails_before_publication(self) -> None:
        r = StateIdentity("R", "prefix-r", "target-v1", 3)
        s = StateIdentity("S", "prefix-s", "target-v1", 3)
        wrong_prefix = StateIdentity("S", "prefix-other", "target-v1", 3)
        wrong_frontier = StateIdentity("S", "prefix-s", "target-v1", 4)
        published: list[str] = []
        with self.assertRaises(ValueError):
            validate_state_bundle(s, [s, r, r])
            published.append("wrong-history")
        self.assertEqual(published, [])
        for stale in (wrong_prefix, wrong_frontier):
            with self.assertRaises(ValueError):
                validate_state_bundle(s, [s, stale])
        validate_state_bundle(s, [s, s, s])
        validate_state_bundle(r, [r, r, r])

    def test_population_cost_is_not_single_trace(self) -> None:
        population = (0.2, 0.4, 0.2, 0.2)
        mean = sum(index * mass for index, mass in enumerate(population))
        self.assertAlmostEqual(mean, 1.4)
        self.assertAlmostEqual(normalized_speedup(mean + 1, 0.18, 1.25, 0.07), 1.6)
        self.assertAlmostEqual(normalized_speedup(1.4, 0.18, 1.25, 0.07), 14 / 15)


class DSparkReferenceTests(unittest.TestCase):
    def test_markov_coupling_and_processed_proposal_mass(self) -> None:
        base = {"a": 0.0, "b": 0.0, "c": 0.0}
        bias = {"a": log(2), "b": log(3), "c": 0.0}
        actual = markov_proposal_law(base, bias, 2.0, ("a", "c"))
        target = {"a": 0.2, "b": 0.3, "c": 0.5}
        self.assertEqual(actual["b"], 0.0)
        self.assertAlmostEqual(actual["a"], 2**0.5 / (1 + 2**0.5))
        retained = selected_proposal_output_law(
            target, actual, dict.fromkeys(base, 1.0)
        )
        for token in target:
            self.assertAlmostEqual(retained[token], target[token])
        wrong = selected_proposal_output_law(
            target, actual, dict.fromkeys(base, 1.0), dict.fromkeys(base, 1 / 3)
        )
        self.assertGreater(max(abs(wrong[t] - target[t]) for t in target), 0.01)
        self.assertNotEqual(
            markov_proposal_law(base, bias), markov_proposal_law(base, base)
        )

    def test_dspark_extends_the_existing_rejection_trace(self) -> None:
        base = {t: log(q) for t, q in DRAFT.items()}
        zero = dict.fromkeys(DRAFT, 0.0)
        plus = {**zero, "a": log(2)}
        minus = {**zero, "a": -log(2)}
        transitions = {"anchor": zero, "a": plus, "b": minus, "c": zero, "eos": zero}
        block = [
            base,
            {t: base[t] - plus[t] for t in base},
            {t: base[t] - minus[t] for t in base},
        ]
        proposals, laws = markov_draft(block, transitions, "anchor", (0.1, 0.5, 0.8))
        self.assertEqual(proposals, ("a", "b", "c"))
        for law in laws:
            for token in DRAFT:
                self.assertAlmostEqual(law[token], DRAFT[token])
        self.assertEqual(
            stochastic_commit(proposals, [TARGET] * 4, laws, (0.2, 0.64, 0.2), 0.6, 10),
            (("a", "a"), 1),
        )
        other = markov_proposal_law(block[1], transitions["b"])
        self.assertNotEqual(other, laws[1])

    def test_admission_bias_and_all_small_constant_controls(self) -> None:
        target, draft = {"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}
        biased = selected_proposal_output_law(target, draft, {"A": 1.0, "B": 0.0})
        self.assertAlmostEqual(biased["A"], 0.8)
        self.assertAlmostEqual(biased["B"], 0.2)
        # 5 target laws × 3 full-support draft laws × 5 constant admission rates.
        for p in range(5):
            for q in range(1, 4):
                for admission in range(5):
                    target = {"A": p / 4, "B": 1 - p / 4}
                    draft = {"A": q / 4, "B": 1 - q / 4}
                    result = selected_proposal_output_law(
                        target, draft, dict.fromkeys(target, admission / 4)
                    )
                    for token in target:
                        self.assertAlmostEqual(result[token], target[token])

    def test_confidence_allocation_and_elapsed_cost(self) -> None:
        a = prefix_survival((0.9, 0.8, 0.5))
        b = prefix_survival((0.6, 0.3, 0.2))
        short = predicted_output_rate((a, b), (2, 1), 1.5)
        full = predicted_output_rate((a, b), (3, 3), 2.0)
        self.assertAlmostEqual(short[0], 4.22)
        self.assertAlmostEqual(full[0], 4.796)
        self.assertGreater(short[1], full[1])
        total_rate = (short[0] + full[0]) / 3.5
        self.assertNotAlmostEqual(total_rate, (short[1] + full[1]) / 2)
        self.assertEqual(predicted_output_rate((a, b), (0, 0), 1), (2, 2))
        with self.assertRaises(ValueError):
            predicted_output_rate(((0.5, 0.7),), (2,), 1)

    def test_direct_reconstruction_matches_causal_replay_at_every_prefix(self) -> None:
        initial = ((1.0, 2.0), (3.0, 4.0))
        inputs = [
            ((0.5, 0.25), (1.0, 0.0), (2.0, 1.0), 0.5),
            ((0.25, 0.5), (0.0, 1.0), (1.0, 3.0), 0.5),
            ((0.5, 0.5), (1.0, 0.0), (0.0, 2.0), 0.25),
        ]
        snapshots, records = [initial], []
        for values in inputs:
            state, record = kda_step_record(snapshots[-1], *values)
            snapshots.append(state)
            records.append(record)
        self.assertEqual(snapshots[1], ((1.25, 0.5), (1.25, 1.0)))
        self.assertEqual(snapshots[2], ((0.3125, 0.625), (0.3125, 1.75)))
        for accepted in range(4):
            recovered = reconstruct_kda(initial, records, accepted)
            self.assertEqual(recovered, snapshots[accepted])
        # A correction after input 1 needs a fresh record at that boundary.
        corrected, correction = kda_step_record(snapshots[1], *inputs[2])
        repaired = reconstruct_kda(initial, [records[0], correction], 2)
        self.assertEqual(repaired, corrected)
        stale = reconstruct_kda(initial, [records[0], records[2]], 2)
        self.assertNotEqual(stale, corrected)

    def test_nondyadic_reconstruction_has_declared_tolerance(self) -> None:
        initial = ((0.1, -0.7), (0.2, 0.9))
        states, records = [initial], []
        for index in range(8):
            state, record = kda_step_record(
                states[-1], (0.93, 0.87), (0.6, 0.8), (0.31, -0.17), 0.11 + index / 100
            )
            states.append(state)
            records.append(record)
        for accepted in range(9):
            recovered = reconstruct_kda(initial, records, accepted)
            for row, expected in zip(recovered, states[accepted], strict=True):
                for value, reference in zip(row, expected, strict=True):
                    self.assertLessEqual(abs(value - reference), 1e-12)

    def test_anchor_counts_and_cache_boundaries(self) -> None:
        # All acceptance lengths for 3-, 7- and 8-token windows: 21 geometries.
        for width in (3, 7, 8):
            for accepted in range(width + 1):
                ledger = window_positions(width, accepted, consumed_before=2)
                self.assertEqual(ledger.draft_queries, width)
                self.assertEqual(ledger.target_inputs, width + 1)
                self.assertEqual(ledger.recovered_inputs, accepted + 1)
                self.assertEqual(ledger.visible_frontier, ledger.consumed_frontier + 1)
        self.assertEqual(window_positions(3, 0, 2).cache_boundaries, ())
        self.assertEqual(window_positions(3, 1, 2).cache_boundaries, (4,))
        self.assertEqual(window_positions(3, 2, 2).cache_boundaries, (4,))

    def test_convolution_and_request_permutation_follow_the_consumed_prefix(
        self,
    ) -> None:
        before, inputs = ("old", "previous"), ("anchor", "a", "b", "c")
        for consumed in range(5):
            expected = (before + inputs[:consumed])[-2:]
            self.assertEqual(
                recover_convolution_history(before, inputs, consumed), expected
            )
        history = recover_convolution_history(before, inputs, 2)
        self.assertEqual(history, ("anchor", "a"))
        self.assertEqual(
            recover_convolution_history(history, ("correction",), 1),
            ("a", "correction"),
        )
        owners = [
            StateIdentity("r", "Paa", "fixed", 4),
            StateIdentity("s", "Qbb", "fixed", 4),
        ]
        histories = [(owners[0], history), (owners[1], ("anchor-s", "b"))]
        for owner, (identity, _) in zip(
            reversed(owners), reversed(histories), strict=True
        ):
            validate_state_bundle(owner, (identity,))
        with self.assertRaises(ValueError):
            validate_state_bundle(owners[1], (histories[0][0],))


if __name__ == "__main__":
    unittest.main()
