from __future__ import annotations

import math
import unittest

from .reference import Cache, Decoder, Scalar, TARGETS, TOKENS, adamw_step, loss


class TransformerFoundationTests(unittest.TestCase):
    def test_full_matches_each_cached_row_and_every_prefix_split(self) -> None:
        model = Decoder()
        tokens = (0, 1, 2, 1, 3)
        full, _ = model.full(tokens)
        for split in range(len(tokens) + 1):
            cache = Cache()
            for chunk in (tokens[:split], tokens[split:]):
                for token in chunk:
                    output = model.decode(token, cache)
                    expected = full[len(cache.keys) - 1]
                    for actual, wanted in zip(output, expected, strict=True):
                        self.assertAlmostEqual(actual.value, wanted.value, places=12)
            self.assertEqual(len(cache.keys), len(tokens))
            self.assertEqual(sum(len(row) for row in cache.keys + cache.values), 40)

    def test_future_token_changes_do_not_change_earlier_logits(self) -> None:
        model = Decoder()
        first, _ = model.full((0, 1, 2))
        second, _ = model.full((0, 1, 3))
        self.assertEqual(
            [[value.value for value in row] for row in first[:2]],
            [[value.value for value in row] for row in second[:2]],
        )
        self.assertNotEqual(
            [value.value for value in first[-1]], [value.value for value in second[-1]]
        )

    def test_known_first_position_and_mean_loss(self) -> None:
        logits, traces = Decoder().full(TOKENS)
        self.assertEqual([value.value for value in logits[0]], [2.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(traces[0]["ffn"][0].value, 1 / (1 + math.exp(-2)))
        self.assertAlmostEqual(loss([logits[0]], (1,)).value, math.log(math.exp(2) + 3))
        expected = (
            sum(
                loss([row], (target,)).value
                for row, target in zip(logits, TARGETS, strict=True)
            )
            / 3
        )
        self.assertAlmostEqual(loss(logits, TARGETS).value, expected)
        self.assertAlmostEqual(expected, 2.351615949748191, places=12)
        printed_rows = (
            (2.0, 0.0, 0.0, 0.0),
            (0.00242090, 1.99999853, 0.0, 0.0),
            (0.25366462, 0.25366462, 1.96756411, 0.0),
        )
        for row, wanted in zip(logits, printed_rows, strict=True):
            for value, target in zip(row, wanted, strict=True):
                self.assertAlmostEqual(value.value, target, delta=1e-8)

    def test_every_parameter_gradient_matches_central_difference(self) -> None:
        model = Decoder()
        logits, _ = model.full(TOKENS)
        loss(logits, TARGETS).backward()
        parameters = model.parameters()
        analytical = [parameter.grad for parameter in parameters]
        self.assertEqual(len(parameters), 144)
        for parameter, expected in zip(parameters, analytical, strict=True):
            original = parameter.value
            parameter.value = original + 1e-5
            plus = loss(model.full(TOKENS)[0], TARGETS).value
            parameter.value = original - 1e-5
            minus = loss(model.full(TOKENS)[0], TARGETS).value
            parameter.value = original
            self.assertAlmostEqual(expected, (plus - minus) / 2e-5, delta=2e-7)

    def test_one_update_and_fresh_model_cache(self) -> None:
        model = Decoder()
        objective = loss(model.full(TOKENS)[0], TARGETS)
        objective.backward()
        before = objective.value
        parameters = model.parameters()
        moments = adamw_step(parameters, [(0.0, 0.0)] * len(parameters), 1)
        after = loss(model.full(TOKENS)[0], TARGETS).value
        self.assertLess(after, before)
        self.assertAlmostEqual(after, 2.2528997739491583, places=12)
        self.assertTrue(any(variance > 0 for _, variance in moments))
        cache = Cache()
        for token in TOKENS:
            decoded = model.decode(token, cache)
        full = model.full(TOKENS)[0][-1]
        for actual, expected in zip(decoded, full, strict=True):
            self.assertAlmostEqual(actual.value, expected.value, places=12)

    def test_shared_graph_and_adamw_scalar_witness(self) -> None:
        value = Scalar(2.0)
        shared = value * value
        (shared + shared * value).backward()
        self.assertEqual(value.grad, 16.0)
        parameter = Scalar(2.0)
        parameter.grad = 0.5
        moments = adamw_step([parameter], [(0.0, 0.0)], 1)
        self.assertAlmostEqual(moments[0][0], 0.05)
        self.assertAlmostEqual(moments[0][1], 0.00025)
        self.assertAlmostEqual(parameter.value, 1.9880000002)


if __name__ == "__main__":
    unittest.main()
