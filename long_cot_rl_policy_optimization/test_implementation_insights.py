"""Partition and normalized-policy checks for Chapter 19's new investigation."""

from __future__ import annotations

from fractions import Fraction
import importlib
import math
from typing import cast
import unittest

from . import reference
from .implementation_insights import (
    Fragment,
    distinct_rollout_advantages,
    local_score_terms,
    normalized_policy_witness,
    reduce_fragments,
    whole_rollout_denominators,
)


def studio_fragments() -> list[Fragment]:
    return [
        Fragment("studio", "y1", 8, (0, 1), 1),
        Fragment("studio", "y1", 8, tuple(range(2, 8)), 1),
        Fragment("studio", "y2", 12, tuple(range(12)), 1),
        Fragment("studio", "y3", 8, tuple(range(8)), 0),
        Fragment("studio", "y4", 20, tuple(range(20)), 1),
    ]


def cut_partition(bits: int) -> tuple[tuple[int, ...], ...]:
    blocks: list[tuple[int, ...]] = []
    start = 0
    for position in range(7):
        if bits & (1 << position):
            blocks.append(tuple(range(start, position + 1)))
            start = position + 1
    blocks.append(tuple(range(start, 8)))
    return tuple(blocks)


class _Scalar:
    """Tiny reverse-mode graph for independent fragment backward calls."""

    def __init__(
        self, value: float, parents: tuple[tuple[_Scalar, float], ...] = ()
    ) -> None:
        self.value = float(value)
        self.parents = parents
        self.grad = 0.0

    def __add__(self, other: _Scalar | float) -> _Scalar:
        rhs = other if isinstance(other, _Scalar) else _Scalar(other)
        return _Scalar(self.value + rhs.value, ((self, 1.0), (rhs, 1.0)))

    __radd__ = __add__

    def __mul__(self, other: _Scalar | float) -> _Scalar:
        rhs = other if isinstance(other, _Scalar) else _Scalar(other)
        return _Scalar(self.value * rhs.value, ((self, rhs.value), (rhs, self.value)))

    __rmul__ = __mul__

    def __neg__(self) -> _Scalar:
        return -1.0 * self

    def exp(self) -> _Scalar:
        value = math.exp(self.value)
        return _Scalar(value, ((self, value),))

    def log(self) -> _Scalar:
        return _Scalar(math.log(self.value), ((self, 1.0 / self.value),))

    def backward(self) -> None:
        ordered: list[_Scalar] = []
        seen: set[int] = set()

        def visit(node: _Scalar) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent, _ in node.parents:
                visit(parent)
            ordered.append(node)

        visit(self)
        self.grad += 1.0
        for node in reversed(ordered):
            for parent, weight in node.parents:
                parent.grad += node.grad * weight


def _fragment_score(
    theta: tuple[_Scalar, _Scalar], positions: tuple[int, ...]
) -> _Scalar:
    total = _Scalar(0.0)
    for position in positions:
        x, y = (6.0, 0.0) if position < 2 else (-2.0, 2.0)
        logit = x * theta[0] + y * theta[1]
        ell = -(1.0 + (-logit).exp()).log() + math.log(2.0)
        if not -0.2 <= ell.value <= 0.2:
            raise ValueError("diagnostic starts inside the declared locality bounds")
        total = total + 0.25 * ell.exp() + -0.1 * ell * ell
    return total


def _separate_backward(
    partitions: tuple[tuple[int, ...], ...], mode: str = "token"
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    if sorted(position for block in partitions for position in block) != list(range(8)):
        raise ValueError("partitions must contain the eight positions once")
    theta = (_Scalar(0.0), _Scalar(0.0))
    score = 0.0
    for block in partitions:
        numerator = _fragment_score(theta, block)  # A fresh graph at fixed theta.
        if mode == "token":
            fragment = numerator * (1.0 / 8.0)
        elif mode == "fragment" and block:
            fragment = numerator * (1.0 / (len(partitions) * len(block)))
        else:
            raise ValueError("a fragment mean requires a nonempty row")
        score += fragment.value
        fragment.backward()  # Accumulate gradients; do not step between rows.
    gradient = (theta[0].grad, theta[1].grad)
    step = tuple(0.4 * value for value in gradient)
    return score, gradient, (step[0], step[1])


class ImplementationInsightsTests(unittest.TestCase):
    def test_distinct_centering_and_prompt_collision(self) -> None:
        rows = studio_fragments() + [
            Fragment("other", "y1", 2, (0, 1), 0),
            Fragment("other", "y2", 2, (0, 1), 1),
        ]
        means, advantage = distinct_rollout_advantages(rows, {"studio": 4, "other": 2})
        self.assertEqual(means, {"studio": Fraction(3, 4), "other": Fraction(1, 2)})
        self.assertEqual(advantage[("studio", "y1")], Fraction(1, 4))
        self.assertEqual(advantage[("studio", "y3")], Fraction(-3, 4))
        self.assertEqual(advantage[("other", "y1")], Fraction(-1, 2))
        self.assertEqual(len(advantage), 6)
        self.assertEqual(whole_rollout_denominators(rows[:5]), (8, 8, 12, 8, 20))

        row_rewards = [Fraction(row.outcome) for row in rows[:5]]
        wrong_mean = sum(row_rewards) / len(row_rewards)
        wrong_advantages = tuple(value - wrong_mean for value in row_rewards)
        self.assertEqual(wrong_mean, Fraction(4, 5))
        self.assertEqual(
            wrong_advantages, (Fraction(1, 5),) * 3 + (Fraction(-4, 5), Fraction(1, 5))
        )
        self.assertEqual(sum(wrong_advantages), 0)

    def test_partition_validation_and_empty_physical_row(self) -> None:
        rows = studio_fragments()
        empty = Fragment("studio", "y1", 8, (), 1)
        self.assertEqual(whole_rollout_denominators(rows + [empty])[-1], 8)
        values = [[1.0] * len(row.positions) for row in rows]
        self.assertEqual(
            reduce_fragments(rows, values, "token"),
            reduce_fragments(rows + [empty], values + [[]], "token"),
        )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            whole_rollout_denominators(rows + [Fragment("studio", "y1", 8, (0,), 1)])
        with self.assertRaisesRegex(ValueError, "missing"):
            whole_rollout_denominators(rows[1:])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            whole_rollout_denominators(rows + [Fragment("studio", "y1", 8, (), 0)])
        with self.assertRaisesRegex(ValueError, "promised"):
            distinct_rollout_advantages(rows, {"studio": 5})
        with self.assertRaisesRegex(ValueError, "positive"):
            whole_rollout_denominators([Fragment("studio", "y1", 0, (), 1)])

    def test_chosen_reduction_and_original_fixture(self) -> None:
        rows = studio_fragments()
        ratios = reference.STUDIO_COMPLETE_LOG_RATIOS
        _, advantages = reference.group_advantages(reference.STUDIO_REWARDS)
        score_rows: list[tuple[float, ...]] = []
        coefficient_rows: list[tuple[float, ...]] = []
        for row in rows:
            index = int(row.rollout_id[1:]) - 1
            row_ratios = tuple(ratios[index][position] for position in row.positions)
            scores, coefficients, _ = local_score_terms(
                row_ratios,
                (0.0,) * len(row_ratios),
                (float(advantages[index]),) * len(row_ratios),
                (True,) * len(row_ratios),
            )
            score_rows.append(scores)
            coefficient_rows.append(coefficients)
        original = reference.complete_studio_score()
        self.assertEqual(original["N"], 48)
        self.assertEqual(original["U"], 45)
        self.assertAlmostEqual(
            reduce_fragments(rows, score_rows, "token"), cast(float, original["score"])
        )
        self.assertAlmostEqual(
            reduce_fragments(rows, coefficient_rows, "token"),
            cast(float, original["normalized_coefficient_sum"]),
        )
        self.assertNotAlmostEqual(
            reduce_fragments(rows, score_rows, "token"),
            reduce_fragments(rows, score_rows, "trajectory"),
        )

    def test_token_and_trajectory_are_distinct_objectives(self) -> None:
        rows = [
            Fragment("example", "short", 2, (0, 1), 1),
            Fragment("example", "long", 6, tuple(range(6)), 0),
        ]
        self.assertEqual(reduce_fragments(rows, ((1, 1), (0,) * 6), "token"), 0.25)
        self.assertEqual(reduce_fragments(rows, ((1, 1), (0,) * 6), "trajectory"), 0.5)

    def test_every_small_partition_preserves_score_and_gradient(self) -> None:
        for bits in range(128):
            score, gradient = normalized_policy_witness(cut_partition(bits))
            self.assertAlmostEqual(score, 0.25, places=12)
            self.assertAlmostEqual(gradient[0], 0.0, places=12)
            self.assertAlmostEqual(gradient[1], 3 / 16, places=12)
        score, gradient = normalized_policy_witness(((0, 2, 4, 6), (1, 3, 5, 7)))
        self.assertAlmostEqual(score, 0.25)
        self.assertAlmostEqual(gradient[1], 3 / 16)
        wrong_score, wrong_gradient = normalized_policy_witness(
            ((0, 1), tuple(range(2, 8))), mode="fragment"
        )
        self.assertAlmostEqual(wrong_score, 0.25)
        self.assertAlmostEqual(wrong_gradient[0], 1 / 4)
        self.assertAlmostEqual(wrong_gradient[1], 1 / 8)

    def test_separate_microbatch_backward_then_one_step(self) -> None:
        full = _separate_backward((tuple(range(8)),))
        self.assertAlmostEqual(full[0], 0.25, places=12)
        for got, want in zip(full[1], (0.0, 3 / 16), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(full[2], (0.0, 0.075), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for blocks in [
            *((cut_partition(bits)) for bits in range(128)),
            (tuple(range(2, 8)), (0, 1)),
            ((0, 2, 4, 6), (1, 3, 5, 7)),
        ]:
            with self.subTest(blocks=blocks):
                score, gradient, step = _separate_backward(blocks)
                self.assertAlmostEqual(score, full[0], places=12)
                for got, want in zip(gradient, full[1], strict=True):
                    self.assertAlmostEqual(got, want, places=12)
                for got, want in zip(step, full[2], strict=True):
                    self.assertAlmostEqual(got, want, places=12)
        wrong = _separate_backward(((0, 1), tuple(range(2, 8))), "fragment")
        self.assertAlmostEqual(wrong[0], full[0])
        for got, want in zip(wrong[1], (1 / 4, 1 / 8), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(wrong[2], (0.1, 0.05), strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_empty_physical_fragments_preserve_backward_and_step(self) -> None:
        base = ((0, 1), tuple(range(2, 8)))
        expected = _separate_backward(base)
        studio = studio_fragments()
        mean, advantage = distinct_rollout_advantages(
            studio + [Fragment("studio", "y1", 8, (), 1)], {"studio": 4}
        )
        self.assertEqual(mean["studio"], Fraction(3, 4))
        self.assertEqual(len(advantage), 4)
        self.assertEqual(
            sum({row.rollout_id: row.rollout_length for row in studio}.values()), 48
        )
        for blocks in (
            ((),) + base,
            (base[0], (), base[1]),
            base + ((),),
            ((), base[0], (), base[1], ()),
        ):
            with self.subTest(blocks=blocks):
                rows = [Fragment("diagnostic", "y1", 8, block, 1) for block in blocks]
                self.assertEqual(whole_rollout_denominators(rows), (8,) * len(rows))
                score, gradient, step = _separate_backward(blocks)
                self.assertAlmostEqual(score, expected[0], places=12)
                for got, want in zip(gradient, expected[1], strict=True):
                    self.assertAlmostEqual(got, want, places=12)
                for got, want in zip(step, expected[2], strict=True):
                    self.assertAlmostEqual(got, want, places=12)

    def test_wrong_whole_score_mask_loses_restoring_gradient(self) -> None:
        for advantage in (0.25, -0.75):
            for tail in (-0.3, 0.3):
                with self.subTest(advantage=advantage, tail=tail):
                    mask = (True,)
                    _, coefficient, denominator = local_score_terms(
                        (tail,), (0.0,), (advantage,), mask
                    )
                    correct_ell = _Scalar(tail)
                    held = math.exp(min(max(tail, -0.2), 0.2))
                    correct = (
                        _Scalar(advantage * held) + -0.1 * correct_ell * correct_ell
                    ) * (float(mask[0]) / denominator)
                    correct.backward()
                    wrong_ell = _Scalar(tail)
                    whole_score = (
                        _Scalar(advantage * held) + -0.1 * wrong_ell * wrong_ell
                    )
                    wrong = whole_score * (
                        float(mask[0]) * float(-0.2 <= tail <= 0.2) / denominator
                    )
                    wrong.backward()
                    self.assertEqual(denominator, sum(mask))
                    self.assertAlmostEqual(correct_ell.grad, coefficient[0], places=12)
                    self.assertAlmostEqual(correct_ell.grad, -0.2 * tail)
                    self.assertEqual(wrong_ell.grad, 0.0)
                    self.assertNotEqual(wrong_ell.grad, correct_ell.grad)

    def test_parameter_gradient_matches_score_difference(self) -> None:
        partitions = ((0, 1), tuple(range(2, 8)))
        score, gradient = normalized_policy_witness(partitions)
        self.assertAlmostEqual(score, 0.25)
        delta = 1e-6
        for coordinate in range(2):
            direction = [0.0, 0.0]
            direction[coordinate] = delta
            plus = normalized_policy_witness(
                partitions, theta=(direction[0], direction[1])
            )[0]
            direction[coordinate] = -delta
            minus = normalized_policy_witness(
                partitions, theta=(direction[0], direction[1])
            )[0]
            self.assertAlmostEqual(
                (plus - minus) / (2 * delta), gradient[coordinate], places=8
            )

    def test_reward_tails_endpoints_and_masked_padding(self) -> None:
        score, derivative, count = local_score_terms(
            (float("nan"), 0.3, -0.3),
            (float("nan"), 0.0, 0.0),
            (float("nan"), 0.25, -0.75),
            (False, True, True),
        )
        self.assertEqual(count, 2)
        self.assertEqual((score[0], derivative[0]), (0.0, 0.0))
        self.assertAlmostEqual(derivative[1], -0.06)
        self.assertAlmostEqual(derivative[2], 0.06)
        for endpoint, side in ((-0.2, 1), (0.2, -1)):
            values, derivatives, _ = local_score_terms(
                (endpoint,), (0.0,), (0.25,), (True,)
            )
            self.assertAlmostEqual(
                derivatives[0], 0.25 * math.exp(endpoint) - 0.2 * endpoint
            )
            nearby = endpoint + side * 1e-6
            other = local_score_terms((nearby,), (0.0,), (0.25,), (True,))[0][0]
            one_sided = (other - values[0]) / (side * 1e-6)
            self.assertAlmostEqual(one_sided, derivatives[0], places=5)
        with self.assertRaisesRegex(ValueError, "finite"):
            local_score_terms((float("nan"),), (0.0,), (0.25,), (True,))
        with self.assertRaisesRegex(ValueError, "no retained"):
            local_score_terms((0.0,), (0.0,), (0.25,), (False,))
        extreme_score, extreme_derivative, _ = local_score_terms(
            (1000.0,), (0.0,), (0.25,), (True,)
        )
        self.assertTrue(math.isfinite(extreme_score[0]))
        self.assertAlmostEqual(extreme_derivative[0], -200.0)

    def test_optional_torch_detachment_and_tail_gradient(self) -> None:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            self.skipTest(
                "PyTorch is not installed; dependency-free oracle remains active"
            )
        now = torch.tensor(
            [0.3, -0.3, 0.2, float("nan")], dtype=torch.float64, requires_grad=True
        )
        old = torch.tensor(
            [0.0, 0.0, 0.0, float("nan")], dtype=torch.float64, requires_grad=True
        )
        advantage = torch.tensor(
            [0.25, -0.75, 0.25, float("nan")], dtype=torch.float64, requires_grad=True
        )
        mask = torch.tensor([True, True, True, False])
        safe_now = torch.where(mask, now, 0.0)
        safe_old = torch.where(mask, old.detach(), 0.0)
        safe_advantage = torch.where(mask, advantage.detach(), 0.0)
        ell = safe_now - safe_old
        inside = (ell >= -0.2) & (ell <= 0.2)
        safe_ell = torch.where(inside, ell, 0.0)
        held = torch.where(
            ell < -0.2,
            math.exp(-0.2),
            torch.where(ell > 0.2, math.exp(0.2), safe_ell.exp()),
        )
        score = (safe_advantage * held - 0.1 * ell.square()).sum() / mask.sum()
        score.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)
        expected = (-0.06 / 3, 0.06 / 3, (0.25 * math.exp(0.2) - 0.04) / 3, 0.0)
        for got, want in zip(now.grad.tolist(), expected, strict=True):
            self.assertAlmostEqual(got, want, places=11)


if __name__ == "__main__":
    unittest.main()
