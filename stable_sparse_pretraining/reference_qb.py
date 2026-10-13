"""Quantile-Balancing causality fixture for the Chapter 14 companion."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def qb_select(
    raw_scores: Sequence[Fraction],
    current_bias: Sequence[Fraction],
    top_k: int = 2,
) -> tuple[tuple[int, ...], tuple[Fraction, ...]]:
    """Select with biased scores and weight selected routes with raw scores."""

    if len(raw_scores) != len(current_bias) or not 0 < top_k <= len(raw_scores):
        raise ValueError("scores, bias, and top_k are inconsistent")
    if any(not Fraction(0) < score < Fraction(1) for score in raw_scores):
        raise ValueError("raw sigmoid scores must lie strictly between zero and one")
    selected = tuple(
        sorted(
            range(len(raw_scores)),
            key=lambda expert: (raw_scores[expert] + current_bias[expert], -expert),
            reverse=True,
        )[:top_k]
    )
    denominator = sum((raw_scores[expert] for expert in selected), start=Fraction(0))
    if denominator <= 0:
        raise ValueError("selected raw-score sum must be positive")
    return selected, tuple(raw_scores[expert] / denominator for expert in selected)


def qb_fixture() -> dict[str, object]:
    """Return a causal current-bias/candidate-next-bias selection record."""

    raw = (Fraction(4, 5), Fraction(17, 25), Fraction(16, 25), Fraction(2, 25))
    current = (Fraction(-1, 2), Fraction(0), Fraction(3, 10), Fraction(1, 5))
    selected, weights = qb_select(raw, current)
    observed_counts = (8, 4, 2, 2)
    target = Fraction(sum(observed_counts), len(observed_counts))
    candidate_next = tuple(
        bias - Fraction(count - target, 16)
        for bias, count in zip(current, observed_counts, strict=True)
    )
    return {
        "raw_scores": raw,
        "current_bias": current,
        "selected": selected,
        "weights": weights,
        "candidate_next_bias": candidate_next,
        "current_bias_after_observation": current,
    }


__all__ = ("qb_fixture", "qb_select")
