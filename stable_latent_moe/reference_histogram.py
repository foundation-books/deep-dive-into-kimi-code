"""Histogram approximation helpers for Stable LatentMoE Quantile Balancing."""

from __future__ import annotations

from typing import Sequence

from .reference_core import _require_width
from .reference_fixture import QB_SCORES, RUNNING_BIASES
from .reference_types import HistogramResult, RUNNING_DIMS, ScoreMatrix, Vector


def token_cutoffs(
    scores: ScoreMatrix,
    biases: Sequence[float],
    selected_count: int,
    *,
    interior: bool = False,
) -> Vector:
    """Return token thresholds at the source endpoint or minimizer interior."""

    cutoffs: list[float] = []
    for row in scores:
        _require_width(row, len(biases), "score row")
        ordered = sorted(
            (score + bias for score, bias in zip(row, biases, strict=True)),
            reverse=True,
        )
        if not 1 <= selected_count < len(ordered):
            raise ValueError("cutoff requires 1 <= k < E")
        threshold = (
            (ordered[selected_count - 1] + ordered[selected_count]) / 2.0
            if interior
            else ordered[selected_count]
        )
        cutoffs.append(threshold)
    return tuple(cutoffs)


def required_bias_samples(
    scores: ScoreMatrix = QB_SCORES,
    biases: Sequence[float] = RUNNING_BIASES,
    selected_count: int = RUNNING_DIMS.selected_experts,
) -> tuple[Vector, ...]:
    """Return one required-bias sample vector per expert."""

    alpha = token_cutoffs(scores, biases, selected_count)
    return tuple(
        tuple(alpha[token] - scores[token][expert] for token in range(len(scores)))
        for expert in range(len(biases))
    )


def histogram_counts(
    samples: Sequence[Sequence[float]],
    lower: float,
    upper: float,
    bins: int,
) -> tuple[tuple[int, ...], ...]:
    """Histogram per-expert samples over one shared uniform range."""

    if bins <= 0 or upper <= lower:
        raise ValueError("histogram requires positive bins and range")
    width = (upper - lower) / bins
    result: list[tuple[int, ...]] = []
    for expert_samples in samples:
        counts = [0] * bins
        for value in expert_samples:
            if value < lower or value > upper:
                raise ValueError("required-bias sample is outside the declared range")
            index = min(int((value - lower) / width), bins - 1)
            counts[index] += 1
        result.append(tuple(counts))
    return tuple(result)


def add_histograms(
    histograms: Sequence[Sequence[Sequence[int]]],
) -> tuple[tuple[int, ...], ...]:
    """Add equally shaped nonnegative per-expert histograms."""

    if not histograms or not histograms[0] or not histograms[0][0]:
        raise ValueError("histograms must be nonempty")
    experts = len(histograms[0])
    bins = len(histograms[0][0])
    if any(
        len(histogram) != experts or any(len(row) != bins for row in histogram)
        for histogram in histograms
    ):
        raise ValueError("all histograms must have equal shape")
    return tuple(
        tuple(
            sum(histogram[expert][bucket] for histogram in histograms)
            for bucket in range(bins)
        )
        for expert in range(experts)
    )


def recover_histogram_quantile(
    counts: Sequence[Sequence[int]],
    lower: float,
    upper: float,
    target_rank: float,
) -> Vector:
    """Apply Appendix D's first-bin and within-bin interpolation rule."""

    if not counts or not counts[0] or upper <= lower or target_rank <= 0:
        raise ValueError("invalid histogram quantile inputs")
    bins = len(counts[0])
    width = (upper - lower) / bins
    recovered: list[float] = []
    for row in counts:
        if len(row) != bins or any(value < 0 for value in row):
            raise ValueError("histogram rows must be rectangular and nonnegative")
        if target_rank > sum(row):
            raise ValueError("target rank exceeds sample count")
        cumulative = 0
        for bucket, count in enumerate(row):
            if cumulative + count >= target_rank:
                fraction = (
                    0.0
                    if count == 0
                    else min(max((target_rank - cumulative) / count, 0.0), 1.0)
                )
                recovered.append(lower + (bucket + fraction) * width)
                break
            cumulative += count
    return tuple(recovered)


def histogram_quantile_estimate(
    scores: ScoreMatrix = QB_SCORES,
    biases: Sequence[float] = RUNNING_BIASES,
    selected_count: int = RUNNING_DIMS.selected_experts,
    bins: int = 32,
) -> HistogramResult:
    """Build and recover the pooled Appendix D histogram estimator."""

    samples = required_bias_samples(scores, biases, selected_count)
    lower = min(biases) - 1.0
    upper = max(biases) + 1.0
    counts = histogram_counts(samples, lower, upper, bins)
    target = len(scores) * selected_count / len(biases)
    recovered = recover_histogram_quantile(counts, lower, upper, target)
    return HistogramResult(lower, upper, (upper - lower) / bins, counts, recovered)


def exact_histogram_targets(
    scores: ScoreMatrix = QB_SCORES,
    biases: Sequence[float] = RUNNING_BIASES,
    selected_count: int = RUNNING_DIMS.selected_experts,
) -> Vector:
    """Return Appendix D's cumulative-rank exact required-bias targets."""

    samples = required_bias_samples(scores, biases, selected_count)
    target = int(len(scores) * selected_count / len(biases))
    return tuple(sorted(row)[target - 1] for row in samples)


def partitioned_histogram_estimate(
    partitions: Sequence[Sequence[int]],
    bins: int = 32,
) -> HistogramResult:
    """Histogram declared token partitions, add counts, and recover globally."""

    flattened = tuple(index for partition in partitions for index in partition)
    if sorted(flattened) != list(range(len(QB_SCORES))):
        raise ValueError("partitions must cover every token exactly once")
    all_samples = required_bias_samples()
    lower = min(RUNNING_BIASES) - 1.0
    upper = max(RUNNING_BIASES) + 1.0
    local_counts = [
        histogram_counts(
            tuple(
                tuple(expert_samples[token] for token in partition)
                for expert_samples in all_samples
            ),
            lower,
            upper,
            bins,
        )
        for partition in partitions
    ]
    counts = add_histograms(local_counts)
    target = len(QB_SCORES) * RUNNING_DIMS.selected_experts / len(RUNNING_BIASES)
    recovered = recover_histogram_quantile(counts, lower, upper, target)
    return HistogramResult(lower, upper, (upper - lower) / bins, counts, recovered)


__all__ = [
    "add_histograms",
    "exact_histogram_targets",
    "histogram_counts",
    "histogram_quantile_estimate",
    "partitioned_histogram_estimate",
    "recover_histogram_quantile",
    "required_bias_samples",
    "token_cutoffs",
]
