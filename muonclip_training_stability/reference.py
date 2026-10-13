"""Dependency-free MuonClip semantic oracles.

The module models logical values, not an official K2 implementation.  It keeps
the forward observation, inherited optimizer candidate, post-clip endpoint,
same-state replay, and next-forward checkpoint separate.  Fractions are used
for the chapter fixture so its advertised values remain exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt, tanh
from typing import Sequence, TypedDict


Scalar = Fraction | int
Vector = tuple[Fraction, ...]


def _fraction(value: Scalar) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def vector(*values: Scalar) -> Vector:
    """Construct an exact vector."""

    return tuple(_fraction(value) for value in values)


def dot(left: Sequence[Scalar], right: Sequence[Scalar]) -> Fraction:
    """Return the dot product of two vectors."""
    if len(left) != len(right):
        raise ValueError("dot operands must have equal length")
    return sum(
        (_fraction(a) * _fraction(b) for a, b in zip(left, right, strict=True)),
        start=Fraction(0),
    )


def scale_vector(values: Sequence[Scalar], factor: Scalar) -> Vector:
    """Scale every vector value by the supplied factor."""
    multiplier = _fraction(factor)
    return tuple(_fraction(value) * multiplier for value in values)


def score_matrix(
    queries: Sequence[Sequence[Scalar]],
    keys: Sequence[Sequence[Scalar]],
    scale: Scalar,
) -> tuple[tuple[Fraction, ...], ...]:
    """Return scaled pre-softmax scores for all query/key pairs."""

    score_scale = _fraction(scale)
    return tuple(
        tuple(score_scale * dot(query, key) for key in keys) for query in queries
    )


def scaled_positive_max(
    scores: Sequence[Sequence[Scalar]],
    admissible: Sequence[Sequence[bool]] | None = None,
) -> Fraction:
    """Return the largest admissible score, without taking absolute values."""

    if admissible is not None and len(admissible) != len(scores):
        raise ValueError("admissible mask must match the score matrix")
    values: list[Fraction] = []
    for row_index, row in enumerate(scores):
        if admissible is not None and len(row) != len(admissible[row_index]):
            raise ValueError("admissible mask must match the score matrix")
        for column_index, value in enumerate(row):
            if admissible is None or admissible[row_index][column_index]:
                values.append(_fraction(value))
    if not values:
        raise ValueError("the admissible score domain must be non-empty")
    return max(values)


def absolute_extreme(scores: Sequence[Sequence[Scalar]]) -> Fraction:
    """Return a separately named absolute-extreme diagnostic."""

    values = [abs(_fraction(value)) for row in scores for value in row]
    if not values:
        raise ValueError("the score domain must be non-empty")
    return max(values)


def large_logit_ratio(
    scores: Sequence[Sequence[Scalar]],
    threshold: Scalar,
    admissible: Sequence[Sequence[bool]] | None = None,
) -> Fraction:
    """Return the fraction of admissible scores strictly above ``threshold``."""

    if admissible is not None and len(admissible) != len(scores):
        raise ValueError("admissible mask must match the score matrix")
    flat: list[Fraction] = []
    for row_index, row in enumerate(scores):
        if admissible is not None and len(row) != len(admissible[row_index]):
            raise ValueError("admissible mask must match the score matrix")
        for column_index, value in enumerate(row):
            if admissible is None or admissible[row_index][column_index]:
                flat.append(_fraction(value))
    if not flat:
        raise ValueError("the score domain must be non-empty")
    boundary = _fraction(threshold)
    return Fraction(sum(value > boundary for value in flat), len(flat))


def clip_factor(maximum: Scalar, threshold: Scalar) -> Fraction:
    """Return K2's piecewise suppressive factor without invalid division."""

    score = _fraction(maximum)
    boundary = _fraction(threshold)
    if boundary <= 0:
        raise ValueError("threshold must be positive")
    return boundary / score if score > boundary else Fraction(1)


def _sqrt_fraction(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("cannot take the rational square root of a negative value")
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        raise ValueError("fixture requires an exact rational square root")
    return Fraction(numerator, denominator)


def mha_allocation_factors(gamma: Scalar, alpha: Fraction) -> tuple[Fraction, Fraction]:
    """Return exact query/key factors for alpha in {0, 1/2, 1}."""

    suppressor = _fraction(gamma)
    if not 0 < suppressor <= 1:
        raise ValueError("gamma must lie in (0, 1]")
    if alpha == 0:
        return Fraction(1), suppressor
    if alpha == Fraction(1, 2):
        root = _sqrt_fraction(suppressor)
        return root, root
    if alpha == 1:
        return suppressor, Fraction(1)
    raise ValueError("the exact fixture supports alpha in {0, 1/2, 1}")


@dataclass(frozen=True)
class MLAHead:
    """Logical unshared K2 MLA components for one focal token pair."""

    q_content: Vector
    k_content: Vector
    q_rotary: Vector


RunningTrace = TypedDict(
    "RunningTrace",
    {
        "threshold": Fraction,
        "observed_scores": tuple[Fraction, ...],
        "factors": tuple[Fraction, ...],
        "same_state_replay": tuple[Fraction, ...],
        "candidate_scores": tuple[Fraction, ...],
        "next_forward_scores": tuple[Fraction, ...],
        "next_factors": tuple[Fraction, ...],
        "forward_shared_key": Vector,
        "candidate_shared_key": Vector,
        "committed_heads": tuple[MLAHead, ...],
    },
)
LagCounterexample = TypedDict(
    "LagCounterexample",
    {
        "threshold": Fraction,
        "observed": Fraction,
        "factor": Fraction,
        "candidate": Fraction,
        "committed": Fraction,
    },
)
ConsensusRecord = TypedDict(
    "ConsensusRecord",
    {
        "local_maxima": tuple[tuple[Fraction, ...], ...],
        "consensus_maxima": tuple[Fraction, ...],
        "common_factors": tuple[Fraction, ...],
        "local_factors": tuple[tuple[Fraction, ...], ...],
    },
)
TriggerStatistics = TypedDict(
    "TriggerStatistics",
    {
        "instantaneous_rates": tuple[Fraction, ...],
        "per_head_counts": tuple[int, ...],
        "cumulative_ever_triggered": Fraction,
        "inactive_steps": tuple[bool, ...],
    },
)


def mla_score(
    head: MLAHead,
    shared_k_rotary: Sequence[Scalar],
    score_scale: Scalar = Fraction(1, 2),
) -> Fraction:
    """Return the content-plus-rotary score for one logical head."""

    return _fraction(score_scale) * (
        dot(head.q_content, head.k_content) + dot(head.q_rotary, shared_k_rotary)
    )


def mla_clip(head: MLAHead, gamma: Scalar) -> MLAHead:
    """Apply K2's head-local MLA factors; the shared rotary key is external."""

    suppressor = _fraction(gamma)
    root = _sqrt_fraction(suppressor)
    return MLAHead(
        q_content=scale_vector(head.q_content, root),
        k_content=scale_vector(head.k_content, root),
        q_rotary=scale_vector(head.q_rotary, suppressor),
    )


def running_trace() -> RunningTrace:
    """Execute one clipping step and its next observation on two heads."""

    threshold = Fraction(100)
    forward_shared_key = vector(40, 0)
    observed_heads = (
        MLAHead(vector(10, 0), vector(40, 0), vector(10, 0)),
        MLAHead(vector(2, 0), vector(40, 0), vector(2, 0)),
    )
    observed = tuple(mla_score(head, forward_shared_key) for head in observed_heads)
    factors = tuple(clip_factor(score, threshold) for score in observed)

    same_state_heads = tuple(
        mla_clip(head, factor)
        for head, factor in zip(observed_heads, factors, strict=True)
    )
    same_state_replay = tuple(
        mla_score(head, forward_shared_key) for head in same_state_heads
    )

    # These candidate components stand in for Chapter 12's inherited Muon
    # transition.  They are frozen fixture values, not an optimizer model.
    candidate_shared_key = vector(32, 0)
    candidate_heads = (
        MLAHead(vector(12, 0), vector(32, 0), vector(12, 0)),
        MLAHead(vector(Fraction(9, 4), 0), vector(32, 0), vector(Fraction(9, 4), 0)),
    )
    candidate_scores = tuple(
        mla_score(head, candidate_shared_key) for head in candidate_heads
    )
    committed_heads = tuple(
        mla_clip(head, factor)
        for head, factor in zip(candidate_heads, factors, strict=True)
    )
    next_forward = tuple(
        mla_score(head, candidate_shared_key) for head in committed_heads
    )
    next_factors = tuple(clip_factor(score, threshold) for score in next_forward)

    return {
        "threshold": threshold,
        "observed_scores": observed,
        "factors": factors,
        "same_state_replay": same_state_replay,
        "candidate_scores": candidate_scores,
        "next_forward_scores": next_forward,
        "next_factors": next_factors,
        "forward_shared_key": forward_shared_key,
        "candidate_shared_key": candidate_shared_key,
        "committed_heads": committed_heads,
    }


def lag_counterexample() -> LagCounterexample:
    """Show why a stored forward factor is not a post-Muon hard bound."""

    threshold = Fraction(100)
    scale = Fraction(1, 2)
    observed = scale * Fraction(20) * Fraction(40)
    factor = clip_factor(observed, threshold)
    candidate = scale * Fraction(80) * Fraction(40)
    committed = factor * candidate
    return {
        "threshold": threshold,
        "observed": observed,
        "factor": factor,
        "candidate": candidate,
        "committed": committed,
    }


def distributed_consensus(
    rank_local_maxima: Sequence[Sequence[Scalar]], threshold: Scalar
) -> ConsensusRecord:
    """Compute logical per-head max consensus and common clip factors."""

    if not rank_local_maxima:
        raise ValueError("at least one rank is required")
    head_count = len(rank_local_maxima[0])
    if head_count == 0 or any(len(row) != head_count for row in rank_local_maxima):
        raise ValueError("all ranks must report the same non-zero head count")
    local = tuple(tuple(_fraction(value) for value in row) for row in rank_local_maxima)
    consensus = tuple(max(row[head] for row in local) for head in range(head_count))
    common_factors = tuple(clip_factor(value, threshold) for value in consensus)
    local_factors = tuple(
        tuple(clip_factor(value, threshold) for value in row) for row in local
    )
    return {
        "local_maxima": local,
        "consensus_maxima": consensus,
        "common_factors": common_factors,
        "local_factors": local_factors,
    }


def trigger_statistics(
    trigger_history: Sequence[Sequence[bool]],
) -> TriggerStatistics:
    """Return instantaneous rates, cumulative ever-triggered fraction, and counts."""

    if not trigger_history:
        raise ValueError("at least one step is required")
    head_count = len(trigger_history[0])
    if head_count == 0 or any(len(row) != head_count for row in trigger_history):
        raise ValueError("all steps must contain the same non-zero head count")
    rates = tuple(Fraction(sum(row), head_count) for row in trigger_history)
    counts = tuple(
        sum(bool(row[head]) for row in trigger_history) for head in range(head_count)
    )
    ever = Fraction(sum(count > 0 for count in counts), head_count)
    inactive_steps = tuple(not any(row) for row in trigger_history)
    return {
        "instantaneous_rates": rates,
        "per_head_counts": counts,
        "cumulative_ever_triggered": ever,
        "inactive_steps": inactive_steps,
    }


def soft_cap(value: float, cap: float) -> float:
    """Apply Gemma 2-style c*tanh(z/c) to an exposed score."""

    if cap <= 0:
        raise ValueError("cap must be positive")
    return cap * tanh(value / cap)
