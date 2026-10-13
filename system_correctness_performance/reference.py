"""Finite arithmetic and accounting oracles for Chapter 45."""

from fractions import Fraction
from itertools import pairwise
from math import ceil, isfinite
from typing import Sequence


def tolerance_accepts(
    reference: Sequence[float], candidate: Sequence[float], atol: float, rtol: float
) -> bool:
    """Directed finite comparison; an empty pair passes only vacuously."""
    if not all(isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    if len(reference) != len(candidate):
        return False
    return all(
        isfinite(x)
        and isfinite(y)
        and abs(Fraction(y) - Fraction(x))
        <= Fraction(atol) + Fraction(rtol) * abs(Fraction(x))
        for x, y in zip(reference, candidate, strict=True)
    )


def error_bound(
    gains: Sequence[float], defects: Sequence[float], initial: float = 0
) -> float:
    """Evaluate the one-step norm inequality, not a model recurrence."""
    if len(gains) != len(defects):
        raise ValueError("one defect per gain")
    if not all(isfinite(v) and v >= 0 for v in (*gains, *defects, initial)):
        raise ValueError("finite nonnegative norm bounds required")
    bound = initial
    for gain, defect in zip(gains, defects, strict=True):
        bound = gain * bound + defect
    return bound


def routed_outputs(
    xs: Sequence[float], routes: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    """Dense independent sum for experts 2x and 3x with fixed route weights."""
    if len(xs) != len(routes) or any(len(row) != 2 for row in routes):
        raise ValueError("two route weights per token")
    if not all(isfinite(v) for v in xs):
        raise ValueError("finite input required")
    if any(
        not all(isfinite(w) and w >= 0 for w in row) or abs(sum(row) - 1) > 1e-12
        for row in routes
    ):
        raise ValueError("normalized finite route weights required")
    return tuple(
        sum(w * factor * x for w, factor in zip(row, (2, 3), strict=True))
        for x, row in zip(xs, routes, strict=True)
    )


def routed_loss(xs: Sequence[float], routes: Sequence[Sequence[float]]) -> float:
    """Half squared output norm on the fixed smooth route branch."""
    return sum(y * y for y in routed_outputs(xs, routes)) / 2


def directional_difference(
    xs: Sequence[float],
    direction: Sequence[float],
    routes: Sequence[Sequence[float]],
    step: float,
) -> float:
    """Centered finite difference of the fixed-route quadratic loss."""
    if len(xs) != len(direction) or not isfinite(step) or step <= 0:
        raise ValueError("matched direction and positive finite step required")
    plus = [x + step * v for x, v in zip(xs, direction, strict=True)]
    minus = [x - step * v for x, v in zip(xs, direction, strict=True)]
    return (routed_loss(plus, routes) - routed_loss(minus, routes)) / (2 * step)


def probability_vector(values: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """Validate exact nonnegative probability mass on a finite vocabulary."""
    result = tuple(values)
    if not result or any(v < 0 for v in result) or sum(result) != 1:
        raise ValueError("nonempty exact normalized law required")
    return result


def total_variation(p: Sequence[Fraction], q: Sequence[Fraction]) -> Fraction:
    """Half the exact L1 discrepancy for matched finite laws."""
    p, q = probability_vector(p), probability_vector(q)
    if len(p) != len(q):
        raise ValueError("matched vocabulary required")
    return sum((abs(a - b) for a, b in zip(p, q, strict=True)), Fraction()) / 2


def emitted_law(p: Sequence[Fraction], q: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """Enumerate accepted and rejection/correction branches exactly."""
    total_variation(p, q)
    accepted = tuple(min(a, b) for a, b in zip(p, q, strict=True))
    rejection = Fraction(1) - sum(accepted, Fraction())
    if rejection == 0:
        return accepted
    residual = tuple(
        max(a - b, Fraction()) / rejection for a, b in zip(p, q, strict=True)
    )
    return tuple(a + rejection * r for a, r in zip(accepted, residual, strict=True))


def latency(
    arrival: float, tokens: Sequence[float], terminal: float
) -> dict[str, object]:
    """Observer-clock latency; absent token or gap populations return None."""
    timeline = (arrival, *tokens, terminal)
    if not all(isfinite(t) for t in timeline) or any(
        a > b for a, b in pairwise(timeline)
    ):
        raise ValueError("finite ordered observer timestamps required")
    gaps = tuple(b - a for a, b in pairwise(tokens))
    return {
        "ttft": tokens[0] - arrival if tokens else None,
        "gaps": gaps,
        "mean_gap": sum(gaps) / len(gaps) if gaps else None,
        "e2e": terminal - arrival,
    }


def quantile(values: Sequence[float], p: float) -> float | None:
    """Empirical inverse-CDF convention, with None for an empty population."""
    if not 0 < p <= 1 or not all(isfinite(v) for v in values):
        raise ValueError("finite observations and probability in (0,1] required")
    return sorted(values)[ceil(p * len(values)) - 1] if values else None


def cohort_metrics(
    records: Sequence[tuple[str, str, bool]], seconds: float
) -> dict[str, float | None]:
    """One cutoff outcome per logical ID; qualified implies terminal success."""
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("positive observation duration required")
    ids = [record[0] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate logical request outcome")
    allowed = {"success", "rejected", "failed", "cancelled", "unfinished"}
    if any(
        outcome not in allowed or (good and outcome != "success")
        for _, outcome, good in records
    ):
        raise ValueError("invalid outcome or qualification")
    successes = sum(outcome == "success" for _, outcome, _ in records)
    good = sum(qualified for _, _, qualified in records)
    return {
        "success_rate": successes / seconds,
        "goodput": good / seconds,
        "offered_attainment": good / len(records) if records else None,
        "conditional_attainment": good / successes if successes else None,
    }


def peak_live(intervals: Sequence[tuple[float, float, int]]) -> int:
    """Half-open lifetimes for distinct objects; no allocator inference."""
    if any(
        not isfinite(a) or not isfinite(b) or a > b or size < 0
        for a, b, size in intervals
    ):
        raise ValueError("ordered finite lifetimes and nonnegative bytes required")
    times = {a for a, _, _ in intervals}
    return max(
        (sum(size for a, b, size in intervals if a <= t < b) for t in times), default=0
    )


def resource_bounds(
    flops: float, byte_count: float, peak: float, bandwidth: float
) -> tuple[float, float, float]:
    """Return intensity, Roofline rate ceiling and resource time floor."""
    if not all(isfinite(v) and v > 0 for v in (flops, byte_count, peak, bandwidth)):
        raise ValueError("positive finite resource inputs required")
    intensity = flops / byte_count
    return (
        intensity,
        min(peak, bandwidth * intensity),
        max(flops / peak, byte_count / bandwidth),
    )


def recovery_start(
    samples: Sequence[tuple[float, bool]], consecutive: int
) -> float | None:
    """First complete consecutive-sample window; samples are prescheduled."""
    if consecutive <= 0:
        raise ValueError("positive window sample count required")
    if any(not isfinite(t) for t, _ in samples) or any(
        a[0] >= b[0] for a, b in pairwise(samples)
    ):
        raise ValueError("strictly increasing finite sample times required")
    for start in range(len(samples) - consecutive + 1):
        if all(ok for _, ok in samples[start : start + consecutive]):
            return samples[start][0]
    return None
