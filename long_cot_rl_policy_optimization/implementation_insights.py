"""Chapter 19 row-partition and local-gradient teaching checks.

These are chapter-created oracles, not Slime, veRL, or Kimi trainer code.
Prompt groups and logical rollouts have separate identities. A fragment owns
scored positions of one rollout, possibly no positions; it is not a new
sampled attempt. Policy witnesses keep every original history fixed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math


RolloutKey = tuple[str, str]


@dataclass(frozen=True)
class Fragment:
    """One physical row with scored positions from one logical rollout."""

    prompt_id: str
    rollout_id: str
    rollout_length: int
    positions: tuple[int, ...]
    outcome: int | float | Fraction


def _reward(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError("outcome must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("outcome must be finite")
    return Fraction(str(value))


def _record_fragment(
    row: Fragment,
    logical: dict[RolloutKey, tuple[int, Fraction]],
    positions: dict[RolloutKey, set[int]],
) -> None:
    if not row.prompt_id or not row.rollout_id:
        raise ValueError("prompt and rollout identities are required")
    length = row.rollout_length
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("logical rollout length must be positive")
    key = (row.prompt_id, row.rollout_id)
    identity = (length, _reward(row.outcome))
    if key in logical and logical[key] != identity:
        raise ValueError("one rollout has conflicting length or outcome")
    logical[key] = identity
    for position in row.positions:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < length
        ):
            raise ValueError("scored position lies outside its rollout")
        if position in positions[key]:
            raise ValueError("scored position is duplicated")
        positions[key].add(position)


def _partitions(rows: Sequence[Fragment]) -> dict[RolloutKey, tuple[int, Fraction]]:
    """Reject conflicting records and missing or repeated scored positions."""

    if not rows:
        raise ValueError("at least one training row is required")
    logical: dict[RolloutKey, tuple[int, Fraction]] = {}
    positions: dict[RolloutKey, set[int]] = defaultdict(set)
    for row in rows:
        _record_fragment(row, logical, positions)
    for key, (length, _) in logical.items():
        if positions[key] != set(range(length)):
            raise ValueError("scored position is missing")
    return logical


def distinct_rollout_advantages(
    rows: Sequence[Fragment], expected_group_sizes: Mapping[str, int]
) -> tuple[dict[str, Fraction], dict[RolloutKey, Fraction]]:
    """Center one outcome per logical rollout within each prompt group."""

    logical = _partitions(rows)
    groups: dict[str, list[RolloutKey]] = defaultdict(list)
    for key in logical:
        groups[key[0]].append(key)
    if set(groups) != set(expected_group_sizes):
        raise ValueError("prompt groups do not match the declared batch")
    means: dict[str, Fraction] = {}
    advantages: dict[RolloutKey, Fraction] = {}
    for prompt, keys in groups.items():
        promised = expected_group_sizes[prompt]
        if isinstance(promised, bool) or not isinstance(promised, int) or promised < 2:
            raise ValueError("each prompt needs a promised group size of at least two")
        if len(keys) != promised:
            raise ValueError("a promised rollout is missing or extra")
        mean = sum((logical[key][1] for key in keys), Fraction(0)) / promised
        means[prompt] = mean
        for key in keys:
            advantages[key] = logical[key][1] - mean
    return means, advantages


def whole_rollout_denominators(rows: Sequence[Fragment]) -> tuple[int, ...]:
    """Carry a selected mask total on each row of its logical rollout."""

    logical = _partitions(rows)
    return tuple(logical[(row.prompt_id, row.rollout_id)][0] for row in rows)


def reduce_fragments(
    rows: Sequence[Fragment], terms: Sequence[Sequence[float]], mode: str
) -> float:
    """Reduce one prompt's row terms by tokens, rollouts, or wrong row means."""

    logical = _partitions(rows)
    if len({key[0] for key in logical}) != 1:
        raise ValueError("a reducer call must select one prompt group")
    if len(terms) != len(rows):
        raise ValueError("one term vector is required per row")
    totals: list[float] = []
    for row, values in zip(rows, terms, strict=True):
        if len(values) != len(row.positions):
            raise ValueError("term count and scored-position count differ")
        values_f = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in values_f):
            raise ValueError("scored terms must be finite")
        totals.append(sum(values_f))
    if mode == "token":
        return sum(totals) / sum(length for length, _ in logical.values())
    if mode == "trajectory":
        return sum(
            total / logical[(row.prompt_id, row.rollout_id)][0]
            for row, total in zip(rows, totals, strict=True)
        ) / len(logical)
    if mode == "fragment":
        if any(not row.positions for row in rows):
            raise ValueError("a fragment mean needs a nonempty row")
        return sum(
            total / len(row.positions) for row, total in zip(rows, totals, strict=True)
        ) / len(rows)
    raise ValueError("unknown reduction mode")


def local_score_terms(
    current_logp: Sequence[float],
    behavior_logp: Sequence[float],
    advantages: Sequence[float],
    base_mask: Sequence[bool],
    lower: float = -0.2,
    upper: float = 0.2,
    regularizer: float = 0.1,
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    """Return retained forward scores and log-ratio derivatives.

    Masked positions are skipped before arithmetic; nonfinite padding cannot
    contaminate the score. Exterior reward factors are boundary constants,
    while the restoring derivative stays active at retained positions.
    """

    count = len(base_mask)
    if not (len(current_logp) == len(behavior_logp) == len(advantages) == count):
        raise ValueError("local-score arrays must have equal lengths")
    lower, upper, regularizer = float(lower), float(upper), float(regularizer)
    if (
        not all(math.isfinite(value) for value in (lower, upper, regularizer))
        or lower >= upper
        or regularizer < 0
    ):
        raise ValueError("invalid local-score configuration")
    if any(type(active) is not bool for active in base_mask):
        raise ValueError("base mask must be Boolean")
    retained = sum(base_mask)
    if retained == 0:
        raise ValueError("no retained action-loss positions")
    scores: list[float] = []
    derivatives: list[float] = []
    for now, old, advantage, active in zip(
        current_logp, behavior_logp, advantages, base_mask, strict=True
    ):
        if not active:
            scores.append(0.0)
            derivatives.append(0.0)
            continue
        score, derivative = _retained_local_score(
            now, old, advantage, lower, upper, regularizer
        )
        scores.append(score)
        derivatives.append(derivative)
    return tuple(scores), tuple(derivatives), retained


def _retained_local_score(
    now: float,
    old: float,
    advantage: float,
    lower: float,
    upper: float,
    regularizer: float,
) -> tuple[float, float]:
    now_f, old_f, advantage_f = float(now), float(old), float(advantage)
    if not all(math.isfinite(value) for value in (now_f, old_f, advantage_f)):
        raise ValueError("retained local-score inputs must be finite")
    ell = now_f - old_f
    if not math.isfinite(ell):
        raise ValueError("retained log ratio must be finite")
    eligible = lower <= ell <= upper
    try:
        held = math.exp(min(max(ell, lower), upper))
        reward_derivative = advantage_f * math.exp(ell) if eligible else 0.0
    except OverflowError as error:
        raise ValueError("local reward factor overflowed") from error
    score = advantage_f * held - regularizer * ell * ell
    derivative = reward_derivative - 2 * regularizer * ell
    if not math.isfinite(score) or not math.isfinite(derivative):
        raise ValueError("local score or derivative is nonfinite")
    return score, derivative


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def _log_sigmoid(value: float) -> float:
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def normalized_policy_witness(
    partitions: Sequence[Sequence[int]],
    mode: str = "token",
    theta: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, tuple[float, float]]:
    """Evaluate the eight-choice logistic policy and its parameter gradient."""

    if len(theta) != 2 or not all(math.isfinite(value) for value in theta):
        raise ValueError("the diagnostic policy needs two finite parameters")
    rows = [Fragment("diagnostic", "y1", 8, tuple(part), 1) for part in partitions]
    _partitions(rows)
    all_scores: list[list[float]] = []
    all_dx: list[list[float]] = []
    all_dy: list[list[float]] = []
    for row in rows:
        row_scores: list[float] = []
        row_dx: list[float] = []
        row_dy: list[float] = []
        for position in row.positions:
            features = (6.0, 0.0) if position < 2 else (-2.0, 2.0)
            logit = sum(x * y for x, y in zip(theta, features, strict=True))
            current_logp = _log_sigmoid(logit)
            score, coefficient, _ = local_score_terms(
                (current_logp,), (-math.log(2),), (0.25,), (True,)
            )
            score_gradient = 1 - _sigmoid(logit)
            row_scores.append(score[0])
            row_dx.append(coefficient[0] * score_gradient * features[0])
            row_dy.append(coefficient[0] * score_gradient * features[1])
        all_scores.append(row_scores)
        all_dx.append(row_dx)
        all_dy.append(row_dy)
    return reduce_fragments(rows, all_scores, mode), (
        reduce_fragments(rows, all_dx, mode),
        reduce_fragments(rows, all_dy, mode),
    )
