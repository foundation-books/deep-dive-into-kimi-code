"""Dependency-free semantic oracles for Chapter 20.

The functions reproduce finite task-sampling arithmetic, difficulty
estimators, rollout lifecycle checks, reuse coordinates, group readiness, and
the mixed-version studio trace.  They are teaching references, not a Moonshot
AI sampler, rollout service, trainer, or performance reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Iterable, Sequence, cast


STUDIO_SUCCESSES = (9, 5, 1)
STUDIO_FAILURES = (1, 5, 9)
STUDIO_LENGTHS = (8, 12, 8, 20)
STUDIO_OUTCOMES = (1, 1, 0, 1)
STUDIO_GENERATION_VERSIONS = ((120,), (120,), (120,), (120, 121))
STUDIO_CONSUMER_VERSION = 123

LEGAL_TRANSITIONS = frozenset(
    {
        ("queued", "running"),
        ("running", "paused"),
        ("paused", "running"),
        ("running", "terminal"),
        ("running", "invalid"),
        ("invalid", "queued"),
        ("invalid", "retired"),
        ("paused", "retired"),
    }
)


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric observation")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return Fraction(str(value))


def normalize_priority(
    scores: Sequence[int | float | Fraction],
    *,
    fallback: Sequence[int | float | Fraction] | None = None,
    epsilon: int | float | Fraction = 0,
    support_reference: Sequence[int | float | Fraction] | None = None,
) -> tuple[Fraction, ...]:
    """Normalize nonnegative scores with explicit zero and support handling."""

    if not scores:
        raise ValueError("at least one score is required")
    values = tuple(_fraction(value) for value in scores)
    if any(value < 0 for value in values):
        raise ValueError("priority scores must be nonnegative")
    total = sum(values, Fraction(0))
    if total == 0:
        if fallback is None:
            raise ValueError("zero priority mass requires a fallback")
        base = normalize_priority(fallback)
    else:
        base = tuple(value / total for value in values)

    guard = _fraction(epsilon)
    if guard < 0 or guard > 1:
        raise ValueError("epsilon must lie in [0, 1]")
    if guard == 0:
        return base
    if support_reference is None:
        reference = (Fraction(1, len(base)),) * len(base)
    else:
        reference = normalize_priority(support_reference)
    if len(reference) != len(base):
        raise ValueError("support reference has the wrong dimension")
    mixed = tuple(
        (1 - guard) * q + guard * u for q, u in zip(base, reference, strict=True)
    )
    if sum(mixed, Fraction(0)) != 1:
        raise AssertionError("mixed acquisition distribution did not normalize")
    return mixed


def importance_weights(
    target: Sequence[int | float | Fraction],
    acquisition: Sequence[int | float | Fraction],
) -> tuple[Fraction, ...]:
    """Return p/q after validating probability mass and support."""

    p = normalize_priority(target)
    q = normalize_priority(acquisition)
    if len(p) != len(q):
        raise ValueError("target and acquisition dimensions differ")
    if any(pi > 0 and qi == 0 for pi, qi in zip(p, q, strict=True)):
        raise ValueError("acquisition lacks target support")
    return tuple(
        Fraction(0) if pi == 0 else pi / qi for pi, qi in zip(p, q, strict=True)
    )


def expectation(
    probabilities: Sequence[int | float | Fraction],
    values: Sequence[int | float | Fraction],
) -> Fraction:
    """Return an exact finite expectation."""

    p = normalize_priority(probabilities)
    f = tuple(_fraction(value) for value in values)
    if len(p) != len(f):
        raise ValueError("probability and value dimensions differ")
    return sum((pi * fi for pi, fi in zip(p, f, strict=True)), Fraction(0))


def corrected_expectation(
    target: Sequence[int | float | Fraction],
    acquisition: Sequence[int | float | Fraction],
    values: Sequence[int | float | Fraction],
) -> Fraction:
    """Evaluate the exact task-level change-of-measure identity."""

    q = normalize_priority(acquisition)
    weights = importance_weights(target, q)
    return expectation(
        q,
        tuple(w * _fraction(v) for w, v in zip(weights, values, strict=True)),
    )


def raw_difficulty(successes: int, failures: int) -> Fraction:
    """Return failures divided by valid attempts."""

    if (
        isinstance(successes, bool)
        or isinstance(failures, bool)
        or not isinstance(successes, int)
        or not isinstance(failures, int)
        or successes < 0
        or failures < 0
    ):
        raise ValueError("counts must be nonnegative integers")
    if successes + failures == 0:
        raise ValueError("raw difficulty needs a valid attempt")
    return Fraction(failures, successes + failures)


def beta_smoothed_difficulty(
    successes: int,
    failures: int,
    alpha: int | float | Fraction,
    beta: int | float | Fraction,
) -> Fraction:
    """Return the posterior mean failure probability."""

    if successes < 0 or failures < 0:
        raise ValueError("counts must be nonnegative")
    a = _fraction(alpha)
    b = _fraction(beta)
    if a <= 0 or b <= 0:
        raise ValueError("Beta parameters must be positive")
    return (b + failures) / (a + b + successes + failures)


def validate_transition(source: str, destination: str) -> bool:
    """Validate one edge of the Chapter 20 lifecycle."""

    if (source, destination) not in LEGAL_TRANSITIONS:
        raise ValueError(f"illegal lifecycle transition: {source}->{destination}")
    return True


@dataclass(frozen=True)
class PauseCheckpoint:
    """Minimum semantic fields for a resumable partial rollout."""

    task_id: str
    group_id: str
    slot_id: int
    attempt_id: int
    behavior_version: int
    sampler_contract: str
    prefix_tokens: tuple[str, ...]
    environment_state: str
    continuation_rule: str
    tokenizer_hash: str
    schema_hash: str
    content_hash: str


def validate_checkpoint(checkpoint: PauseCheckpoint) -> bool:
    """Reject missing identity, state, continuation, or integrity fields."""

    strings = (
        checkpoint.task_id,
        checkpoint.group_id,
        checkpoint.sampler_contract,
        checkpoint.environment_state,
        checkpoint.continuation_rule,
        checkpoint.tokenizer_hash,
        checkpoint.schema_hash,
        checkpoint.content_hash,
    )
    if not all(value.strip() for value in strings):
        raise ValueError("checkpoint has an empty required field")
    if checkpoint.slot_id < 1 or checkpoint.attempt_id < 1:
        raise ValueError("slot and attempt IDs must be positive")
    if checkpoint.behavior_version < 0:
        raise ValueError("behavior version must be nonnegative")
    if not checkpoint.prefix_tokens:
        raise ValueError("teaching checkpoint requires a nonempty prefix")
    return True


@dataclass(frozen=True)
class ReuseVector:
    """Continuation, data-selection, loss-inclusion, optimizer-use counts."""

    continuation: int
    data_selections: int
    loss_included: int
    optimizer_steps: int

    def validate(self) -> bool:
        values = (
            self.continuation,
            self.data_selections,
            self.loss_included,
            self.optimizer_steps,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("reuse coordinates must be integers")
        if self.continuation not in (0, 1) or self.loss_included not in (0, 1):
            raise ValueError("continuation and loss coordinates must be binary")
        if self.data_selections < 0 or self.optimizer_steps < 0:
            raise ValueError("reuse counts must be nonnegative")
        return True


def group_status(
    slot_dispositions: Sequence[str],
    accepted_representatives: Sequence[int],
) -> dict[str, bool]:
    """Return strict settled and ready predicates for fixed logical slots."""

    if not slot_dispositions or len(slot_dispositions) != len(accepted_representatives):
        raise ValueError("slot records need equal nonzero length")
    if any(count not in (0, 1) for count in accepted_representatives):
        raise ValueError("each slot may have at most one accepted representative")
    allowed = {"active", "paused", "terminal", "retired"}
    if any(state not in allowed for state in slot_dispositions):
        raise ValueError("unknown slot disposition")
    settled = all(
        (state == "terminal" and count == 1) or (state == "retired" and count == 0)
        for state, count in zip(
            slot_dispositions, accepted_representatives, strict=True
        )
    )
    ready = all(
        state == "terminal" and count == 1
        for state, count in zip(
            slot_dispositions, accepted_representatives, strict=True
        )
    )
    return {"settled": settled, "ready": ready}


def validate_weak_fair_schedule(
    schedule: Sequence[str], continuously_eligible: Iterable[str]
) -> bool:
    """Check a finite witness in which every declared eligible slot is served."""

    eligible = set(continuously_eligible)
    if not eligible:
        raise ValueError("fairness witness needs at least one eligible slot")
    if not eligible.issubset(schedule):
        raise ValueError("a continuously eligible slot was starved")
    return True


@dataclass(frozen=True)
class Segment:
    """One gap-free token interval and its generating policy version."""

    start: int
    end: int
    generation_version: int


def validate_segments(
    length: int, segments: Sequence[Segment], consumer_version: int
) -> dict[str, object]:
    """Validate a partition and return chronological version lags."""

    if length <= 0 or not segments:
        raise ValueError("positive length and at least one segment are required")
    expected = 1
    lags: list[int] = []
    for segment in segments:
        if segment.start != expected or segment.end < segment.start:
            raise ValueError("segments are not ordered, adjacent, and gap-free")
        if segment.generation_version > consumer_version:
            raise ValueError("consumer predates a generation segment")
        lags.append(consumer_version - segment.generation_version)
        expected = segment.end + 1
    if expected != length + 1:
        raise ValueError("segments do not cover the trajectory")
    return {"lags": tuple(lags), "maximum_lag": max(lags)}


def studio_trace() -> dict[str, object]:
    """Reproduce the Chapter 20 teaching trace and its conservation values."""

    difficulty = tuple(
        raw_difficulty(s, f)
        for s, f in zip(STUDIO_SUCCESSES, STUDIO_FAILURES, strict=True)
    )
    acquisition = normalize_priority(difficulty)
    target = (Fraction(1, 3),) * 3
    weights = importance_weights(target, acquisition)
    slot_4_versions = STUDIO_GENERATION_VERSIONS[3]
    segments = validate_segments(
        20,
        (
            Segment(1, 10, slot_4_versions[0]),
            Segment(11, 20, slot_4_versions[1]),
        ),
        STUDIO_CONSUMER_VERSION,
    )
    status = group_status(("terminal",) * 4, (1, 1, 1, 1))
    payload = {
        "difficulty": [str(value) for value in difficulty],
        "target": [str(value) for value in target],
        "acquisition": [str(value) for value in acquisition],
        "weights": [str(value) for value in weights],
        "selected_task": "studio-layout-003",
        "selected_task_probability": str(acquisition[2]),
        "slots": 4,
        "pause_threshold": 3,
        "outcomes": list(STUDIO_OUTCOMES),
        "lengths": list(STUDIO_LENGTHS),
        "total_tokens": sum(STUDIO_LENGTHS),
        "slot_4_segments": [[1, 10, 120], [11, 20, 121]],
        "slot_4_lags": list(cast(tuple[int, ...], segments["lags"])),
        "maximum_lag": segments["maximum_lag"],
        "settled": status["settled"],
        "ready": status["ready"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload
