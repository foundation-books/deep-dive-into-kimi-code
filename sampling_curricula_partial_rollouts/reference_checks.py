"""Deterministic protocol records for the Chapter 20 reference module."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

from .reference_core import (
    LEGAL_TRANSITIONS,
    STUDIO_CONSUMER_VERSION,
    STUDIO_FAILURES,
    STUDIO_GENERATION_VERSIONS,
    STUDIO_SUCCESSES,
    PauseCheckpoint,
    ReuseVector,
    Segment,
    beta_smoothed_difficulty,
    corrected_expectation,
    group_status,
    raw_difficulty,
    studio_trace,
    validate_checkpoint,
    validate_segments,
    validate_transition,
    validate_weak_fair_schedule,
)


def _sampling_check(trace: dict[str, object]) -> dict[str, object]:
    q = tuple(Fraction(value) for value in cast(list[str], trace["acquisition"]))
    p = (Fraction(1, 3),) * 3
    w = tuple(Fraction(value) for value in cast(list[str], trace["weights"]))
    if sum(q) != 1 or any(qi * wi != pi for qi, wi, pi in zip(q, w, p, strict=True)):
        raise AssertionError("D01 change-of-measure check failed")
    if corrected_expectation(p, q, (1, 0, -1)) != 0:
        raise AssertionError("D01 expectation identity failed")
    return {
        "protocol": "CH18-D01",
        "status": "pass",
        "acquisition": trace["acquisition"],
        "weights": trace["weights"],
        "corrected_signed_mean": "0",
    }


def _difficulty_check() -> dict[str, object]:
    raw = tuple(
        raw_difficulty(successes, failures)
        for successes, failures in zip(STUDIO_SUCCESSES, STUDIO_FAILURES, strict=True)
    )
    smooth = tuple(
        beta_smoothed_difficulty(successes, failures, 1, 1)
        for successes, failures in zip(STUDIO_SUCCESSES, STUDIO_FAILURES, strict=True)
    )
    if raw != (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)):
        raise AssertionError("D02 raw difficulty failed")
    if smooth != (Fraction(1, 6), Fraction(1, 2), Fraction(5, 6)):
        raise AssertionError("D02 smoothed difficulty failed")
    return {
        "protocol": "CH18-D02",
        "status": "pass",
        "raw": [str(value) for value in raw],
        "smoothed": [str(value) for value in smooth],
    }


def _lifecycle_check() -> dict[str, object]:
    checkpoint = PauseCheckpoint(
        "studio-layout-003",
        "studio-group-001",
        4,
        1,
        STUDIO_GENERATION_VERSIONS[0][0],
        "teaching-sampler-v1",
        tuple(f"t{index}" for index in range(1, 11)),
        "boxes-observed",
        "fresh-suffix-under-v121",
        "tokenizer-sha",
        "schema-sha",
        "content-sha",
    )
    validate_checkpoint(checkpoint)
    for edge in LEGAL_TRANSITIONS:
        validate_transition(*edge)
    return {
        "protocol": "CH18-D03",
        "status": "pass",
        "legal_transition_count": len(LEGAL_TRANSITIONS),
        "checkpoint_valid": True,
    }


def _reuse_check() -> dict[str, object]:
    prefix = ReuseVector(1, 0, 0, 0)
    suffix = ReuseVector(0, 0, 1, 1)
    prefix.validate()
    suffix.validate()
    segment_record = validate_segments(
        20,
        (Segment(1, 10, 120), Segment(11, 20, 121)),
        STUDIO_CONSUMER_VERSION,
    )
    return {
        "protocol": "CH18-D04",
        "status": "pass",
        "prefix_reuse": list(prefix.__dict__.values()),
        "suffix_reuse": list(suffix.__dict__.values()),
        "segment_lags": list(cast(tuple[int, ...], segment_record["lags"])),
    }


def _admission_check() -> dict[str, object]:
    ready = group_status(("terminal",) * 4, (1, 1, 1, 1))
    settled_not_ready = group_status(
        ("terminal", "terminal", "terminal", "retired"), (1, 1, 1, 0)
    )
    validate_weak_fair_schedule(("slot-4", "slot-2"), {"slot-2", "slot-4"})
    if not ready["ready"] or not settled_not_ready["settled"]:
        raise AssertionError("D05 group status failed")
    if settled_not_ready["ready"]:
        raise AssertionError("D05 settled/ready distinction failed")
    return {
        "protocol": "CH18-D05",
        "status": "pass",
        "ready_group": ready,
        "settled_not_ready_group": settled_not_ready,
    }


def _studio_trace_check(trace: dict[str, object]) -> dict[str, object]:
    if (
        trace["total_tokens"] != 48
        or trace["pause_threshold"] != 3
        or trace["slot_4_lags"] != [3, 2]
        or not trace["ready"]
    ):
        raise AssertionError("D06 studio trace failed")
    return {"protocol": "CH18-D06", "status": "pass", "trace": trace}


def all_checks() -> dict[str, dict[str, object]]:
    """Return the six deterministic chapter records or raise on failure."""

    trace = studio_trace()
    return {
        "RESULT_CH18_SAMPLING_ORACLE": _sampling_check(trace),
        "RESULT_CH18_DIFFICULTY_ORACLE": _difficulty_check(),
        "RESULT_CH18_LIFECYCLE_ORACLE": _lifecycle_check(),
        "RESULT_CH18_REUSE_ORACLE": _reuse_check(),
        "RESULT_CH18_ADMISSION_ORACLE": _admission_check(),
        "RESULT_CH18_STUDIO_TRACE_ORACLE": _studio_trace_check(trace),
    }
