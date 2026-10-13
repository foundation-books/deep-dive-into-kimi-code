"""Executable negative witnesses for the six Chapter 34 trace faults."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from . import reference as ref


FAULT_IDS = (
    "foreign_owner",
    "layout_misinterpretation",
    "premature_depth_commit",
    "missing_partial_tail",
    "missing_readiness_event",
    "rejected_append_commit",
)


def _require_equal(boundary: str, expected: object, observed: object) -> None:
    if observed != expected:
        raise ValueError(f"{boundary}: observed value differs from the reference")


def _foreign_owner_fault() -> None:
    expected = ref.OwnerTag(
        "A", "prefix:5", 4, None, "model-r1", "token-major", "bf16", "accepted"
    )
    foreign = replace(expected, request="B", endpoint_or_forward="prefix:3")
    ref.validate_owner(expected, foreign)


def _layout_misinterpretation_fault() -> None:
    logical = ((ref.f(1), ref.f(2)), (ref.f(3), ref.f(5)))
    misread = ref.logical_to_feature_major(logical)
    _require_equal("layout bijection", logical, misread)


def _premature_depth_commit_fault() -> None:
    _, partial = ref.block_index_partition(2, 4)
    if partial:
        raise ValueError("group boundary: an incomplete depth partial cannot commit")


def _missing_partial_tail_fault() -> None:
    completed, _ = ref.block_index_partition(10, 4)
    observed = tuple(index for group in completed for index in group)
    _require_equal("source conservation", tuple(range(1, 11)), observed)


def _missing_readiness_event_fault() -> None:
    producer_revision = "side-summary-r2"
    visible_revision = "side-summary-r1"
    event_observed = False
    if not event_observed or visible_revision != producer_revision:
        raise ValueError(
            "readiness event: the merge can observe a stale summary revision"
        )


def _rejected_append_commit_fault() -> None:
    base = ref.make_bundle("A", 5)
    candidate = ref.stage_continuation(base, token=6)
    corrupted = replace(
        base,
        gmla_records=(base.gmla_records[0], candidate.gmla_records[1]),
    )
    _require_equal("hybrid atomicity", base, corrupted)


def _capture_rejection(
    expected_fragment: str, operation: Callable[[], None]
) -> dict[str, object]:
    observed = "no rejection"
    try:
        operation()
    except ValueError as error:
        observed = str(error)
    return ref.check_record(
        expected_fragment in observed,
        expected_rejection=expected_fragment,
        observed=observed,
    )


def fault_witnesses() -> dict[str, dict[str, object]]:
    """Run all six negative controls and record their earliest rejection."""

    cases = (
        (FAULT_IDS[0], "owner-tag mismatch", _foreign_owner_fault),
        (FAULT_IDS[1], "layout bijection", _layout_misinterpretation_fault),
        (FAULT_IDS[2], "group boundary", _premature_depth_commit_fault),
        (FAULT_IDS[3], "source conservation", _missing_partial_tail_fault),
        (FAULT_IDS[4], "readiness event", _missing_readiness_event_fault),
        (FAULT_IDS[5], "hybrid atomicity", _rejected_append_commit_fault),
    )
    return {
        fault_id: _capture_rejection(expected_fragment, operation)
        for fault_id, expected_fragment, operation in cases
    }
