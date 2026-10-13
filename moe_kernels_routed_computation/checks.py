"""Ten deterministic evidence checks for Chapter 35."""

from __future__ import annotations

from itertools import pairwise

from . import faults, reference as ref


def _typed_branch_domains() -> dict[str, object]:
    accepted = True
    ref.validate_branch_widths((0,) * 8, (0,) * 4)
    joined = ref.stable_latent_join(
        (3.0, 4.0, 0.0, 0.0),
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        (1.0, 1.0, 1.0, 1.0),
        0.0,
        ((1.0,) * 8,),
    )
    rejected = False
    try:
        ref.validate_branch_widths((0,) * 4, (0,) * 4)
    except ValueError:
        rejected = True
    return ref.check_record(
        accepted and rejected and joined == (2.2, 2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        model_width=8,
        latent_width=4,
        joined=list(joined),
    )


def _selected_score_weights() -> dict[str, object]:
    raw = (ref.f(1), ref.f(2), ref.f(3), ref.f(4))
    correction = (ref.f(10), ref.f(0), ref.f(0), ref.f(0))
    selected = ref.selected_weights(raw, correction, 2)
    passed = selected == ((1, ref.f(1, 5)), (4, ref.f(4, 5)))
    return ref.check_record(
        passed,
        experts=[expert for expert, _ in selected],
        weights=[str(weight) for _, weight in selected],
    )


def _route_cardinality() -> dict[str, object]:
    records = ref.toy_routes()
    return ref.check_record(
        len(records) == 20 and len({record.identity for record in records}) == 20,
        logical_routes=len(records),
        origins=len({record.origin for record in records}),
    )


def _count_scan_coverage() -> dict[str, object]:
    loads, offsets = ref.loads_and_offsets(ref.toy_routes(), 4)
    ref.validate_partition(ref.toy_routes(), 4, loads, offsets)
    intervals = tuple(pairwise(offsets))
    covered = tuple(index for lo, hi in intervals for index in range(lo, hi))
    return ref.check_record(
        loads == ref.EXPECTED_LOADS
        and offsets == ref.EXPECTED_OFFSETS
        and covered == tuple(range(20)),
        loads=list(loads),
        offsets=list(offsets),
    )


def _permutation_round_trip() -> dict[str, object]:
    records = ref.toy_routes()
    token_rows = {
        origin: (ref.f(index), ref.f(-index))
        for index, origin in enumerate(ref.ORIGINS, 1)
    }
    canonical = ref.repeated_rows(token_rows, records)
    dispatch = ref.stable_dispatch(records, canonical, 4)
    restored = ref.restore_logical(dispatch.rows, dispatch)
    return ref.check_record(
        restored == canonical
        and tuple(dispatch.inverse[index] for index in dispatch.permutation)
        == tuple(range(20)),
        permutation=list(dispatch.permutation),
    )


def _padding_invisibility() -> dict[str, object]:
    masks = ref.padding_mask(ref.EXPECTED_LOADS, 4)
    logical = sum(sum(mask) for mask in masks)
    issued = sum(len(mask) for mask in masks)
    lane_values = tuple(
        ref.f(1) if valid else ref.f(99_999) for mask in masks for valid in mask
    )
    flat_mask = tuple(valid for mask in masks for valid in mask)
    masked_sum = sum(
        (value for value, valid in zip(lane_values, flat_mask, strict=True) if valid),
        ref.f(0),
    )
    return ref.check_record(
        logical == 20 and issued == 24 and masked_sum == 20,
        logical_rows=logical,
        issued_rows=issued,
        inflation="6/5",
    )


def _grouped_expert_parity() -> dict[str, object]:
    independent, grouped = ref.grouped_situ_fixture()
    max_error = max(
        abs(left - right)
        for expected, actual in zip(independent, grouped, strict=True)
        for left, right in zip(expected, actual, strict=True)
    )
    return ref.check_record(
        independent == grouped and len(grouped) == 3,
        logical_rows=3,
        max_abs_error=max_error,
    )


def _restore_checksum() -> dict[str, object]:
    restored, checksum = ref.diagnostic_trace()
    expected_tau3 = ref.f(702_654_629, 1_000_000)
    return ref.check_record(
        checksum == ref.EXPECTED_CHECKSUM and restored[("tau", 3)][0] == expected_tau3,
        checksum=str(checksum),
        tau3=str(restored[("tau", 3)][0]),
    )


def _schedule_equivalence() -> dict[str, object]:
    records = ref.toy_routes()
    token = ref.token_centric_diagnostic(records)
    expert, _ = ref.diagnostic_trace()
    return ref.check_record(token == expert, origins=len(token), exact=True)


def _backward_replay() -> dict[str, object]:
    records = ref.toy_routes()
    token_rows = {
        origin: (ref.f(index),) for index, origin in enumerate(ref.ORIGINS, 1)
    }
    dispatch = ref.stable_dispatch(records, ref.repeated_rows(token_rows, records), 4)
    replay = ref.replay_dispatch(dispatch)
    gradients = ref.scalar_backward(
        ref.f(2), (ref.f(1, 3), ref.f(2, 3)), (ref.f(3), ref.f(-1))
    )
    expected_dz = ref.f(2, 9)
    expected_dp = (ref.f(4), ref.f(-4, 3))
    expected_dw = (ref.f(4, 9), ref.f(8, 9))
    expected = {
        "output": ref.f(2, 3),
        "dz": expected_dz,
        "dp": expected_dp,
        "dw": expected_dw,
    }
    variables = (ref.f(2), ref.f(1, 3), ref.f(2, 3), ref.f(3), ref.f(-1))
    step = ref.f(1, 100)

    def loss(values: tuple[ref.Scalar, ...]) -> ref.Scalar:
        z, p1, p2, w1, w2 = values
        output = p1 * w1 * z + p2 * w2 * z
        return output * output / 2

    finite_differences = []
    for index in range(len(variables)):
        upper = list(variables)
        lower = list(variables)
        upper[index] += step
        lower[index] -= step
        finite_differences.append(
            (loss(tuple(upper)) - loss(tuple(lower))) / (2 * step)
        )
    expected_derivatives = (expected_dz, *expected_dp, *expected_dw)
    witnesses = faults.fault_witnesses()
    return ref.check_record(
        replay == dispatch
        and gradients == expected
        and tuple(finite_differences) == expected_derivatives
        and all(witnesses.values()),
        gradients={key: str(value) for key, value in gradients.items()},
        finite_differences=[str(value) for value in finite_differences],
        fault_mutants=list(witnesses),
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Return exactly ten outcomes in stable order."""
    return {
        "typed_branch_domains": _typed_branch_domains(),
        "selected_score_weights": _selected_score_weights(),
        "route_cardinality": _route_cardinality(),
        "count_scan_coverage": _count_scan_coverage(),
        "permutation_round_trip": _permutation_round_trip(),
        "padding_invisibility": _padding_invisibility(),
        "grouped_expert_parity": _grouped_expert_parity(),
        "restore_checksum": _restore_checksum(),
        "schedule_equivalence": _schedule_equivalence(),
        "backward_replay": _backward_replay(),
    }
