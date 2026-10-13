"""Single-fault mutants for Chapter 35 oracle-sensitivity checks."""

from __future__ import annotations

from dataclasses import replace

from . import reference as ref

FAULT_IDS = (
    "wrong_branch_width",
    "biased_execution_weight",
    "duplicate_identity",
    "bad_final_offset",
    "mismatched_permutation",
    "visible_padding",
    "wrong_expert_parameter",
    "wrong_origin",
    "dropped_token_route",
    "changed_replay_record",
)


def _bad_final_offset_is_rejected(
    records: tuple[ref.RouteRecord, ...],
    loads: tuple[int, ...],
    offsets: tuple[int, ...],
) -> bool:
    bad_offsets = (*offsets[:-1], offsets[-1] - 1)
    try:
        ref.validate_partition(records, 4, loads, bad_offsets)
    except ValueError:
        return True
    return False


def _padding_leak_changes_result(loads: tuple[int, ...], logical_rows: int) -> bool:
    masks = ref.padding_mask(loads, 4)
    lane_values = tuple(
        ref.f(1) if valid else ref.f(99_999) for mask in masks for valid in mask
    )
    flat_mask = tuple(valid for mask in masks for valid in mask)
    masked = sum(
        (value for value, valid in zip(lane_values, flat_mask, strict=True) if valid),
        ref.f(0),
    )
    return masked == logical_rows and sum(lane_values, ref.f(0)) != masked


def fault_witnesses() -> dict[str, bool]:
    """Return True exactly when each injected fault is detected."""
    records = ref.toy_routes()
    detected: dict[str, bool] = {}

    try:
        ref.validate_branch_widths((0,) * 4, (0,) * 4)
    except ValueError:
        detected[FAULT_IDS[0]] = True

    raw = (ref.f(1), ref.f(2), ref.f(3))
    bias = (ref.f(10), ref.f(0), ref.f(0))
    selected = ref.selected_weights(raw, bias, 2)
    biased_denominator = sum(
        (raw[index - 1] + bias[index - 1] for index, _ in selected), ref.f(0)
    )
    detected[FAULT_IDS[1]] = any(
        weight != (raw[index - 1] + bias[index - 1]) / biased_denominator
        for index, weight in selected
    )

    duplicate = list(records)
    duplicate[1] = duplicate[0]
    try:
        ref.validate_records(duplicate, 2)
    except ValueError:
        detected[FAULT_IDS[2]] = True

    loads, offsets = ref.loads_and_offsets(records, 4)
    detected[FAULT_IDS[3]] = _bad_final_offset_is_rejected(records, loads, offsets)

    token_rows = {
        origin: (ref.f(index),) for index, origin in enumerate(ref.ORIGINS, 1)
    }
    dispatch = ref.stable_dispatch(records, ref.repeated_rows(token_rows, records), 4)
    wrong_rows = tuple(reversed(dispatch.rows))
    detected[FAULT_IDS[4]] = any(
        row != token_rows[record.origin]
        for record, row in zip(dispatch.records, wrong_rows, strict=True)
    )

    detected[FAULT_IDS[5]] = _padding_leak_changes_result(loads, len(records))

    independent, grouped = ref.grouped_situ_fixture()
    wrong_grouped = list(grouped)
    wrong_grouped[0] = tuple(-value for value in wrong_grouped[0])
    detected[FAULT_IDS[6]] = tuple(wrong_grouped) != independent

    outputs = ref.diagnostic_outputs(records)
    correct = ref.combine(records, outputs)
    wrong_records = list(records)
    wrong_records[0] = ref.replace_origin(wrong_records[0], "tau", 1)
    wrong = ref.combine(wrong_records, outputs)
    detected[FAULT_IDS[7]] = wrong != correct

    try:
        ref.token_centric_diagnostic(records[:-1])
    except ValueError:
        detected[FAULT_IDS[8]] = True

    changed = list(dispatch.records)
    changed[0] = replace(changed[0], expert=4)
    detected[FAULT_IDS[9]] = tuple(changed) != dispatch.records
    return detected
