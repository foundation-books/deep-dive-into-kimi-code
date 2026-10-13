"""Five-record studio fixture and negative checks for Chapter 16."""

from __future__ import annotations

from fractions import Fraction
from .reference_serialization import (
    canonical_checksum,
    causal_contract_valid,
    crop_box,
    media_identity_swap_detected,
    normalize_box,
    reflect_box_horizontally,
    serialized_payload,
    serialized_record,
    validate_serialized_record,
)
from .reference_studio_inputs import studio_record_inputs
from .reference_types import Box, SerializedRecord


def _input_contract() -> dict[str, object]:
    return {
        "semantic_artifact": "red circle above blue square",
        "records": studio_record_inputs(),
        "geometry_box": [100, 40, 180, 120],
        "reflection_frame": [400, 200],
        "retained_crop": [240, 0, 400, 200],
        "removed_evidence_crop": [0, 0, 80, 200],
        "negative_cases": [
            "visual_vocabulary_target",
            "active_null_target",
            "missing_visual_coordinate",
            "swapped_media_owner",
            "stale_reflected_coordinate",
            "answer_visible_in_prefix",
            "evidence_removing_crop",
        ],
    }


def _serialize_specification(
    specification: dict[str, object],
) -> tuple[str, SerializedRecord, dict[str, object]]:
    name = str(specification["name"])
    visual_positions = _integer_field(specification, "visual_positions")
    prompt_tokens = _integer_field(specification, "prompt_tokens")
    target_tokens = _integer_field(specification, "target_tokens")
    media_value = specification["media_id"]
    media_id = None if media_value is None else str(media_value)
    record = serialized_record(
        visual_positions=visual_positions,
        prompt_tokens=prompt_tokens,
        target_tokens=target_tokens,
        media_id=media_id,
        target_prefix=str(specification["target_prefix"]),
    )
    validate_serialized_record(record)
    causal_prefix_valid = causal_contract_valid(
        first_target_position=visual_positions + prompt_tokens,
        required_evidence_end=visual_positions,
        external_bos=bool(specification["external_bos"]),
    )
    if not causal_prefix_valid:
        raise AssertionError(f"{name}: canonical target order is noncausal")
    payload = {
        "family": specification["family"],
        "assets": specification["assets"],
        "provenance": {
            "source_id": specification["source_id"],
            "media_id": media_id,
            "transformations": specification["transformations"],
        },
        "realized_grid": specification["realized_grid"],
        "coordinate_frame": specification["coordinate_frame"],
        "source_coordinate_frame": specification.get("source_coordinate_frame"),
        "target_dependency": specification["target_dependency"],
        "causal_prefix_valid": causal_prefix_valid,
        "counts": {
            "length": len(record.modality),
            "visual_positions": record.modality.count("visual"),
            "prompt_tokens": prompt_tokens,
            "target_positions": sum(record.loss_mask),
        },
        "serialization": serialized_payload(record),
    }
    return name, record, payload


def _integer_field(specification: dict[str, object], name: str) -> int:
    value = specification[name]
    if not isinstance(value, int):
        raise AssertionError(f"studio record field {name} must be an integer")
    return value


def _serialize_records(
    specifications: object,
) -> tuple[dict[str, dict[str, object]], dict[str, SerializedRecord]]:
    if not isinstance(specifications, list):
        raise AssertionError("studio record inputs must be a list")
    records: dict[str, dict[str, object]] = {}
    serialized: dict[str, SerializedRecord] = {}
    for specification in specifications:
        if not isinstance(specification, dict):
            raise AssertionError("studio record input must be an object")
        name, record, payload = _serialize_specification(specification)
        records[name] = payload
        serialized[name] = record
    return records, serialized


def _record_rejected(record: SerializedRecord) -> bool:
    try:
        validate_serialized_record(record)
    except ValueError:
        return True
    return False


def _negative_record_checks(record: SerializedRecord) -> dict[str, bool]:
    visual_target = SerializedRecord(
        modality=record.modality,
        media_id=record.media_id,
        coordinate_state=record.coordinate_state,
        target_token=tuple(
            f"invalid-visual-{index}" if index < 4 else token
            for index, token in enumerate(record.target_token)
        ),
        loss_mask=tuple(
            1 if index < 4 else value for index, value in enumerate(record.loss_mask)
        ),
    )
    null_targets = list(record.target_token)
    null_targets[-1] = None
    active_null = SerializedRecord(
        modality=record.modality,
        media_id=record.media_id,
        coordinate_state=record.coordinate_state,
        target_token=tuple(null_targets),
        loss_mask=record.loss_mask,
    )
    missing_coordinates = list(record.coordinate_state)
    missing_coordinates[0] = None
    missing_coordinate = SerializedRecord(
        modality=record.modality,
        media_id=record.media_id,
        coordinate_state=tuple(missing_coordinates),
        target_token=record.target_token,
        loss_mask=record.loss_mask,
    )
    return {
        "visual_targets_rejected": _record_rejected(visual_target),
        "active_null_target_rejected": _record_rejected(active_null),
        "missing_visual_coordinate_rejected": _record_rejected(missing_coordinate),
        "media_identity_swap_detected": media_identity_swap_detected(),
        "leaked_answer_rejected": not causal_contract_valid(
            first_target_position=48,
            required_evidence_end=48,
            external_bos=False,
            answer_visible_in_prefix=True,
        ),
    }


def _geometry_checks() -> dict[str, object]:
    box: Box = (Fraction(100), Fraction(40), Fraction(180), Fraction(120))
    reflected = reflect_box_horizontally(box, 400, 200)
    crop: Box = (Fraction(240), Fraction(0), Fraction(400), Fraction(200))
    removed_crop: Box = (Fraction(0), Fraction(0), Fraction(80), Fraction(200))
    clipped, clip_valid = crop_box(reflected, crop, 400, 200, policy="clip_and_retain")
    rejected, reject_valid = crop_box(
        reflected, crop, 400, 200, policy="reject_partial_object"
    )
    removed, removed_valid = crop_box(
        reflected,
        removed_crop,
        400,
        200,
        policy="clip_and_retain",
    )
    if clipped is None:
        raise AssertionError("the canonical reflected box must intersect the crop")
    crop_width = int(crop[2] - crop[0])
    crop_height = int(crop[3] - crop[1])
    return {
        "stale_reflected_coordinates_detected": box != reflected,
        "normalized_box": [str(value) for value in normalize_box(box, 400, 200)],
        "reflected_box": [str(value) for value in reflected],
        "clipped_crop_local_box": [str(value) for value in clipped],
        "clipped_crop_local_normalized": [
            str(value) for value in normalize_box(clipped, crop_width, crop_height)
        ],
        "clip_and_retain_valid": clip_valid,
        "reject_partial_object_valid": reject_valid,
        "reject_partial_object_box": rejected,
        "evidence_removing_crop_rejected": removed is None and not removed_valid,
    }


def _media_ids(records: dict[str, SerializedRecord]) -> list[str]:
    return sorted(
        {
            item
            for record in records.values()
            for item in record.media_id
            if item is not None
        }
    )


def serialization_fixture() -> dict[str, object]:
    """Return all five records, geometry, provenance, and negative checks."""

    input_contract = _input_contract()
    records, serialized_records = _serialize_records(input_contract["records"])
    caption = serialized_records["e_C"]
    output_payload = {
        "records": records,
        "record_names": list(records),
        "record_length": len(caption.modality),
        "visual_positions": caption.modality.count("visual"),
        "prompt_positions": 12,
        "target_positions": sum(caption.loss_mask),
        "media_ids": _media_ids(serialized_records),
        **_negative_record_checks(caption),
        **_geometry_checks(),
    }
    return {
        "input_contract": input_contract,
        "input_checksum": canonical_checksum(input_contract),
        **output_payload,
        "output_checksum": canonical_checksum(output_payload),
    }


__all__ = ["serialization_fixture"]
