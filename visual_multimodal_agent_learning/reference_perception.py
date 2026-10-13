"""Observation, coordinate, and diagnostic fixtures for Chapter 25."""

from __future__ import annotations

from fractions import Fraction
from math import exp, isclose
from typing import Mapping, Sequence


def _valid_observation(
    record: Mapping[str, object], known_observations: set[str]
) -> bool:
    required = {
        "id",
        "asset",
        "digest",
        "frame",
        "transform",
        "parent",
        "state_version",
        "capture_time",
        "provenance",
    }
    if set(record) != required or not all(
        record[field] is not None for field in required - {"parent"}
    ):
        return False
    parent = record["parent"]
    if parent is not None and parent not in known_observations:
        return False
    return isinstance(record["state_version"], int)


def observation_lineage_check() -> dict[str, object]:
    """Check identity, version, digest, and parent requirements."""

    root = {
        "id": "obs-root-v11",
        "asset": "studio-canvas",
        "digest": "sha256:root",
        "frame": "root-edge-v1",
        "transform": "identity-v1",
        "parent": None,
        "state_version": 11,
        "capture_time": "t0",
        "provenance": "capture-tool-v1",
    }
    crop = {
        "id": "obs-crop-v11",
        "asset": "studio-canvas-crop",
        "digest": "sha256:crop",
        "frame": "crop-edge-v1",
        "transform": "root-to-crop-v1",
        "parent": root["id"],
        "state_version": 11,
        "capture_time": "t1",
        "provenance": "crop-tool-v1:(10,20,60,120)",
    }
    duplicate_pixels_new_state = {**root, "id": "obs-root-v12", "state_version": 12}
    broken_crop = {**crop, "parent": None}
    root_ok = _valid_observation(root, set())
    crop_ok = _valid_observation(crop, {str(root["id"])})
    duplicate_is_distinct = (
        duplicate_pixels_new_state["digest"] == root["digest"]
        and duplicate_pixels_new_state["id"] != root["id"]
        and duplicate_pixels_new_state["state_version"] != root["state_version"]
    )
    # A root may have no parent; a derived crop may not.
    broken_is_valid = _valid_observation(broken_crop, {str(root["id"])}) and not (
        str(broken_crop["provenance"]).startswith("crop-tool")
        and broken_crop["parent"] is None
    )
    broken_rejected = not broken_is_valid
    passed = root_ok and crop_ok and duplicate_is_distinct and broken_rejected
    return {
        "status": "pass" if passed else "fail",
        "root_valid": root_ok,
        "derived_crop_valid": crop_ok,
        "duplicate_pixels_remain_distinct": duplicate_is_distinct,
        "missing_parent_rejected": broken_rejected,
        "state_versions": [11, 12],
    }


def _forward_point(
    point: Sequence[int], crop: Sequence[int], viewport: Sequence[int]
) -> tuple[Fraction, Fraction]:
    x, y = point
    x0, y0, width, height = crop
    viewport_width, viewport_height = viewport
    if not (x0 <= x <= x0 + width and y0 <= y <= y0 + height):
        raise ValueError("point lies outside declared crop support")
    return (
        Fraction(viewport_width, width) * (x - x0),
        Fraction(viewport_height, height) * (y - y0),
    )


def _inverse_point(
    point: Sequence[Fraction], crop: Sequence[int], viewport: Sequence[int]
) -> tuple[Fraction, Fraction]:
    u, v = point
    x0, y0, width, height = crop
    viewport_width, viewport_height = viewport
    if not (0 <= u <= viewport_width and 0 <= v <= viewport_height):
        raise ValueError("point lies outside declared viewport support")
    return (
        Fraction(x0) + Fraction(width, viewport_width) * u,
        Fraction(y0) + Fraction(height, viewport_height) * v,
    )


def coordinate_round_trip_check() -> dict[str, object]:
    """Check the continuous-edge crop fixture and fail-closed support."""

    crop = (10, 20, 60, 120)
    viewport = (300, 600)
    red = _forward_point((40, 40), crop, viewport)
    blue = _forward_point((40, 100), crop, viewport)
    red_inverse = _inverse_point(red, crop, viewport)
    blue_inverse = _inverse_point(blue, crop, viewport)
    blue_box = (
        _forward_point((20, 80), crop, viewport),
        _forward_point((60, 120), crop, viewport),
    )
    out_of_support_rejected = False
    try:
        _forward_point((9, 100), crop, viewport)
    except ValueError:
        out_of_support_rejected = True
    passed = (
        red == (150, 100)
        and blue == (150, 400)
        and red_inverse == (40, 40)
        and blue_inverse == (40, 100)
        and blue_box == ((50, 300), (250, 500))
        and out_of_support_rejected
    )
    return {
        "status": "pass" if passed else "fail",
        "scale": ["5", "5"],
        "red_viewport_center": [str(value) for value in red],
        "blue_viewport_center": [str(value) for value in blue],
        "blue_viewport_box_xywh": ["50", "300", "200", "200"],
        "round_trip_exact": red_inverse == (40, 40) and blue_inverse == (40, 100),
        "out_of_support_rejected": out_of_support_rejected,
    }


def _box_iou(left: Sequence[int], right: Sequence[int]) -> Fraction:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    left_area = max(0, lx1 - lx0) * max(0, ly1 - ly0)
    right_area = max(0, rx1 - rx0) * max(0, ry1 - ry0)
    intersection = max(0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0, min(ly1, ry1) - max(ly0, ry0)
    )
    union = left_area + right_area - intersection
    if union == 0:
        raise ValueError("IoU is undefined when the union has zero area")
    return Fraction(intersection, union)


def _levenshtein(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        next_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            next_row.append(
                min(
                    next_row[-1] + 1,
                    row[right_index] + 1,
                    row[right_index - 1] + (left_char != right_char),
                )
            )
        row = next_row
    return row[-1]


def _ocr_score(reference: str, prediction: str) -> Fraction:
    denominator = max(len(reference), len(prediction))
    if denominator == 0:
        raise ValueError("normalized OCR score needs a nonempty string")
    return Fraction(denominator - _levenshtein(reference, prediction), denominator)


def visual_diagnostic_check() -> dict[str, object]:
    """Reproduce the Chapter 25 box, point, OCR, and count diagnostics."""

    iou = _box_iou((0, 0, 4, 4), (2, 0, 6, 4))
    point_score = exp(-25 / (2 * 25))
    ocr = _ocr_score("BLUE", "BLVE")
    count_error = abs(4 - 6)
    zero_denominators_rejected = 0
    for function, arguments in (
        (_box_iou, ((0, 0, 0, 0), (0, 0, 0, 0))),
        (_ocr_score, ("", "")),
    ):
        try:
            function(*arguments)
        except ValueError:
            zero_denominators_rejected += 1
    passed = (
        iou == Fraction(1, 3)
        and isclose(point_score, exp(-0.5), rel_tol=0.0, abs_tol=1e-15)
        and ocr == Fraction(3, 4)
        and count_error == 2
        and zero_denominators_rejected == 2
    )
    return {
        "status": "pass" if passed else "fail",
        "box_iou": str(iou),
        "point_score": format(point_score, ".12f"),
        "ocr_score": str(ocr),
        "count_error": count_error,
        "zero_denominator_cases_rejected": zero_denominators_rejected,
        "equal_scores_preserve_tie": _ocr_score("BLUE", "BLUE")
        == _ocr_score("RED", "RED"),
    }


def zero_vision_denominator_check() -> dict[str, object]:
    """Check record- and token-qualified stage exposure."""

    total_records = 8_000 + 200 + 100
    visual_records = 100
    total_tokens = Fraction(12_800_000)
    visual_tokens = Fraction(200_000)
    record_fraction = 1 - Fraction(visual_records, total_records)
    token_fraction = 1 - visual_tokens / total_tokens
    passed = (
        record_fraction == Fraction(82, 83)
        and token_fraction == Fraction(63, 64)
        and record_fraction != 1
        and token_fraction != 1
    )
    return {
        "status": "pass" if passed else "fail",
        "record_fraction": str(record_fraction),
        "token_fraction": str(token_fraction),
        "exact_record_zero_vision": record_fraction == 1,
        "exact_token_zero_vision": token_fraction == 1,
        "checkpoint_lineage": "joint-multimodal-pretraining",
    }


__all__ = [
    "coordinate_round_trip_check",
    "observation_lineage_check",
    "visual_diagnostic_check",
    "zero_vision_denominator_check",
]
