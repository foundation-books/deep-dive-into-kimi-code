"""Typed record and geometry semantics for the Chapter 16 companion."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from .reference_types import Box, SerializedRecord


def canonical_checksum(payload: object) -> str:
    """Hash a JSON-compatible fixture payload with deterministic encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def serialized_record(
    *,
    visual_positions: int,
    prompt_tokens: int,
    target_tokens: int,
    media_id: str | None,
    target_prefix: str,
) -> SerializedRecord:
    """Construct one exact run-length studio serialization."""

    if min(visual_positions, prompt_tokens, target_tokens) < 0:
        raise ValueError("record span counts must be nonnegative")
    if target_tokens == 0 or visual_positions + prompt_tokens + target_tokens == 0:
        raise ValueError("a studio training record requires active targets")
    if visual_positions and media_id is None:
        raise ValueError("visual spans require a media identity")
    if not visual_positions and media_id is not None:
        raise ValueError("text-only records cannot carry a visual media identity")
    visual_coordinates = tuple(
        f"merged-grid:{index // 4}:{index % 4}" for index in range(visual_positions)
    )
    return SerializedRecord(
        modality=("visual",) * visual_positions
        + ("text",) * (prompt_tokens + target_tokens),
        media_id=(media_id,) * visual_positions
        + (None,) * (prompt_tokens + target_tokens),
        coordinate_state=visual_coordinates + (None,) * (prompt_tokens + target_tokens),
        target_token=(None,) * (visual_positions + prompt_tokens)
        + tuple(f"{target_prefix}-{index}" for index in range(target_tokens)),
        loss_mask=(0,) * (visual_positions + prompt_tokens) + (1,) * target_tokens,
    )


def caption_record(
    visual_positions: int = 16,
    prompt_tokens: int = 12,
    caption_tokens: int = 20,
) -> SerializedRecord:
    """Construct the canonical studio caption record."""

    if min(visual_positions, prompt_tokens, caption_tokens) <= 0:
        raise ValueError("all three record spans must be positive")
    return serialized_record(
        visual_positions=visual_positions,
        prompt_tokens=prompt_tokens,
        target_tokens=caption_tokens,
        media_id="studio-image-1",
        target_prefix="caption",
    )


def _validate_visual_position(
    media_id: str | None, coordinate: str | None, target: str | None, active: int
) -> None:
    if media_id is None:
        raise ValueError("every visual position requires one media identity")
    if coordinate is None:
        raise ValueError("every visual position requires coordinate state")
    if active:
        raise ValueError("active NTP targets must be vocabulary-bearing text positions")
    if target is not None:
        raise ValueError("inactive positions cannot carry fixture targets")


def _validate_text_position(
    media_id: str | None, coordinate: str | None, target: str | None, active: int
) -> None:
    if media_id is not None:
        raise ValueError("the fixture assigns media identities only to visual input")
    if coordinate is not None:
        raise ValueError("text positions cannot carry fixture visual coordinates")
    if active and target is None:
        raise ValueError("active NTP targets must be vocabulary-bearing text positions")
    if not active and target is not None:
        raise ValueError("inactive positions cannot carry fixture targets")


def _validate_position(
    modality: str,
    media_id: str | None,
    coordinate: str | None,
    target: str | None,
    active: int,
) -> None:
    if modality == "visual":
        _validate_visual_position(media_id, coordinate, target, active)
    elif modality == "text":
        _validate_text_position(media_id, coordinate, target, active)
    else:
        raise ValueError("unknown modality")


def validate_serialized_record(record: SerializedRecord) -> None:
    """Check typing, ownership, target-domain, and target-count invariants."""

    lengths = {
        len(record.modality),
        len(record.media_id),
        len(record.coordinate_state),
        len(record.target_token),
        len(record.loss_mask),
    }
    if len(lengths) != 1 or not record.modality:
        raise ValueError("serialized fields must have one nonzero length")
    if any(value not in (0, 1) for value in record.loss_mask):
        raise ValueError("loss masks are binary")
    for position in zip(
        record.modality,
        record.media_id,
        record.coordinate_state,
        record.target_token,
        record.loss_mask,
        strict=True,
    ):
        _validate_position(*position)
    if sum(record.loss_mask) == 0:
        raise ValueError("a training record requires at least one active target")


def normalize_box(box: Box, width: int, height: int) -> Box:
    """Normalize a half-open continuous pixel box to the unit square."""

    _validate_box(box, width, height)
    x0, y0, x1, y1 = box
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def reflect_box_horizontally(box: Box, width: int, height: int) -> Box:
    """Reflect a half-open continuous box around the image's vertical axis."""

    _validate_box(box, width, height)
    x0, y0, x1, y1 = box
    return (Fraction(width) - x1, y0, Fraction(width) - x0, y1)


def crop_box(
    box: Box,
    crop: Box,
    width: int,
    height: int,
    *,
    policy: str,
) -> tuple[Box | None, bool]:
    """Map a box into crop-local coordinates under a named partial-box policy."""

    _validate_box(box, width, height)
    _validate_box(crop, width, height)
    bx0, by0, bx1, by1 = box
    cx0, cy0, cx1, cy1 = crop
    ix0, iy0 = max(bx0, cx0), max(by0, cy0)
    ix1, iy1 = min(bx1, cx1), min(by1, cy1)
    if ix0 >= ix1 or iy0 >= iy1:
        return None, False
    partial = (ix0, iy0, ix1, iy1) != box
    if policy == "reject_partial_object" and partial:
        return None, False
    if policy not in {"clip_and_retain", "reject_partial_object"}:
        raise ValueError("unknown crop policy")
    return (ix0 - cx0, iy0 - cy0, ix1 - cx0, iy1 - cy0), True


def _validate_box(box: Box, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("box must be ordered and inside the declared frame")


def media_identity_swap_detected() -> bool:
    """Detect a shape-preserving owner swap across two identical patch grids."""

    positions = tuple(
        (owner, row, column)
        for owner in ("studio-A", "studio-B")
        for row in range(4)
        for column in range(4)
    )
    corrupted = list(positions)
    owner_a, row_a, col_a = corrupted[0]
    owner_b, row_b, col_b = corrupted[16]
    corrupted[0] = (owner_b, row_a, col_a)
    corrupted[16] = (owner_a, row_b, col_b)
    expected_owner = tuple(("studio-A",) * 16 + ("studio-B",) * 16)
    return any(
        observed[0] != expected
        for observed, expected in zip(corrupted, expected_owner, strict=True)
    )


def causal_contract_valid(
    *,
    first_target_position: int,
    required_evidence_end: int,
    external_bos: bool,
    answer_visible_in_prefix: bool = False,
) -> bool:
    """Check the fixture's minimal causal-prefix obligations."""

    has_prefix = first_target_position > 0 or external_bos
    return (
        has_prefix
        and first_target_position >= required_evidence_end
        and not answer_visible_in_prefix
    )


def serialized_payload(record: SerializedRecord) -> dict[str, list[object]]:
    """Expose every position-indexed field in JSON-compatible form."""

    return {
        "modality": list(record.modality),
        "media_id": list(record.media_id),
        "coordinate_state": list(record.coordinate_state),
        "target_token": list(record.target_token),
        "loss_mask": list(record.loss_mask),
    }


__all__ = [
    "caption_record",
    "crop_box",
    "media_identity_swap_detected",
    "normalize_box",
    "reflect_box_horizontally",
    "validate_serialized_record",
]
