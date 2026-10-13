"""Spatial and temporal realization ledgers for Chapter 16."""

from __future__ import annotations

from .reference_types import GridLedger


def grid_ledger(
    frames: int,
    patch_rows: int,
    patch_columns: int,
    *,
    temporal_group: int = 1,
    merge_rows: int = 2,
    merge_columns: int = 2,
    pad_temporal_tail: bool = False,
) -> GridLedger:
    """Compute exact encoder and backbone counts under declared reductions."""

    if (
        min(
            frames,
            patch_rows,
            patch_columns,
            temporal_group,
            merge_rows,
            merge_columns,
        )
        <= 0
    ):
        raise ValueError("grid and reduction factors must be positive")
    if patch_rows % merge_rows or patch_columns % merge_columns:
        raise ValueError("spatial grid must be divisible by the merge factors")
    temporal_padding = (-frames) % temporal_group
    if temporal_padding and not pad_temporal_tail:
        raise ValueError("temporal tail requires an explicit padding policy")
    realized_frames = frames + temporal_padding
    encoder_positions = frames * patch_rows * patch_columns
    post_temporal = realized_frames // temporal_group * patch_rows * patch_columns
    backbone = post_temporal // (merge_rows * merge_columns)
    return GridLedger(
        frames=frames,
        patch_rows=patch_rows,
        patch_columns=patch_columns,
        temporal_group=temporal_group,
        merge_rows=merge_rows,
        merge_columns=merge_columns,
        temporal_padding=temporal_padding,
        encoder_positions=encoder_positions,
        post_temporal_positions=post_temporal,
        backbone_positions=backbone,
    )


def _image_payload(
    ledger: GridLedger, dimensions: tuple[int, int]
) -> dict[str, object]:
    return {
        "original_dimensions": list(dimensions),
        "realized_dimensions": list(dimensions),
        "patch_grid": [ledger.patch_rows, ledger.patch_columns],
        "merge": [ledger.merge_rows, ledger.merge_columns],
        "encoder_positions": ledger.encoder_positions,
        "backbone_positions": ledger.backbone_positions,
        "padding": ledger.temporal_padding,
        "target_valid": True,
    }


def _wide_payload(ledger: GridLedger) -> dict[str, object]:
    return {
        "original_dimensions": [400, 200],
        "resize_dimensions": [112, 56],
        "padding_top_right_bottom_left": [28, 0, 28, 0],
        "realized_dimensions": [112, 112],
        "patch_grid": [ledger.patch_rows, ledger.patch_columns],
        "merge": [ledger.merge_rows, ledger.merge_columns],
        "encoder_positions": ledger.encoder_positions,
        "backbone_positions": ledger.backbone_positions,
        "target_valid": True,
        "coordinate_map": "scale_by_7_over_25_then_add_y_28",
    }


def _video_payload(ledger: GridLedger) -> dict[str, object]:
    return {
        "sampled_frames": [0, 1, 2, 3],
        "timestamps_seconds": [0, 1, 2, 3],
        "patch_grid_per_frame": [ledger.patch_rows, ledger.patch_columns],
        "temporal_group": ledger.temporal_group,
        "temporal_groups": 1,
        "merge": [ledger.merge_rows, ledger.merge_columns],
        "encoder_positions": ledger.encoder_positions,
        "post_temporal_positions": ledger.post_temporal_positions,
        "backbone_positions": ledger.backbone_positions,
        "padding": ledger.temporal_padding,
        "target_valid": True,
        "target": "square moves right",
    }


def _padded_tail_payload(ledger: GridLedger) -> dict[str, object]:
    return {
        "sampled_frames": [0, 1, 2, 3, 4],
        "temporal_group": ledger.temporal_group,
        "temporal_groups": 2,
        "temporal_padding": ledger.temporal_padding,
        "backbone_positions": ledger.backbone_positions,
        "padding_positions_supervised": False,
        "target_valid": True,
    }


def _studio_realizations(
    low: GridLedger, wide: GridLedger, video: GridLedger
) -> dict[str, dict[str, object]]:
    return {
        "e_T": {
            "realization": "text_only",
            "media_positions": 0,
            "target_valid": True,
        },
        "e_C": {
            "realization": "low",
            "media_positions": low.backbone_positions,
            "target_valid": True,
        },
        "e_G": {
            "realization": "wide_geometry_to_low_grid",
            "media_positions": wide.backbone_positions,
            "target_valid": True,
        },
        "e_R": {
            "realization": "low",
            "media_positions": low.backbone_positions,
            "target_valid": True,
        },
        "e_V": {
            "realization": "four_frame",
            "media_positions": video.backbone_positions,
            "target_valid": True,
        },
    }


def resolution_fixture() -> dict[str, object]:
    """Return exact media realizations and target-validity decisions."""

    low = grid_ledger(1, 8, 8)
    high = grid_ledger(1, 16, 16)
    video = grid_ledger(4, 8, 8, temporal_group=4)
    padded_tail = grid_ledger(5, 8, 8, temporal_group=4, pad_temporal_tail=True)
    wide = grid_ledger(1, 8, 8)
    studio_records = _studio_realizations(low, wide, video)
    return {
        "low": _image_payload(low, (112, 112)),
        "high": _image_payload(high, (224, 224)),
        "wide_geometry_to_low_grid": _wide_payload(wide),
        "four_frame": _video_payload(video),
        "five_frame_padded_tail": _padded_tail_payload(padded_tail),
        "temporal_negative_cases": {
            "reversed_frames_require_relabeling": True,
            "reversed_target": "square moves left",
            "repeated_still_target_rejected": True,
            "missing_event_frames_rejected": True,
        },
        "studio_records": studio_records,
        "all_studio_realizations_declared": set(studio_records)
        == {"e_T", "e_C", "e_G", "e_R", "e_V"},
        "all_studio_targets_valid": all(
            item["target_valid"] is True for item in studio_records.values()
        ),
        "equal_record_expected_backbone_positions": "32",
        "fifty_record_visual_budget": 1600,
        "token_counts_are_not_runtime": True,
    }


__all__ = ["grid_ledger", "resolution_fixture"]
