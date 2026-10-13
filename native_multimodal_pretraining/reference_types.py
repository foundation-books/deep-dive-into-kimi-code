"""Shared exact types for the Chapter 16 companion."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


Box = tuple[Fraction, Fraction, Fraction, Fraction]


@dataclass(frozen=True)
class SerializedRecord:
    """A minimal typed multimodal record after serialization."""

    modality: tuple[str, ...]
    media_id: tuple[str | None, ...]
    coordinate_state: tuple[str | None, ...]
    target_token: tuple[str | None, ...]
    loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class FamilySpec:
    """Exact family expectations under one declared cost unit."""

    name: str
    probability: Fraction
    backbone_length: int
    target_count: int
    visual_positions: int
    compute_units: int


@dataclass(frozen=True)
class GridLedger:
    """Spatial and temporal position counts for one realized media item."""

    frames: int
    patch_rows: int
    patch_columns: int
    temporal_group: int
    merge_rows: int
    merge_columns: int
    temporal_padding: int
    encoder_positions: int
    post_temporal_positions: int
    backbone_positions: int


__all__ = ["Box", "FamilySpec", "GridLedger", "SerializedRecord"]
