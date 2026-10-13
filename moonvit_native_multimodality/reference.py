"""Dependency-free reference contracts for Chapter 9.

The routines make grid, packing, merge, projection, splice, and accounting
invariants executable.  They are small mathematical fixtures, not the released
MoonViT-V2 model, a video preprocessor, or a performance benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, inf, sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RealizedImage:
    """Original and budget-realized image geometry."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_width: int
    pad_height: int
    patch_size: int
    merge_size: int

    @property
    def canvas_width(self) -> int:
        return self.resized_width + self.pad_width

    @property
    def canvas_height(self) -> int:
        return self.resized_height + self.pad_height

    @property
    def patch_rows(self) -> int:
        return self.canvas_height // self.patch_size

    @property
    def patch_cols(self) -> int:
        return self.canvas_width // self.patch_size

    @property
    def premerge_tokens(self) -> int:
        return self.patch_rows * self.patch_cols

    @property
    def merged_rows(self) -> int:
        return self.patch_rows // self.merge_size

    @property
    def merged_cols(self) -> int:
        return self.patch_cols // self.merge_size

    @property
    def projected_tokens(self) -> int:
        return self.merged_rows * self.merged_cols


@dataclass(frozen=True)
class PackedGrid:
    """Concatenated patch coordinates plus item boundaries."""

    offsets: tuple[int, ...]
    sample_ids: tuple[int, ...]
    coordinates: tuple[tuple[int, int, int, int], ...]

    @property
    def length(self) -> int:
        return len(self.sample_ids)


def realize_image(
    width: int,
    height: int,
    *,
    patch_size: int = 14,
    merge_size: int = 2,
    in_patch_limit: int = 65_536,
    patch_limit_on_one_side: int = 512,
) -> RealizedImage:
    """Reproduce the pinned public K3 image size and padding arithmetic."""

    if min(width, height, patch_size, merge_size, in_patch_limit) <= 0:
        raise ValueError("dimensions and limits must be positive")
    if patch_limit_on_one_side <= 0:
        raise ValueError("one-side patch limit must be positive")
    estimated_patches = max(1.0, width // patch_size) * max(1.0, height // patch_size)
    scale = min(
        1.0,
        sqrt(in_patch_limit / estimated_patches),
        patch_limit_on_one_side * patch_size / width,
        patch_limit_on_one_side * patch_size / height,
    )
    resized_width = min(
        max(1, int(width * scale)), patch_limit_on_one_side * patch_size
    )
    resized_height = min(
        max(1, int(height * scale)), patch_limit_on_one_side * patch_size
    )
    factor = patch_size * merge_size
    pad_width = (factor - resized_width % factor) % factor
    pad_height = (factor - resized_height % factor) % factor
    return RealizedImage(
        width,
        height,
        resized_width,
        resized_height,
        pad_width,
        pad_height,
        patch_size,
        merge_size,
    )


def row_major_patch_coordinates(
    frames: int, rows: int, cols: int
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(frame,row,column)`` coordinates in patchification order."""

    if min(frames, rows, cols) <= 0:
        raise ValueError("grid dimensions must be positive")
    return tuple(
        (frame, row, col)
        for frame in range(frames)
        for row in range(rows)
        for col in range(cols)
    )


def pack_grids(
    grids: Sequence[tuple[int, int, int]],
) -> PackedGrid:
    """Pack ``(frames,rows,cols)`` grids while preserving item coordinates."""

    if not grids:
        raise ValueError("at least one grid is required")
    offsets = [0]
    sample_ids: list[int] = []
    coordinates: list[tuple[int, int, int, int]] = []
    for sample, (frames, rows, cols) in enumerate(grids):
        local = row_major_patch_coordinates(frames, rows, cols)
        sample_ids.extend([sample] * len(local))
        coordinates.extend((sample, *coordinate) for coordinate in local)
        offsets.append(offsets[-1] + len(local))
    return PackedGrid(tuple(offsets), tuple(sample_ids), tuple(coordinates))


def isolation_mask(sample_ids: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    """Return the additive sample-isolation mask for packed attention."""

    if not sample_ids:
        raise ValueError("sample ids must be nonempty")
    return tuple(
        tuple(0.0 if left == right else -inf for right in sample_ids)
        for left in sample_ids
    )


def mask_is_sample_isolated(
    sample_ids: Sequence[int], mask: Sequence[Sequence[float]]
) -> bool:
    """Return whether a mask permits exactly within-sample attention."""
    if len(mask) != len(sample_ids) or any(len(row) != len(sample_ids) for row in mask):
        return False
    return all(
        (mask[i][j] == 0.0) if sample_ids[i] == sample_ids[j] else (mask[i][j] == -inf)
        for i in range(len(sample_ids))
        for j in range(len(sample_ids))
    )


def merge_groups_2x2(rows: int, cols: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return source indices for row-major 2-by-2 spatial rearrangement."""

    if rows <= 0 or cols <= 0 or rows % 2 or cols % 2:
        raise ValueError("2-by-2 merge requires positive even grid dimensions")
    return tuple(
        (
            (2 * row) * cols + 2 * col,
            (2 * row) * cols + 2 * col + 1,
            (2 * row + 1) * cols + 2 * col,
            (2 * row + 1) * cols + 2 * col + 1,
        )
        for row in range(rows // 2)
        for col in range(cols // 2)
    )


def merge_features_2x2(
    features: Sequence[Sequence[float]], rows: int, cols: int
) -> tuple[tuple[float, ...], ...]:
    """Concatenate each 2-by-2 group without discarding a source coordinate."""

    if len(features) != rows * cols:
        raise ValueError("feature count does not match the declared grid")
    if not features or not features[0]:
        raise ValueError("features must be nonempty")
    width = len(features[0])
    if any(len(feature) != width for feature in features):
        raise ValueError("features must have one common width")
    return tuple(
        tuple(value for index in group for value in features[index])
        for group in merge_groups_2x2(rows, cols)
    )


def attention_interactions(frames: int, spatial_sites: int) -> dict[str, int]:
    """Count full and report-level factorized score interactions."""

    if frames <= 0 or spatial_sites <= 0:
        raise ValueError("frames and spatial sites must be positive")
    full = (frames * spatial_sites) ** 2
    spatial = frames * spatial_sites**2
    temporal = spatial_sites * frames**2
    return {
        "full": full,
        "factorized_spatial": spatial,
        "factorized_temporal": temporal,
        "factorized_total": spatial + temporal,
    }


def temporal_pool(
    frame_vectors: Sequence[Sequence[float]], group_size: int, *, tail: str = "keep"
) -> tuple[tuple[float, ...], ...]:
    """Mean-pool consecutive frames with an explicit incomplete-tail rule."""

    if group_size <= 0 or not frame_vectors or not frame_vectors[0]:
        raise ValueError("nonempty vectors and positive group size are required")
    width = len(frame_vectors[0])
    if any(len(vector) != width for vector in frame_vectors):
        raise ValueError("frame vectors must have one common width")
    if tail not in {"keep", "reject"}:
        raise ValueError("tail must be 'keep' or 'reject'")
    if tail == "reject" and len(frame_vectors) % group_size:
        raise ValueError("incomplete temporal group")
    pooled = []
    for start in range(0, len(frame_vectors), group_size):
        group = frame_vectors[start : start + group_size]
        pooled.append(
            tuple(sum(vector[j] for vector in group) / len(group) for j in range(width))
        )
    return tuple(pooled)


def _matvec(
    vector: Sequence[float], matrix: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    if not vector or len(matrix) != len(vector) or not matrix[0]:
        raise ValueError("matrix input dimension must equal vector width")
    output_width = len(matrix[0])
    if any(len(row) != output_width for row in matrix):
        raise ValueError("matrix must be rectangular")
    return tuple(
        sum(vector[i] * matrix[i][j] for i in range(len(vector)))
        for j in range(output_width)
    )


def rms_norm(vector: Sequence[float], epsilon: float = 1e-6) -> tuple[float, ...]:
    """Return the vector after RMS normalization."""
    if not vector or epsilon < 0.0:
        raise ValueError("RMSNorm requires a nonempty vector and nonnegative epsilon")
    denominator = sqrt(sum(value * value for value in vector) / len(vector) + epsilon)
    if denominator == 0.0:
        raise ValueError("zero vector is undefined when epsilon is zero")
    return tuple(value / denominator for value in vector)


def bias_free_mlp_projector(
    merged: Sequence[float],
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Evaluate Linear--GELU--Linear--RMSNorm with no additive biases."""

    hidden = _matvec(merged, first)
    activated = tuple(0.5 * value * (1.0 + erf(value / sqrt(2.0))) for value in hidden)
    return rms_norm(_matvec(activated, second))


def splice_media(
    template: Sequence[str], media: Sequence[Sequence[str]], marker: str = "<image>"
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Replace ordered media markers and return half-open visual spans."""

    if sum(token == marker for token in template) != len(media):
        raise ValueError("media marker count must equal media item count")
    output: list[str] = []
    spans: list[tuple[int, int]] = []
    media_index = 0
    for token in template:
        if token != marker:
            output.append(token)
            continue
        item = tuple(media[media_index])
        if not item:
            raise ValueError("each media item must contribute at least one token")
        start = len(output)
        output.extend(item)
        spans.append((start, len(output)))
        media_index += 1
    return tuple(output), tuple(spans)


def context_budget(
    text_tokens: int, control_tokens: int, visual_tokens: Iterable[int]
) -> int:
    """Return the total text, control, and visual token count."""
    counts = tuple(visual_tokens)
    if min((text_tokens, control_tokens, *counts), default=0) < 0:
        raise ValueError("token counts must be nonnegative")
    return text_tokens + control_tokens + sum(counts)


def running_trace() -> dict[str, object]:
    """Return the complete scaled-down red-circle/blue-square ledger."""

    image = realize_image(56, 56)
    packed = pack_grids(((1, image.patch_rows, image.patch_cols), (1, 4, 6)))
    stream, spans = splice_media(
        ("<image>", "Which", "shape", "is", "below", "red", "circle?"),
        (("v1", "v2", "v3", "v4"),),
    )
    return {
        "image": {
            "original_width": image.original_width,
            "original_height": image.original_height,
            "resized_width": image.resized_width,
            "resized_height": image.resized_height,
            "pad_width": image.pad_width,
            "pad_height": image.pad_height,
            "patch_size": image.patch_size,
            "merge_size": image.merge_size,
        },
        "premerge_tokens": image.premerge_tokens,
        "projected_tokens": image.projected_tokens,
        "packed_offsets": packed.offsets,
        "packed_length": packed.length,
        "stream": stream,
        "visual_spans": spans,
        "backbone_positions": len(stream),
        "next_token_not_in_prompt": "square" not in stream,
    }


def fault_detection_summary() -> dict[str, bool]:
    """Exercise the smallest oracle for seven common contract violations."""

    packed = pack_grids(((1, 2, 2), (1, 2, 3)))
    good_mask = isolation_mask(packed.sample_ids)
    bad_mask = tuple(tuple(0.0 for _ in packed.sample_ids) for _ in packed.sample_ids)
    checks: dict[str, bool] = {
        "cross_sample_attention": not mask_is_sample_isolated(
            packed.sample_ids, bad_mask
        ),
        "isolated_reference_accepted": mask_is_sample_isolated(
            packed.sample_ids, good_mask
        ),
    }
    for name, action in (
        ("odd_spatial_merge", lambda: merge_groups_2x2(3, 4)),
        ("placeholder_mismatch", lambda: splice_media(("<image>",), ())),
        ("coordinate_count_mismatch", lambda: merge_features_2x2(((1.0,),), 2, 2)),
        (
            "temporal_tail_undeclared",
            lambda: temporal_pool(((1.0,),) * 3, 2, tail="reject"),
        ),
        ("negative_context_count", lambda: context_budget(6, 0, (4, -1))),
    ):
        try:
            action()
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    return checks
