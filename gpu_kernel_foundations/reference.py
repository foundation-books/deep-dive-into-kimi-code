"""Finite worker, address and lifetime arithmetic; no GPU simulation."""

from collections.abc import Sequence
from typing import NamedTuple

WARP_SIZE = 32
MAX_LENGTH = 128
OFFSETS = (1, 2, 4, 8, 16)
MATRIX_A = ((1, 0, 2, 0), (0, 1, 0, 2), (1, 2, 3, 4), (2, 1, 0, 1))
MATRIX_B = ((1, 2, 0, 0), (3, 4, 0, 0), (5, 6, 1, 0), (7, 8, 0, 1))


class TileCopyRow(NamedTuple):
    """One local value in the chapter's four-worker copy partition."""

    worker: int
    local_value: int
    coordinate: tuple[int, int]
    source_offset: int
    destination_offset: int
    value: int


class StageUse(NamedTuple):
    """A teaching generation, not a hardware barrier state."""

    slice: int
    slot: int
    generation: int
    load_interval: tuple[int, int]
    compute_interval: tuple[int, int]
    reusable_at: int


def warp_and_lane(thread: int, block_threads: int = 128) -> tuple[int, int]:
    """Return block-local identities for a one-dimensional block."""
    if not 0 <= thread < block_threads:
        raise ValueError("thread must belong to the declared block")
    return divmod(thread, WARP_SIZE)


def lane_indices(length: int, lane: int) -> tuple[int, ...]:
    """Valid striped coordinates; empty work does not remove a participant."""
    if not 0 <= length <= MAX_LENGTH:
        raise ValueError("length must be between 0 and 128")
    if not 0 <= lane < WARP_SIZE:
        raise ValueError("lane must be between 0 and 31")
    return tuple(range(lane, length, WARP_SIZE))


def butterfly_stages(partials: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Include the initial state and five exchanges, each using the old state."""
    if len(partials) != WARP_SIZE:
        raise ValueError("the example requires all 32 participants")
    stages = [tuple(partials)]
    for offset in OFFSETS:
        previous = stages[-1]
        stages.append(
            tuple(previous[lane] + previous[lane ^ offset] for lane in range(WARP_SIZE))
        )
    return tuple(stages)


def dot_trace(x: Sequence[int], weights: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Compute exact products in 32 lanes; lane 0 supplies the sole output."""
    if len(x) != len(weights):
        raise ValueError("operands must have the same length")
    partials = [
        sum(x[i] * weights[i] for i in lane_indices(len(x), lane))
        for lane in range(WARP_SIZE)
    ]
    return butterfly_stages(partials)


def tile_slice(first_k: int) -> tuple[tuple[int, ...], ...]:
    """Compute one two-term contribution to the lower-left output tile."""
    if first_k not in (0, 2):
        raise ValueError("the fixed example has slices beginning at 0 and 2")
    return tuple(
        tuple(
            sum(MATRIX_A[i][k] * MATRIX_B[k][j] for k in range(first_k, first_k + 2))
            for j in range(2)
        )
        for i in range(2, 4)
    )


def tile_copy_rows() -> tuple[TileCopyRow, ...]:
    """Invert worker=2*u+v, then evaluate two distinct storage layouts."""
    rows = []
    for worker in range(4):
        u, v = divmod(worker, 2)
        value = sum(MATRIX_A[2 + u][k] * MATRIX_B[k][v] for k in range(4))
        rows.append(TileCopyRow(worker, 0, (u, v), 8 + 4 * u + v, 2 * u + v, value))
    return tuple(rows)


def xor_storage_offset(row: int, column: int) -> int:
    """Element offset in the separate 32-by-32 scalar bank example."""
    if not (0 <= row < 32 and 0 <= column < 32):
        raise ValueError("row and column must be between 0 and 31")
    return 32 * row + (column ^ row)


def two_slot_schedule(
    slices: int, load: int = 2, compute: int = 3
) -> tuple[StageUse, ...]:
    """One loader and one computer; slots release at compute completion."""
    if slices < 0 or load <= 0 or compute <= 0:
        raise ValueError("nonnegative slice count and positive durations required")
    loader_free = computer_free = 0
    reusable = [0, 0]
    uses = []
    for index in range(slices):
        slot = index % 2
        load_start = max(loader_free, reusable[slot])
        load_end = load_start + load
        compute_start = max(computer_free, load_end)
        compute_end = compute_start + compute
        uses.append(
            StageUse(
                index,
                slot,
                index // 2,
                (load_start, load_end),
                (compute_start, compute_end),
                compute_end,
            )
        )
        loader_free, computer_free = load_end, compute_end
        reusable[slot] = compute_end
    return tuple(uses)
