"""Typed schedule and accounting contracts for the Chapter 5 reference."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Sequence, cast


LayerType = Literal["K", "G"]
Vector = tuple[Fraction, ...]
Schedule = tuple[LayerType, ...]

RUNNING_SCHEDULE: Schedule = ("K", "K", "K", "G", "K", "K", "K", "G")
RUNNING_MODEL_WIDTH = 8
RUNNING_HEADS = 2
RUNNING_KEY_WIDTH = 2
RUNNING_VALUE_WIDTH = 2
RUNNING_GLOBAL_RECORD_WIDTH = 2


@dataclass(frozen=True)
class TokenInput:
    """One already-projected toy hidden vector."""

    label: str
    values: Vector


@dataclass(frozen=True)
class LayerInventory:
    """Counts induced by one typed schedule."""

    layers: int
    kda_layers: int
    global_layers: int
    complete_blocks: int
    terminal_global: bool


@dataclass(frozen=True)
class KdaLayerState:
    """Opaque fixed-extent memory for one KDA layer."""

    layer: int
    prefix_length: int
    values: Vector


@dataclass(frozen=True)
class GlobalLayerCache:
    """Ordered token records for one causal global layer."""

    layer: int
    prefix_length: int
    records: tuple[Vector, ...]


@dataclass(frozen=True)
class HybridPrefixState:
    """Complete accepted-prefix state across all hybrid layers."""

    schedule: Schedule
    prefix_length: int
    kda_states: tuple[KdaLayerState, ...]
    global_caches: tuple[GlobalLayerCache, ...]


@dataclass(frozen=True)
class LayerStep:
    """One layer's output and persistent-memory extent at one position."""

    layer: int
    layer_type: LayerType
    output: Vector
    persistent_elements: int
    prefix_length: int


@dataclass(frozen=True)
class PositionTrace:
    """Depth-major trace for one accepted position."""

    position: int
    label: str
    layer_steps: tuple[LayerStep, ...]
    output: Vector


@dataclass(frozen=True)
class HybridTrace:
    """Position-major hybrid trace and its final accepted-prefix state."""

    positions: tuple[PositionTrace, ...]
    final_state: HybridPrefixState


@dataclass(frozen=True)
class LogicalFootprint:
    """Narrow persistent-element ledger."""

    kda_elements: int
    global_elements: int
    total_elements: int


def f(numerator: int, denominator: int = 1) -> Fraction:
    """Return an exact rational used by the frozen reference."""

    return Fraction(numerator, denominator)


def validate_schedule(schedule: Sequence[str]) -> Schedule:
    """Return a nonempty typed schedule or raise."""

    if not schedule:
        raise ValueError("a hybrid schedule must contain at least one layer")
    if any(layer_type not in {"K", "G"} for layer_type in schedule):
        raise ValueError("schedule entries must be K or G")
    return cast(Schedule, tuple(schedule))


def repeating_schedule(
    ratio: int,
    complete_blocks: int,
    *,
    terminal_global: bool = False,
) -> Schedule:
    """Construct K repeated ratio times followed by G in every block."""

    if ratio < 1:
        raise ValueError("ratio must be positive")
    if complete_blocks < 1:
        raise ValueError("complete_blocks must be positive")
    block = cast(Schedule, tuple(["K"] * ratio + ["G"]))
    schedule = block * complete_blocks
    if terminal_global:
        schedule += ("G",)
    return schedule


def inventory(
    schedule: Sequence[str],
    *,
    ratio: int = 3,
) -> LayerInventory:
    """Count layer types, complete leading blocks, and a terminal exception."""

    typed = validate_schedule(schedule)
    block = tuple(["K"] * ratio + ["G"])
    blocks = 0
    offset = 0
    while typed[offset : offset + len(block)] == block:
        blocks += 1
        offset += len(block)
    terminal = typed[offset:] == ("G",)
    return LayerInventory(
        layers=len(typed),
        kda_layers=typed.count("K"),
        global_layers=typed.count("G"),
        complete_blocks=blocks,
        terminal_global=terminal,
    )


def validate_repeating_inventory(
    schedule: Sequence[str],
    *,
    ratio: int,
    complete_blocks: int,
    terminal_global: bool,
) -> LayerInventory:
    """Require exactly the declared repeating schedule."""

    expected = repeating_schedule(
        ratio, complete_blocks, terminal_global=terminal_global
    )
    typed = validate_schedule(schedule)
    if typed != expected:
        raise ValueError("schedule does not match the declared repeating inventory")
    return inventory(typed, ratio=ratio)


def k3_schedule() -> Schedule:
    """Return the report's 23 complete 3:1 blocks plus terminal global layer."""

    return repeating_schedule(3, 23, terminal_global=True)


def logical_footprint(
    prefix_length: int,
    *,
    schedule: Sequence[str] = RUNNING_SCHEDULE,
    batch: int = 1,
    heads: int = RUNNING_HEADS,
    key_width: int = RUNNING_KEY_WIDTH,
    value_width: int = RUNNING_VALUE_WIDTH,
    global_record_width: int = RUNNING_GLOBAL_RECORD_WIDTH,
) -> LogicalFootprint:
    """Count declared persistent elements and nothing else."""

    if prefix_length < 0:
        raise ValueError("prefix_length must be nonnegative")
    values = (batch, heads, key_width, value_width, global_record_width)
    if any(value < 1 for value in values):
        raise ValueError("all dimensions must be positive")
    counts = inventory(schedule)
    kda = batch * counts.kda_layers * heads * key_width * value_width
    global_elements = batch * counts.global_layers * prefix_length * global_record_width
    return LogicalFootprint(kda, global_elements, kda + global_elements)


def relative_global_cache_slope(schedule: Sequence[str]) -> Fraction:
    """Return L_G/L against an equal-width all-global schedule."""

    counts = inventory(schedule)
    return f(counts.global_layers, counts.layers)


def ratio_inventory(depth: int, ratio: int) -> LayerInventory:
    """Return an equal-depth complete-block inventory, with 0 meaning all-global."""

    if depth < 1:
        raise ValueError("depth must be positive")
    if ratio == 0:
        return inventory(("G",) * depth)
    block_length = ratio + 1
    if ratio < 1 or depth % block_length:
        raise ValueError("depth must contain complete ratio blocks")
    return inventory(repeating_schedule(ratio, depth // block_length), ratio=ratio)


def kda_prefill_flops(length: int, chunk_size: int, head_width: int) -> int:
    """Evaluate Kimi Linear Equation 13 for one head."""

    if min(length, chunk_size, head_width) < 1:
        raise ValueError("cost dimensions must be positive")
    return (
        6 * length * head_width**2
        + 3 * length * chunk_size * head_width
        + length * chunk_size**2
    )


def global_prefill_flops(length: int, head_width: int) -> int:
    """Evaluate Kimi Linear Equation 14 for one head."""

    if min(length, head_width) < 1:
        raise ValueError("cost dimensions must be positive")
    return 2 * length**2 * head_width


def hybrid_prefill_attention_flops(
    length: int,
    chunk_size: int,
    head_width: int,
    schedule: Sequence[str] = RUNNING_SCHEDULE,
) -> int:
    """Assemble attention-only work over the typed layer inventory."""

    counts = inventory(schedule)
    return counts.kda_layers * kda_prefill_flops(
        length, chunk_size, head_width
    ) + counts.global_layers * global_prefill_flops(length, head_width)


def fraction_text(value: Fraction) -> str:
    """Render one exact scalar compactly."""

    return str(value.numerator) if value.denominator == 1 else str(value)


__all__ = (
    "GlobalLayerCache",
    "HybridPrefixState",
    "HybridTrace",
    "KdaLayerState",
    "LayerInventory",
    "LayerStep",
    "LayerType",
    "LogicalFootprint",
    "PositionTrace",
    "RUNNING_GLOBAL_RECORD_WIDTH",
    "RUNNING_HEADS",
    "RUNNING_KEY_WIDTH",
    "RUNNING_MODEL_WIDTH",
    "RUNNING_SCHEDULE",
    "RUNNING_VALUE_WIDTH",
    "Schedule",
    "TokenInput",
    "Vector",
    "f",
    "fraction_text",
    "global_prefill_flops",
    "hybrid_prefill_attention_flops",
    "inventory",
    "k3_schedule",
    "kda_prefill_flops",
    "logical_footprint",
    "ratio_inventory",
    "relative_global_cache_slope",
    "repeating_schedule",
    "validate_repeating_inventory",
    "validate_schedule",
)
