"""Typed state for the Chapter 7 Attention Residuals oracle."""

from __future__ import annotations

from dataclasses import dataclass


Vector = tuple[float, ...]
Grid = tuple[Vector, ...]

RUNNING_LABELS = (
    "v1",
    "v2",
    "v3",
    "v4",
    "which",
    "shape",
    "below",
    "red",
    "circle",
    "?",
)


@dataclass(frozen=True)
class Dimensions:
    """Token-grid and depth dimensions for one AttnRes schedule."""

    tokens: int
    model: int
    decoder_layers: int
    block_size: int
    points_per_layer: int = 2

    def __post_init__(self) -> None:
        if (
            min(
                self.tokens,
                self.model,
                self.decoder_layers,
                self.block_size,
                self.points_per_layer,
            )
            <= 0
        ):
            raise ValueError("all dimensions must be positive")

    @property
    def transformations(self) -> int:
        return self.decoder_layers * self.points_per_layer

    @property
    def block_lengths(self) -> tuple[int, ...]:
        complete, remainder = divmod(self.decoder_layers, self.block_size)
        lengths = (self.block_size,) * complete
        return lengths + ((remainder,) if remainder else ())

    @property
    def blocks(self) -> int:
        return len(self.block_lengths)

    @property
    def terminal_sources(self) -> int:
        return 1 + self.blocks


RUNNING_DIMS = Dimensions(
    tokens=10,
    model=8,
    decoder_layers=7,
    block_size=3,
)
K3_DIMS = Dimensions(
    tokens=1,
    model=7168,
    decoder_layers=93,
    block_size=12,
)


@dataclass(frozen=True)
class MixResult:
    """One token-wise depth mixture over a declared source axis."""

    output: Grid
    logits: tuple[Vector, ...]
    weights: tuple[Vector, ...]


@dataclass(frozen=True)
class FullStep:
    """One Full AttnRes target and the transformation it feeds."""

    target: int
    decoder_layer: int
    point: str
    source_ids: tuple[str, ...]
    mixture: MixResult
    transformation_output: Grid


@dataclass(frozen=True)
class FullTrace:
    """Complete Full AttnRes trace plus terminal depth aggregate."""

    embedding: Grid
    steps: tuple[FullStep, ...]
    terminal_source_ids: tuple[str, ...]
    terminal: MixResult


@dataclass(frozen=True)
class DepthIdentity:
    """Identity and lifetime of one ephemeral depth-history state."""

    forward_id: str
    model_revision: str
    tokens: int
    width: int
    lifetime: str = "one-model-forward"


@dataclass(frozen=True)
class BlockState:
    """Completed block summaries plus at most one current partial sum."""

    identity: DepthIdentity
    embedding: Grid
    completed: tuple[Grid, ...] = ()
    completed_indices: tuple[tuple[int, ...], ...] = ()
    partial: Grid | None = None
    partial_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlockStep:
    """One source-resolved pre-attention or pre-MLP Block AttnRes step."""

    target: int
    decoder_layer: int
    point: str
    block_index: int
    source_ids: tuple[str, ...]
    mixture: MixResult
    transformation_output: Grid
    state_after: BlockState


@dataclass(frozen=True)
class BlockTrace:
    """Complete Block AttnRes trace plus terminal depth aggregate."""

    steps: tuple[BlockStep, ...]
    terminal_source_ids: tuple[str, ...]
    terminal: MixResult
    terminal_state: BlockState


@dataclass(frozen=True)
class OnlineState:
    """Stable softmax statistics for one token and one source group."""

    maximum: float
    normalizer: float
    numerator: Vector


@dataclass(frozen=True)
class ResourceLedger:
    """Logical parameter, interaction, and maximum live-source counts."""

    pseudo_query_parameters: int
    source_interactions: int
    maximum_live_sources: int
    live_source_scalars: int


__all__ = [
    "BlockState",
    "BlockStep",
    "BlockTrace",
    "DepthIdentity",
    "Dimensions",
    "FullStep",
    "FullTrace",
    "Grid",
    "K3_DIMS",
    "MixResult",
    "OnlineState",
    "RUNNING_DIMS",
    "RUNNING_LABELS",
    "ResourceLedger",
    "Vector",
]
