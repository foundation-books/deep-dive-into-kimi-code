"""Types and frozen dimensions for the Chapter 6 Gated MLA oracle."""

from __future__ import annotations

from dataclasses import dataclass


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]

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
DECODE_LABEL = "square"


@dataclass(frozen=True)
class Dimensions:
    """Widths of one Gated MLA layer."""

    model: int
    heads: int
    query_rank: int
    latent: int
    content_key: int
    shared_key: int
    value: int

    def __post_init__(self) -> None:
        if (
            min(
                self.model,
                self.heads,
                self.query_rank,
                self.latent,
                self.content_key,
                self.shared_key,
                self.value,
            )
            <= 0
        ):
            raise ValueError("all dimensions must be positive")

    @property
    def query_head(self) -> int:
        return self.content_key + self.shared_key

    @property
    def query_projection(self) -> int:
        return self.heads * self.query_head

    @property
    def attention_output(self) -> int:
        return self.heads * self.value

    @property
    def compact_record(self) -> int:
        return self.latent + self.shared_key

    @property
    def materialized_record(self) -> int:
        return self.heads * (self.query_head + self.value)


RUNNING_DIMS = Dimensions(
    model=8,
    heads=2,
    query_rank=4,
    latent=2,
    content_key=2,
    shared_key=1,
    value=2,
)
K3_DIMS = Dimensions(
    model=7168,
    heads=96,
    query_rank=1536,
    latent=512,
    content_key=128,
    shared_key=64,
    value=128,
)


@dataclass(frozen=True)
class Weights:
    """Dense matrices needed by the toy layer."""

    q_down: Matrix
    q_up: Matrix
    kv_down: Matrix
    k_up: tuple[Matrix, ...]
    v_up: tuple[Matrix, ...]
    gate: Matrix
    output: Matrix


@dataclass(frozen=True)
class CompactRecord:
    """One logical token cache record: normalized latent plus shared key."""

    latent: Vector
    shared_key: Vector


@dataclass(frozen=True)
class Projection:
    """All query-side and record-side projections for one token."""

    query_content: tuple[Vector, ...]
    query_shared: tuple[Vector, ...]
    record: CompactRecord
    keys: tuple[Vector, ...]
    values: tuple[Vector, ...]
    gate: Vector


@dataclass(frozen=True)
class PositionTrace:
    """One causal output with its probabilities and compact record."""

    position: int
    label: str
    weights: tuple[Vector, ...]
    ungated: Vector
    gate: Vector
    output: Vector
    record: CompactRecord


@dataclass(frozen=True)
class SequenceTrace:
    """Complete reference or absorbed evaluation."""

    positions: tuple[PositionTrace, ...]


@dataclass(frozen=True)
class CacheIdentity:
    """Semantic identity of one accepted-prefix layer cache."""

    request_id: str
    model_revision: str
    layer: int
    endpoint: int
    convention: str = "normalized-latent-plus-shared-nope-key"
    dtype: str = "float64-reference"


@dataclass(frozen=True)
class LatentCache:
    """Ordered compact records for one layer and one accepted prefix."""

    identity: CacheIdentity
    records: tuple[CompactRecord, ...] = ()


__all__ = [
    "CacheIdentity",
    "CompactRecord",
    "DECODE_LABEL",
    "Dimensions",
    "K3_DIMS",
    "LatentCache",
    "Matrix",
    "PositionTrace",
    "Projection",
    "RUNNING_DIMS",
    "RUNNING_LABELS",
    "SequenceTrace",
    "Vector",
    "Weights",
]
