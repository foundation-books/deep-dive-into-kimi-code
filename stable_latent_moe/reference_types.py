"""Typed records for the Chapter 8 Stable LatentMoE oracle."""

from __future__ import annotations

from dataclasses import dataclass


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]
ScoreMatrix = tuple[Vector, ...]


@dataclass(frozen=True)
class Dimensions:
    """Widths and cardinalities for one Stable LatentMoE layer."""

    tokens: int
    model: int
    latent: int
    intermediate: int
    routed_experts: int
    selected_experts: int
    shared_experts: int

    def __post_init__(self) -> None:
        if (
            min(
                self.tokens,
                self.model,
                self.latent,
                self.intermediate,
                self.routed_experts,
                self.selected_experts,
                self.shared_experts,
            )
            <= 0
        ):
            raise ValueError("all Stable LatentMoE dimensions must be positive")
        if self.latent > self.model:
            raise ValueError("latent width must not exceed model width")
        if self.selected_experts > self.routed_experts:
            raise ValueError("selected count must not exceed routed-expert count")

    @property
    def route_records(self) -> int:
        return self.tokens * self.selected_experts

    @property
    def target_load(self) -> int:
        routes = self.route_records
        if routes % self.routed_experts:
            raise ValueError("exact balanced target is not integral")
        return routes // self.routed_experts


RUNNING_DIMS = Dimensions(
    tokens=10,
    model=8,
    latent=4,
    intermediate=3,
    routed_experts=4,
    selected_experts=2,
    shared_experts=1,
)

K3_DIMS = Dimensions(
    tokens=1,
    model=7168,
    latent=3584,
    intermediate=3072,
    routed_experts=896,
    selected_experts=16,
    shared_experts=2,
)


@dataclass(frozen=True)
class RoutingDecision:
    """One ordered selected set and its raw-score execution weights."""

    raw_scores: Vector
    biased_scores: Vector
    selected: tuple[int, ...]
    weights: Vector


@dataclass(frozen=True)
class ExpertTrace:
    """One selected routed expert's typed intermediate values."""

    expert: int
    gate_pre: Vector
    up_pre: Vector
    situ_intermediate: Vector
    output: Vector


@dataclass(frozen=True)
class TokenTrace:
    """Complete Stable LatentMoE computation for one token row."""

    origin: str
    model_input: Vector
    latent_input: Vector
    routing: RoutingDecision
    experts: tuple[ExpertTrace, ...]
    routed_aggregate: Vector
    normalized_aggregate: Vector
    routed_output: Vector
    shared_output: Vector
    output: Vector


@dataclass(frozen=True)
class ForwardTrace:
    """One ten-token forward with immutable token origins."""

    tokens: tuple[TokenTrace, ...]


@dataclass(frozen=True)
class HistogramResult:
    """Per-expert count histogram and recovered required-bias quantiles."""

    lower: float
    upper: float
    width: float
    counts: tuple[tuple[int, ...], ...]
    recovered: Vector


@dataclass(frozen=True)
class ResourceLedger:
    """Logical counts whose units are explicit and not timing proxies."""

    router_parameters: int
    down_parameters: int
    routed_expert_parameters: int
    aggregate_norm_parameters: int
    up_parameters: int
    matrix_multiply_adds_per_token: int
    route_payload_scalars: int


__all__ = [
    "Dimensions",
    "ExpertTrace",
    "ForwardTrace",
    "HistogramResult",
    "K3_DIMS",
    "Matrix",
    "ResourceLedger",
    "RoutingDecision",
    "RUNNING_DIMS",
    "ScoreMatrix",
    "TokenTrace",
    "Vector",
]
