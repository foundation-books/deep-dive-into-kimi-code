"""Dependency-free reference semantics for Chapter 8 Stable LatentMoE.

The code fixes the ten-token Chapter 3 routing identities, evaluates a typed
``d -> lambda -> h -> lambda -> d`` routed branch, and implements exact and
histogram Quantile Balancing checks.  Its matrices are transparent toy
fixtures, not learned Kimi weights or production kernels.
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence

from .reference_fixture import ORIGINS, RUNNING_BIASES, RUNNING_LOGITS, W_DOWN, W_UP
from .reference_proofs import sigmoid, situ_coordinate
from .reference_types import (
    Dimensions,
    ExpertTrace,
    ForwardTrace,
    Matrix,
    RoutingDecision,
    RUNNING_DIMS,
    TokenTrace,
    Vector,
)


def _require_width(vector: Sequence[object], width: int, name: str) -> None:
    if len(vector) != width:
        raise ValueError(f"{name} must have width {width}, got {len(vector)}")


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    """Multiply a row-major matrix by a vector with strict shape checks."""

    if not matrix or not vector:
        raise ValueError("matrix and vector must be nonempty")
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix row width must match vector width")
    return tuple(
        sum(weight * value for weight, value in zip(row, vector, strict=True))
        for row in matrix
    )


def rms_norm(vector: Sequence[float], epsilon: float = 1e-6) -> Vector:
    """Apply gain-one RMSNorm to one vector."""

    if not vector:
        raise ValueError("RMSNorm requires a nonempty vector")
    if epsilon < 0.0:
        raise ValueError("RMSNorm epsilon must be nonnegative")
    squared_mean = sum(value * value for value in vector) / len(vector)
    denominator = sqrt(squared_mean + epsilon)
    if denominator == 0.0:
        raise ValueError("zero vector has undefined epsilon-zero RMSNorm")
    return tuple(value / denominator for value in vector)


def running_inputs(dims: Dimensions = RUNNING_DIMS) -> dict[str, Vector]:
    """Return Chapter 3's deterministic width-eight token payload pattern."""

    if dims.model != 8 or dims.tokens != 10:
        raise ValueError("running inputs are fixed at ten tokens and width eight")
    return {
        origin: tuple(
            (index + 1) * (coordinate + 1) / 100.0 for coordinate in range(dims.model)
        )
        for index, origin in enumerate(ORIGINS)
    }


def route_from_scores(
    scores: Sequence[float],
    biases: Sequence[float],
    selected_count: int,
) -> RoutingDecision:
    """Select with scores+bias and weight with selected raw scores only."""

    if not scores or len(scores) != len(biases):
        raise ValueError("scores and biases must have equal positive width")
    if any(score <= 0.0 for score in scores):
        raise ValueError("execution scores must be positive")
    if not 1 <= selected_count <= len(scores):
        raise ValueError("invalid selected count")
    biased = tuple(score + bias for score, bias in zip(scores, biases, strict=True))
    # The printed fixtures use exact decimal scores and an increasing-index
    # tie rule. Canonicalizing at far finer precision than those inputs keeps
    # equivalent decimal sums (for example .20 + .15 and .70 - .35) tied
    # instead of allowing a float64 ulp to choose the route.
    indices = sorted(
        range(len(scores)), key=lambda index: (-round(biased[index], 12), index)
    )
    selected = tuple(index + 1 for index in indices[:selected_count])
    denominator = sum(scores[index - 1] for index in selected)
    weights = tuple(scores[index - 1] / denominator for index in selected)
    return RoutingDecision(tuple(scores), biased, selected, weights)


def route_from_logits(
    logits: Sequence[float],
    biases: Sequence[float],
    selected_count: int,
) -> RoutingDecision:
    """Sigmoid-score one full-width router row, then route it."""

    return route_from_scores(
        tuple(sigmoid(value) for value in logits), biases, selected_count
    )


def running_decisions() -> dict[str, RoutingDecision]:
    """Return the ten Chapter 3 routing decisions."""

    return {
        origin: route_from_logits(
            RUNNING_LOGITS[origin], RUNNING_BIASES, RUNNING_DIMS.selected_experts
        )
        for origin in ORIGINS
    }


def expert_matrices(expert: int) -> tuple[Matrix, Matrix, Matrix]:
    """Construct small distinct lambda-to-h-to-lambda matrices."""

    if not 1 <= expert <= RUNNING_DIMS.routed_experts:
        raise ValueError("unknown routed expert")
    shift = (expert - 1) % RUNNING_DIMS.latent
    gate_rows: list[Vector] = []
    up_rows: list[Vector] = []
    for row in range(RUNNING_DIMS.intermediate):
        gate_rows.append(
            tuple(
                (1.0 if column == (row + shift) % 4 else 0.0)
                + (0.1 * expert if column == (row + shift + 1) % 4 else 0.0)
                for column in range(4)
            )
        )
        up_rows.append(
            tuple(
                (0.8 if column == (row + shift + 1) % 4 else 0.0)
                - (0.05 * expert if column == (row + shift + 2) % 4 else 0.0)
                for column in range(4)
            )
        )
    output = tuple(
        tuple(
            (0.6 + 0.05 * expert if row == column else 0.0)
            + (0.2 if (row + 1) % 4 == column else 0.0)
            for column in range(3)
        )
        for row in range(4)
    )
    return tuple(gate_rows), tuple(up_rows), output


def routed_expert(
    expert: int,
    latent: Sequence[float],
    beta_gate: float = 4.0,
    beta_up: float = 25.0,
) -> ExpertTrace:
    """Evaluate one toy SiTU-GLU routed expert."""

    _require_width(latent, RUNNING_DIMS.latent, "routed-expert input")
    gate_matrix, up_matrix, output_matrix = expert_matrices(expert)
    gate_pre = matvec(gate_matrix, latent)
    up_pre = matvec(up_matrix, latent)
    intermediate = tuple(
        situ_coordinate(gate, up, beta_gate, beta_up)
        for gate, up in zip(gate_pre, up_pre, strict=True)
    )
    output = matvec(output_matrix, intermediate)
    return ExpertTrace(expert, gate_pre, up_pre, intermediate, output)


def shared_expert(model_input: Sequence[float]) -> Vector:
    """Apply one transparent shape-preserving toy shared map."""

    _require_width(model_input, RUNNING_DIMS.model, "shared-expert input")
    reversed_input = tuple(reversed(model_input))
    return tuple(
        0.1 * value + 0.05 * reverse
        for value, reverse in zip(model_input, reversed_input, strict=True)
    )


def forward_token(
    origin: str,
    model_input: Sequence[float],
    logits: Sequence[float],
    biases: Sequence[float] = RUNNING_BIASES,
    epsilon: float = 1e-6,
) -> TokenTrace:
    """Evaluate the complete typed Stable LatentMoE map for one token."""

    _require_width(model_input, RUNNING_DIMS.model, "model input")
    _require_width(logits, RUNNING_DIMS.routed_experts, "router logits")
    routing = route_from_logits(logits, biases, RUNNING_DIMS.selected_experts)
    latent = matvec(W_DOWN, model_input)
    experts = tuple(routed_expert(expert, latent) for expert in routing.selected)
    aggregate = tuple(
        sum(
            weight * trace.output[coordinate]
            for weight, trace in zip(routing.weights, experts, strict=True)
        )
        for coordinate in range(RUNNING_DIMS.latent)
    )
    normalized = rms_norm(aggregate, epsilon)
    routed_output = matvec(W_UP, normalized)
    shared_output = shared_expert(model_input)
    output = tuple(
        routed + shared
        for routed, shared in zip(routed_output, shared_output, strict=True)
    )
    return TokenTrace(
        origin=origin,
        model_input=tuple(model_input),
        latent_input=latent,
        routing=routing,
        experts=experts,
        routed_aggregate=aggregate,
        normalized_aggregate=normalized,
        routed_output=routed_output,
        shared_output=shared_output,
        output=output,
    )


def stable_latent_moe_forward() -> ForwardTrace:
    """Evaluate the complete ten-token running fixture."""

    inputs = running_inputs()
    return ForwardTrace(
        tuple(
            forward_token(origin, inputs[origin], RUNNING_LOGITS[origin])
            for origin in ORIGINS
        )
    )


def stable_latent_moe_forward_oracle() -> ForwardTrace:
    """Evaluate the same fixture through an explicit source-major route loop."""

    inputs = running_inputs()
    traces: list[TokenTrace] = []
    for origin in ORIGINS:
        model_input = inputs[origin]
        decision = route_from_logits(
            RUNNING_LOGITS[origin], RUNNING_BIASES, RUNNING_DIMS.selected_experts
        )
        latent = matvec(W_DOWN, model_input)
        expert_records = tuple(
            routed_expert(expert, latent) for expert in decision.selected
        )
        aggregate_values = [0.0] * RUNNING_DIMS.latent
        for route_index, expert_record in enumerate(expert_records):
            for coordinate, value in enumerate(expert_record.output):
                aggregate_values[coordinate] += decision.weights[route_index] * value
        aggregate = tuple(aggregate_values)
        normalized = rms_norm(aggregate)
        routed = matvec(W_UP, normalized)
        shared = shared_expert(model_input)
        output = tuple(routed[index] + shared[index] for index in range(8))
        traces.append(
            TokenTrace(
                origin,
                tuple(model_input),
                latent,
                decision,
                expert_records,
                aggregate,
                normalized,
                routed,
                shared,
                output,
            )
        )
    return ForwardTrace(tuple(traces))
