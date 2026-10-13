"""Dependency-free reference semantics for Chapter 12.

The routines are intentionally small and transparent.  They reproduce the
chapter's finite matrix examples and contracts; they are not a training
implementation and make no throughput or model-quality claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .reference_matrix import (
    add,
    newton_schulz_zeroth_power,
    scale,
    shape,
    shape_scale,
    zeros,
)
from .reference_types import FlatTensor, Matrix


def momentum_candidate(
    previous: Matrix,
    gradient: Matrix,
    coefficient: float = 0.95,
    nesterov: bool = True,
) -> tuple[Matrix, Matrix]:
    """Return the new buffer and the matrix sent to orthogonalization."""

    if not 0 <= coefficient < 1:
        raise ValueError("momentum coefficient must lie in [0,1)")
    buffer = add(scale(previous, coefficient), gradient)
    candidate = add(gradient, scale(buffer, coefficient)) if nesterov else buffer
    return buffer, candidate


def decoupled_weight_decay(
    weight: Matrix, learning_rate: float, decay: float
) -> Matrix:
    """Apply W <- (1-eta*lambda)W independently of the update direction."""

    if learning_rate < 0 or decay < 0:
        raise ValueError("learning rate and decay must be nonnegative")
    return scale(weight, 1.0 - learning_rate * decay)


@dataclass(frozen=True)
class MuonStep:
    """Capture an updated weight, momentum buffer, candidate, and direction."""

    weight: Matrix
    momentum: Matrix
    candidate: Matrix
    direction: Matrix


def muon_step(
    weight: Matrix,
    gradient: Matrix,
    previous_momentum: Matrix,
    learning_rate: float = 1e-3,
    decay: float = 0.1,
    momentum: float = 0.95,
    nesterov: bool = True,
    base: float = 0.2,
) -> MuonStep:
    """Execute one atomic Muon step for a matrix parameter."""

    if shape(weight) != shape(gradient) or shape(weight) != shape(previous_momentum):
        raise ValueError("weight, gradient, and momentum shapes must agree")
    buffer, candidate = momentum_candidate(
        previous_momentum, gradient, momentum, nesterov
    )
    rows, columns = shape(weight)
    finite_polar = newton_schulz_zeroth_power(candidate)
    direction = scale(finite_polar, shape_scale(rows, columns, base))
    decayed = decoupled_weight_decay(weight, learning_rate, decay)
    updated = add(decayed, scale(direction, -learning_rate))
    return MuonStep(updated, buffer, candidate, direction)


@dataclass(frozen=True)
class AdamWFlatState:
    """Store flat first and second moments plus the completed step count."""

    first_moment: FlatTensor
    second_moment: FlatTensor
    step: int


@dataclass(frozen=True)
class AdamWFlatStep:
    """Capture an updated flat weight, optimizer state, and update direction."""

    weight: FlatTensor
    state: AdamWFlatState
    direction: FlatTensor


def adamw_step_flat(
    weight: Sequence[float],
    gradient: Sequence[float],
    state: AdamWFlatState,
    learning_rate: float = 1e-3,
    decay: float = 0.1,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamWFlatStep:
    """Execute one bias-corrected, decoupled AdamW step on a flat tensor."""

    count = len(weight)
    if count == 0 or len(gradient) != count:
        raise ValueError("weight and gradient must have the same positive size")
    if len(state.first_moment) != count or len(state.second_moment) != count:
        raise ValueError("AdamW moment sizes must match the parameter")
    if state.step < 0:
        raise ValueError("AdamW step must be nonnegative")
    if learning_rate < 0 or decay < 0 or epsilon <= 0:
        raise ValueError("invalid AdamW learning rate, decay, or epsilon")
    if not 0 <= beta_1 < 1 or not 0 <= beta_2 < 1:
        raise ValueError("AdamW beta coefficients must lie in [0,1)")

    first = tuple(
        beta_1 * old + (1.0 - beta_1) * float(g)
        for old, g in zip(state.first_moment, gradient, strict=True)
    )
    second = tuple(
        beta_2 * old + (1.0 - beta_2) * float(g) ** 2
        for old, g in zip(state.second_moment, gradient, strict=True)
    )
    step = state.step + 1
    first_correction = 1.0 - beta_1**step
    second_correction = 1.0 - beta_2**step
    direction = tuple(
        (m / first_correction) / (math.sqrt(v / second_correction) + epsilon)
        for m, v in zip(first, second, strict=True)
    )
    shrink = 1.0 - learning_rate * decay
    updated = tuple(
        shrink * float(value) - learning_rate * update
        for value, update in zip(weight, direction, strict=True)
    )
    return AdamWFlatStep(updated, AdamWFlatState(first, second, step), direction)


MUON_ROLES = frozenset({"attention_weight", "mlp_weight", "router_weight"})
ADAMW_ROLES = frozenset({"embedding", "output_head", "normalization", "bias"})


def optimizer_for(role: str, dimensions: int) -> str:
    """Resolve optimizer eligibility by semantic role, then matrix rank."""

    if role in ADAMW_ROLES:
        return "AdamW"
    if role in MUON_ROLES and dimensions >= 2:
        return "Muon"
    return "AdamW"


def partition_parameters(
    parameters: Mapping[str, tuple[str, tuple[int, ...]]],
) -> dict[str, tuple[str, ...]]:
    """Partition parameters by their optimizer family."""
    buckets: dict[str, list[str]] = {"Muon": [], "AdamW": []}
    for name, (role, dimensions) in parameters.items():
        buckets[optimizer_for(role, len(dimensions))].append(name)
    return {key: tuple(value) for key, value in buckets.items()}


def global_mean_gradient(
    local_means: Sequence[Matrix], normalization_masses: Sequence[float]
) -> Matrix:
    """Combine rank-local mean gradients for one global-mean objective."""

    if not local_means or len(local_means) != len(normalization_masses):
        raise ValueError("each local mean needs one normalization mass")
    dimensions = shape(local_means[0])
    if any(shape(gradient) != dimensions for gradient in local_means):
        raise ValueError("local gradient shapes must agree")
    if any(not math.isfinite(mass) or mass <= 0 for mass in normalization_masses):
        raise ValueError("normalization masses must be finite and positive")
    total_mass = sum(normalization_masses)
    numerator = zeros(*dimensions)
    for gradient, mass in zip(local_means, normalization_masses, strict=True):
        numerator = add(numerator, scale(gradient, mass))
    return scale(numerator, 1.0 / total_mass)


@dataclass(frozen=True)
class StateLedger:
    """Account for Muon momentum and AdamW optimizer-state storage."""

    muon_momentum_elements: int
    adamw_moment_elements: int
    adamw_step_counters: int
    total_elements: int
    fp32_bytes: int


def state_ledger(parameters: Mapping[str, tuple[str, tuple[int, ...]]]) -> StateLedger:
    """Count optimizer-state elements and FP32 bytes by parameter role."""
    partition = partition_parameters(parameters)
    counts = {
        name: math.prod(dimensions) for name, (_, dimensions) in parameters.items()
    }
    muon = sum(counts[name] for name in partition["Muon"])
    adam = 2 * sum(counts[name] for name in partition["AdamW"])
    counters = len(partition["AdamW"])
    total = muon + adam + counters
    return StateLedger(muon, adam, counters, total, 4 * total)


def slice_rows(matrix: Matrix, start: int, stop: int) -> Matrix:
    """Return a contiguous row slice of a matrix."""
    if not 0 <= start < stop <= len(matrix):
        raise ValueError("invalid row slice")
    return matrix[start:stop]


def concatenate_rows(shards: Sequence[Matrix]) -> Matrix:
    """Concatenate matrix shards along their row dimension."""
    if not shards:
        raise ValueError("at least one shard is required")
    width = shape(shards[0])[1]
    if any(shape(shard)[1] != width for shard in shards):
        raise ValueError("shard widths must agree")
    return tuple(row for shard in shards for row in shard)


def gathered_muon_slices(
    candidate_shards: Sequence[Matrix], base: float = 0.2
) -> tuple[Matrix, ...]:
    """Gather, transform once, then return the corresponding row slices."""

    full = concatenate_rows(candidate_shards)
    rows, columns = shape(full)
    transformed = scale(
        newton_schulz_zeroth_power(full), shape_scale(rows, columns, base)
    )
    slices: list[Matrix] = []
    offset = 0
    for shard in candidate_shards:
        shard_rows = shape(shard)[0]
        slices.append(slice_rows(transformed, offset, offset + shard_rows))
        offset += shard_rows
    return tuple(slices)


def independently_transformed_shards(
    candidate_shards: Sequence[Matrix], base: float = 0.2
) -> tuple[Matrix, ...]:
    """Incorrect comparison path: transform each row shard as a separate matrix."""

    return tuple(
        scale(
            newton_schulz_zeroth_power(shard),
            shape_scale(*shape(shard), base),
        )
        for shard in candidate_shards
    )


def compute_for_target_loss(coefficient: float, exponent: float, loss: float) -> float:
    """Invert L(C)=coefficient*C**(-exponent) for positive inputs."""

    if coefficient <= 0 or exponent <= 0 or loss <= 0:
        raise ValueError("scaling-law terms must be positive")
    return (coefficient / loss) ** (1.0 / exponent)


def moonlight_compute_ratio(loss: float) -> float:
    """Return Muon's required compute divided by AdamW's at a target loss."""
    muon_compute = compute_for_target_loss(2.506, 0.052, loss)
    adamw_compute = compute_for_target_loss(2.608, 0.054, loss)
    return muon_compute / adamw_compute


def normalized_spectral_entropy(singular_values: Sequence[float]) -> float:
    """Entropy of normalized squared singular values, divided by log(n)."""

    values = tuple(abs(float(value)) for value in singular_values)
    squared = tuple(value * value for value in values)
    if len(values) < 2 or sum(squared) == 0:
        return 0.0
    total = sum(squared)
    probabilities = tuple(value / total for value in squared if value > 0)
    return -sum(p * math.log(p) for p in probabilities) / math.log(len(values))
