"""Accounting and balancing helpers for the Chapter 3 CPU reference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import ceil, exp, log1p
from typing import Protocol


Vector = tuple[float, ...]
ScoreMatrix = tuple[Vector, ...]

QUANTILE_DEMO_SCORES: ScoreMatrix = (
    (0.2923, 0.7246, 0.4377, 0.7147),
    (0.3713, 0.6007, 0.5669, 0.2321),
    (0.8944, 0.5628, 0.7428, 0.4281),
    (0.6068, 0.5123, 0.0010, 0.4612),
    (0.6489, 0.8740, 0.1396, 0.1316),
    (0.7380, 0.5989, 0.7207, 0.0933),
    (0.7212, 0.0150, 0.9917, 0.4841),
    (0.9814, 0.3441, 0.6390, 0.5140),
)

PRODUCTION_COMPARISON: dict[str, dict[str, int | float | None]] = {
    "K2": {
        "total_parameters_trillion": 1.04,
        "activated_parameters_billion": 32.6,
        "model_width": 7168,
        "latent_width": None,
        "expert_hidden_width": 2048,
        "routed_experts": 384,
        "selected_routed_experts": 8,
        "shared_experts": 1,
        "initial_dense_layers": 1,
    },
    "K3": {
        "total_parameters_trillion": 2.78,
        "activated_parameters_billion": 104.2,
        "model_width": 7168,
        "latent_width": 3584,
        "expert_hidden_width": 3072,
        "routed_experts": 896,
        "selected_routed_experts": 16,
        "shared_experts": 2,
        "initial_dense_layers": 1,
    },
}


class RoutingDecision(Protocol):
    """Fields required by the controlled latent-shape oracle."""

    selected: tuple[int, ...]
    weights: Vector


class RouteIdentity(Protocol):
    """Identity fields required by the placement ledger."""

    origin: str
    expert: int


class RouteRecord(Protocol):
    """Record field required by the placement ledger."""

    identity: RouteIdentity


def _validate_parallel_vectors(*vectors: Sequence[float]) -> int:
    """Return common nonzero width or reject inconsistent vectors."""

    if not vectors or not vectors[0]:
        raise ValueError("vectors must be nonempty")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("vectors must have equal width")
    return width


def _log_sigmoid(value: float) -> float:
    """Return the sigmoid score in log space without overflow or underflow."""

    if value >= 0.0:
        return -log1p(exp(-value))
    return value - log1p(exp(value))


def _topk_indices(
    scores: Sequence[float],
    biases: Sequence[float],
    k: int,
    tie_breakers: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Rank biased scores with an optional stable secondary ordering."""

    width = _validate_parallel_vectors(scores, biases)
    if not 1 <= k <= width:
        raise ValueError("k must satisfy 1 <= k <= number of experts")
    secondary = (0.0,) * width if tie_breakers is None else tie_breakers
    _validate_parallel_vectors(scores, secondary)
    return tuple(
        sorted(
            range(width),
            key=lambda index: (
                -(scores[index] + biases[index]),
                -secondary[index],
                index,
            ),
        )[:k]
    )


def capacity_ledger(
    loads: Sequence[int], capacity_factor: float
) -> dict[str, int | float]:
    """Evaluate the chapter's named static route-capacity policy."""

    if not loads or any(load < 0 for load in loads):
        raise ValueError("loads must be nonnegative and nonempty")
    if capacity_factor <= 0.0:
        raise ValueError("capacity factor must be positive")
    ideal = sum(loads) / len(loads)
    capacity = ceil(capacity_factor * ideal)
    overflow = sum(max(0, load - capacity) for load in loads)
    padding = sum(max(0, capacity - load) for load in loads)
    return {
        "capacity_factor": capacity_factor,
        "ideal_load": ideal,
        "capacity": capacity,
        "allocated_slots": capacity * len(loads),
        "overflow_routes": overflow,
        "padding_slots": padding,
    }


def loss_free_bias_update(
    biases: Sequence[float], loads: Sequence[int], step_size: float
) -> Vector:
    """Apply the fixed-step selection-bias update."""

    _validate_parallel_vectors(biases, loads)
    if step_size <= 0.0:
        raise ValueError("step size must be positive")
    target = sum(loads) / len(loads)
    directions = tuple(
        1.0 if load < target else -1.0 if load > target else 0.0 for load in loads
    )
    return tuple(
        bias + step_size * direction
        for bias, direction in zip(biases, directions, strict=True)
    )


def score_loads(
    scores: ScoreMatrix, biases: Sequence[float], k: int
) -> tuple[int, ...]:
    """Count selected experts for a matrix of already-computed scores."""

    if not scores:
        raise ValueError("score matrix must be nonempty")
    width = _validate_parallel_vectors(scores[0], biases)
    if any(len(row) != width for row in scores):
        raise ValueError("score matrix must be rectangular")
    selected = [_topk_indices(row, biases, k) for row in scores]
    return tuple(sum(expert in row for row in selected) for expert in range(width))


def quantile_bias_update(
    scores: ScoreMatrix, biases: Sequence[float], k: int
) -> Vector:
    """Apply the no-tie, exact-order-statistic QB reference update."""

    if not scores:
        raise ValueError("score matrix must be nonempty")
    expert_count = _validate_parallel_vectors(scores[0], biases)
    if any(len(row) != expert_count for row in scores):
        raise ValueError("score matrix must be rectangular")
    if not 1 <= k < expert_count:
        raise ValueError("QB reference requires 1 <= k < expert count")
    token_count = len(scores)
    route_count = token_count * k
    if route_count % expert_count:
        raise ValueError("exact target load must be integral")
    target = route_count // expert_count
    cutoffs: list[float] = []
    for row in scores:
        ranked = sorted(
            (row[index] + biases[index] for index in range(expert_count)),
            reverse=True,
        )
        cutoffs.append(ranked[k])
    candidates: list[float] = []
    for expert in range(expert_count):
        margins = sorted(
            (scores[token][expert] - cutoffs[token] for token in range(token_count)),
            reverse=True,
        )
        candidates.append(-margins[target])
    mean_candidate = sum(candidates) / expert_count
    return tuple(candidate - mean_candidate for candidate in candidates)


def balancing_demo() -> dict[str, object]:
    """Return fixed-step and synthetic exact-quantile arithmetic."""

    zero_biases = (0.0, 0.0, 0.0, 0.0)
    fixed = loss_free_bias_update(zero_biases, (7, 6, 4, 3), 0.1)
    initial_loads = score_loads(QUANTILE_DEMO_SCORES, zero_biases, 1)
    quantile_bias = quantile_bias_update(QUANTILE_DEMO_SCORES, zero_biases, 1)
    final_loads = score_loads(QUANTILE_DEMO_SCORES, quantile_bias, 1)
    return {
        "fixed_step_bias": fixed,
        "fixed_step_directions": tuple(
            1 if value > 0.0 else -1 if value < 0.0 else 0 for value in fixed
        ),
        "quantile_initial_loads": initial_loads,
        "quantile_bias": quantile_bias,
        "quantile_final_loads": final_loads,
    }


def payload_ledger(
    token_count: int, k: int, model_width: int, latent_width: int
) -> dict[str, int | float]:
    """Return full-width and latent logical route payloads."""

    if min(token_count, k, model_width, latent_width) <= 0:
        raise ValueError("payload dimensions must be positive")
    if latent_width > model_width:
        raise ValueError("latent width must not exceed model width")
    full = token_count * k * model_width
    latent = token_count * k * latent_width
    return {
        "route_records": token_count * k,
        "full_width_one_way_scalars": full,
        "latent_one_way_scalars": latent,
        "full_to_latent_ratio": full / latent,
    }


def _project_down(payload: Sequence[float], latent_width: int) -> Vector:
    """Select the first latent coordinates for a controlled shape oracle."""

    if not 1 <= latent_width <= len(payload):
        raise ValueError("invalid latent width")
    return tuple(payload[:latent_width])


def _project_up(payload: Sequence[float], model_width: int) -> Vector:
    """Pad controlled latent coordinates back to model width."""

    if not 1 <= len(payload) <= model_width:
        raise ValueError("invalid model width")
    return tuple([*payload, *([0.0] * (model_width - len(payload)))])


def standard_latent_shape_check(
    origins: Sequence[str],
    payloads: Mapping[str, Vector],
    decisions: Mapping[str, RoutingDecision],
    expert: Callable[[int, Sequence[float]], Vector],
    max_difference: Callable[[Sequence[float], Sequence[float]], float],
) -> dict[str, object]:
    """Check the controlled full-width and latent interfaces."""

    model_width = 8
    latent_width = 4
    maximum_error = 0.0
    for origin in origins:
        latent_input = _project_down(payloads[origin], latent_width)
        decision = decisions[origin]
        latent_experts = [expert(index, latent_input) for index in decision.selected]
        latent_mixture = tuple(
            sum(
                decision.weights[route] * latent_experts[route][coordinate]
                for route in range(len(latent_experts))
            )
            for coordinate in range(latent_width)
        )
        latent_output = _project_up(latent_mixture, model_width)
        full_experts = [_project_up(output, model_width) for output in latent_experts]
        controlled_full = tuple(
            sum(
                decision.weights[route] * full_experts[route][coordinate]
                for route in range(len(full_experts))
            )
            for coordinate in range(model_width)
        )
        maximum_error = max(
            maximum_error,
            max_difference(latent_output, controlled_full),
        )
    return {
        "model_width": model_width,
        "latent_width": latent_width,
        "output_shape": [len(origins), model_width],
        "controlled_max_abs_error": maximum_error,
        "tolerance": 1e-12,
        "normalization": "identity for controlled equivalence only",
        "payload": payload_ledger(len(origins), 2, model_width, latent_width),
    }


def placement_ledger(records: Sequence[RouteRecord]) -> dict[str, dict[str, object]]:
    """Evaluate the chapter's two-rank placement counterexample."""

    owners = {
        **{origin: 0 for origin in ("v1", "v2", "v3", "v4", "x1")},
        **{origin: 1 for origin in ("x2", "x3", "x4", "x5", "x6")},
    }
    placements = {
        "A": {1: 0, 2: 0, 3: 1, 4: 1},
        "B": {1: 0, 2: 1, 3: 0, 4: 1},
    }
    result: dict[str, dict[str, object]] = {}
    for name, placement in placements.items():
        received = [0, 0]
        crossing: list[str] = []
        for record in records:
            destination = placement[record.identity.expert]
            received[destination] += 1
            if owners[record.identity.origin] != destination:
                crossing.append(f"{record.identity.origin}->E{record.identity.expert}")
        result[name] = {
            "rank_received_loads": received,
            "crossing_dispatch_routes": len(crossing),
            "crossing_identities": crossing,
        }
    return result


def production_comparison() -> dict[str, object]:
    """Return frozen reported values plus transparent deductions."""

    k2 = PRODUCTION_COMPARISON["K2"]
    k3 = PRODUCTION_COMPARISON["K3"]
    k2_experts = int(k2["routed_experts"] or 0)
    k2_selected = int(k2["selected_routed_experts"] or 0)
    k3_experts = int(k3["routed_experts"] or 0)
    k3_selected = int(k3["selected_routed_experts"] or 0)
    model_width = int(k3["model_width"] or 0)
    latent_width = int(k3["latent_width"] or 0)
    return {
        "reported": PRODUCTION_COMPARISON,
        "deduced": {
            "K2_routed_sparsity": k2_experts / k2_selected,
            "K3_routed_sparsity": k3_experts / k3_selected,
            "expert_pool_ratio_K3_over_K2": k3_experts / k2_experts,
            "active_routed_ratio_K3_over_K2": k3_selected / k2_selected,
            "full_to_latent_width_ratio": model_width / latent_width,
        },
        "claim_boundary": (
            "Reported configuration values and arithmetic deductions only; "
            "no causal quality or performance claim."
        ),
    }
