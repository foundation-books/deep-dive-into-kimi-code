"""Dependency-free reference contracts for Chapter 3.

The routines in this module validate a scaled-down sparse-MoE trace. They are
not a production K2 or K3 implementation, a training experiment, or a latency
model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from math import exp, isfinite
from typing import Sequence, cast

from .reference_ledgers import (
    PRODUCTION_COMPARISON as PRODUCTION_COMPARISON,
    QUANTILE_DEMO_SCORES as QUANTILE_DEMO_SCORES,
    RouteRecord as LedgerRouteRecord,
    RoutingDecision,
    _log_sigmoid,
    _topk_indices,
    _validate_parallel_vectors,
    balancing_demo as balancing_demo,
    capacity_ledger as capacity_ledger,
    loss_free_bias_update as loss_free_bias_update,
    payload_ledger as payload_ledger,
    placement_ledger as build_placement_ledger,
    production_comparison as production_comparison,
    quantile_bias_update as quantile_bias_update,
    score_loads as score_loads,
    standard_latent_shape_check as check_standard_latent_shape,
)
from .reference_fixture import (
    ORIGINS as ORIGINS,
    RUNNING_BIASES as RUNNING_BIASES,
    RUNNING_LOGITS as RUNNING_LOGITS,
    Vector as Vector,
)
from .reference_types import RouteIdentity, RouteRecord, RouterResult


def sigmoid(value: float) -> float:
    """Evaluate the logistic sigmoid without overflow."""

    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    scaled = exp(value)
    return scaled / (1.0 + scaled)


def dense_ffn(
    payload: Sequence[float],
    expansion_weights: Sequence[Sequence[float]],
    contraction_weights: Sequence[Sequence[float]],
) -> Vector:
    """Evaluate a bias-free ReLU FFN with a typed ``d -> h -> d`` interface.

    Explicit matrices keep this baseline independent of the routed reference.
    It is a shape and arithmetic oracle, not a model-checkpoint layer.
    """

    if not payload:
        raise ValueError("payload must be nonempty")
    model_width = len(payload)
    if not expansion_weights:
        raise ValueError("expansion weights must have positive hidden width")
    if any(len(row) != model_width for row in expansion_weights):
        raise ValueError("every expansion row must match the model width")
    hidden_width = len(expansion_weights)
    if len(contraction_weights) != model_width:
        raise ValueError("contraction weights must return to the model width")
    if any(len(row) != hidden_width for row in contraction_weights):
        raise ValueError("every contraction row must match the hidden width")

    hidden = tuple(
        max(
            0.0, sum(weight * value for weight, value in zip(row, payload, strict=True))
        )
        for row in expansion_weights
    )
    return tuple(
        sum(weight * value for weight, value in zip(row, hidden, strict=True))
        for row in contraction_weights
    )


def route(logits: Sequence[float], biases: Sequence[float], k: int) -> RouterResult:
    """Apply sigmoid scoring, biased membership, and unbiased normalization."""

    _validate_parallel_vectors(logits, biases)
    if any(not isfinite(value) for value in (*logits, *biases)):
        raise ValueError("router logits and biases must be finite")
    scores = tuple(sigmoid(value) for value in logits)
    log_scores = tuple(_log_sigmoid(value) for value in logits)
    biased = tuple(score + bias for score, bias in zip(scores, biases, strict=True))
    selected_zero_based = _topk_indices(scores, biases, k, log_scores)
    maximum = max(log_scores[index] for index in selected_zero_based)
    numerators = tuple(
        exp(log_scores[index] - maximum) for index in selected_zero_based
    )
    denominator = sum(numerators)
    selected = tuple(index + 1 for index in selected_zero_based)
    weights = tuple(value / denominator for value in numerators)
    return RouterResult(scores, biased, selected, weights)


def incorrect_biased_weights(result: RouterResult, biases: Sequence[float]) -> Vector:
    """Return the deliberately incorrect selected biased-score normalization."""

    if len(result.scores) != len(biases):
        raise ValueError("bias width does not match routing result")
    numerators = tuple(
        result.scores[expert - 1] + biases[expert - 1] for expert in result.selected
    )
    denominator = sum(numerators)
    return tuple(value / denominator for value in numerators)


def running_decisions() -> dict[str, RouterResult]:
    """Route every origin in the ten-token running example."""

    return {
        origin: route(RUNNING_LOGITS[origin], RUNNING_BIASES, 2) for origin in ORIGINS
    }


def incidence_matrix(
    decisions: dict[str, RouterResult] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Construct the running route-incidence matrix."""

    routed = decisions or running_decisions()
    expert_count = len(RUNNING_BIASES)
    return tuple(
        tuple(
            1 if expert in routed[origin].selected else 0
            for expert in range(1, expert_count + 1)
        )
        for origin in ORIGINS
    )


def expert_loads(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Sum a rectangular incidence matrix by expert column."""

    if not matrix or not matrix[0]:
        raise ValueError("incidence matrix must be nonempty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("incidence matrix must be rectangular")
    return tuple(sum(row[column] for row in matrix) for column in range(width))


def running_token_payloads(width: int = 8) -> dict[str, Vector]:
    """Return deterministic model-width payloads for the route oracle."""

    if width <= 0:
        raise ValueError("payload width must be positive")
    return {
        origin: tuple(
            (origin_index + 1) * (coordinate + 1) / 100.0 for coordinate in range(width)
        )
        for origin_index, origin in enumerate(ORIGINS)
    }


def build_route_records(
    payloads: dict[str, Vector] | None = None,
    decisions: dict[str, RouterResult] | None = None,
) -> tuple[RouteRecord, ...]:
    """Expand token-major decisions into individually tagged records."""

    tokens = payloads or running_token_payloads()
    routed = decisions or running_decisions()
    if set(tokens) != set(ORIGINS):
        raise ValueError("payload map must contain exactly the running origins")
    records: list[RouteRecord] = []
    for origin in ORIGINS:
        decision = routed[origin]
        for ordinal, (expert, weight) in enumerate(
            zip(decision.selected, decision.weights, strict=True), start=1
        ):
            records.append(
                RouteRecord(
                    RouteIdentity(origin, expert, ordinal),
                    weight,
                    tokens[origin],
                )
            )
    return tuple(records)


def dispatch(records: Sequence[RouteRecord]) -> tuple[RouteRecord, ...]:
    """Stably permute route records into expert-major order."""

    return tuple(sorted(records, key=lambda record: record.identity.expert))


def expert_output(expert: int, payload: Sequence[float]) -> Vector:
    """Apply a deterministic, shape-preserving toy expert."""

    if expert <= 0 or not payload:
        raise ValueError("expert and payload must be nonempty")
    scale = 1.0 + 0.1 * expert
    return tuple(
        scale * value + 0.01 * expert * (coordinate + 1)
        for coordinate, value in enumerate(payload)
    )


def shared_output(payload: Sequence[float]) -> Vector:
    """Apply the one always-active toy shared expert."""

    if not payload:
        raise ValueError("shared payload must be nonempty")
    return tuple(
        0.2 * value + 0.005 * (coordinate + 1)
        for coordinate, value in enumerate(payload)
    )


def execute(records: Sequence[RouteRecord]) -> tuple[RouteRecord, ...]:
    """Execute every dispatched record without changing identity or weight."""

    return tuple(
        replace(
            record,
            payload=expert_output(record.identity.expert, record.payload),
        )
        for record in records
    )


def validate_returns(
    dispatched: Sequence[RouteRecord],
    returned: Sequence[RouteRecord],
    tolerance: float = 1e-12,
) -> tuple[str, ...]:
    """Return route-integrity errors without trusting record order."""

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("route-weight tolerance must be finite and nonnegative")
    dispatched_counts = Counter(record.identity for record in dispatched)
    returned_counts = Counter(record.identity for record in returned)
    errors: list[str] = []
    if any(count != 1 for count in dispatched_counts.values()):
        errors.append("dispatched identities are not unique")
    if any(count != 1 for count in returned_counts.values()):
        errors.append("returned identities are not unique")
    missing = dispatched_counts - returned_counts
    unexpected = returned_counts - dispatched_counts
    if missing:
        errors.append(f"missing identities: {sorted(missing.elements())}")
    if unexpected:
        errors.append(f"unexpected identities: {sorted(unexpected.elements())}")
    expected_weights = {record.identity: record.weight for record in dispatched}
    for record in returned:
        expected = expected_weights.get(record.identity)
        if expected is None:
            continue
        if not isfinite(expected) or not isfinite(record.weight):
            errors.append(f"non-finite weight for {record.identity}")
        elif abs(record.weight - expected) > tolerance:
            errors.append(f"wrong weight for {record.identity}")
    return tuple(errors)


def combine(
    payloads: dict[str, Vector],
    dispatched: Sequence[RouteRecord],
    returned: Sequence[RouteRecord],
) -> dict[str, Vector]:
    """Validate, weight, reduce by origin, and add one shared contribution."""

    errors = validate_returns(dispatched, returned)
    if errors:
        raise ValueError("; ".join(errors))
    widths = {len(payload) for payload in payloads.values()}
    if len(widths) != 1:
        raise ValueError("token payloads must have one common width")
    width = widths.pop()
    routed = {origin: [0.0] * width for origin in ORIGINS}
    for record in returned:
        accumulator = routed[record.identity.origin]
        if len(record.payload) != width:
            raise ValueError("expert output changed external width")
        for index, value in enumerate(record.payload):
            accumulator[index] += record.weight * value
    return {
        origin: tuple(
            routed_value + shared_value
            for routed_value, shared_value in zip(
                routed[origin], shared_output(payloads[origin]), strict=True
            )
        )
        for origin in ORIGINS
    }


def token_major_reference(
    payloads: dict[str, Vector] | None = None,
    decisions: dict[str, RouterResult] | None = None,
) -> dict[str, Vector]:
    """Evaluate the toy sparse layer directly in token-major order."""

    tokens = payloads or running_token_payloads()
    routed = decisions or running_decisions()
    outputs: dict[str, Vector] = {}
    for origin in ORIGINS:
        decision = routed[origin]
        contributions = [
            expert_output(expert, tokens[origin]) for expert in decision.selected
        ]
        weighted = tuple(
            sum(
                decision.weights[route_index] * contributions[route_index][coordinate]
                for route_index in range(len(contributions))
            )
            for coordinate in range(len(tokens[origin]))
        )
        outputs[origin] = tuple(
            left + right
            for left, right in zip(weighted, shared_output(tokens[origin]), strict=True)
        )
    return outputs


def max_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Return maximum coordinate difference for equal-width vectors."""

    _validate_parallel_vectors(left, right)
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def inject_fault(
    returned: Sequence[RouteRecord], fault: str
) -> tuple[RouteRecord, ...]:
    """Create one controlled identity or weight failure."""

    records = list(returned)
    if not records:
        raise ValueError("cannot inject a fault into an empty route list")
    if fault == "drop":
        return tuple(records[:-1])
    if fault == "duplicate":
        return tuple([*records, records[0]])
    if fault == "wrong_expert":
        identity = records[0].identity
        new_expert = 2 if identity.expert != 2 else 3
        records[0] = replace(
            records[0],
            identity=replace(identity, expert=new_expert),
        )
        return tuple(records)
    if fault == "wrong_origin":
        records[0] = replace(
            records[0],
            identity=replace(records[0].identity, origin="corrupt-origin"),
        )
        return tuple(records)
    if fault == "wrong_weight":
        records[0] = replace(records[0], weight=records[0].weight + 0.125)
        return tuple(records)
    raise ValueError(f"unknown route fault: {fault}")


def fault_detection_summary() -> dict[str, bool]:
    """Report whether every declared fault is rejected by the record oracle."""

    dispatched = dispatch(build_route_records())
    returned = execute(dispatched)
    return {
        fault: bool(validate_returns(dispatched, inject_fault(returned, fault)))
        for fault in (
            "drop",
            "duplicate",
            "wrong_expert",
            "wrong_origin",
            "wrong_weight",
        )
    }


def standard_latent_shape_check() -> dict[str, object]:
    """Check controlled full/latent interfaces and logical payload counts.

    The numerical equality uses an identity normalization and a full-width
    reference deliberately defined as the same down/expert/up composition. It
    does not claim equality to K3's RMS-normalized production layer.
    """

    return check_standard_latent_shape(
        ORIGINS,
        running_token_payloads(8),
        cast(Mapping[str, RoutingDecision], running_decisions()),
        expert_output,
        max_abs_difference,
    )


def placement_ledger() -> dict[str, dict[str, object]]:
    """Evaluate the chapter's two-rank placement counterexample."""

    return build_placement_ledger(
        cast(Sequence[LedgerRouteRecord], build_route_records())
    )
