"""Quantile Balancing, histogram, and fault-detection witnesses."""

from __future__ import annotations

from math import isclose
from typing import Sequence

from .reference_core import (
    forward_token,
    matvec,
    rms_norm,
    route_from_scores,
    stable_latent_moe_forward,
)
from .reference_fixture import QB_SCORES, RUNNING_BIASES
from .reference_histogram import (
    add_histograms as add_histograms,
    exact_histogram_targets as exact_histogram_targets,
    histogram_counts as histogram_counts,
    histogram_quantile_estimate as histogram_quantile_estimate,
    partitioned_histogram_estimate as partitioned_histogram_estimate,
    recover_histogram_quantile as recover_histogram_quantile,
    required_bias_samples as required_bias_samples,
    token_cutoffs as token_cutoffs,
)
from .reference_proofs import (
    local_quantile_average_counterexample,
    situ_coordinate_bound,
)
from .reference_types import (
    ForwardTrace,
    RUNNING_DIMS,
    ScoreMatrix,
)


def max_forward_error(left: ForwardTrace, right: ForwardTrace) -> float:
    """Return the largest error across every numeric forward checkpoint."""

    errors: list[float] = []
    for first, second in zip(left.tokens, right.tokens, strict=True):
        for left_vector, right_vector in (
            (first.latent_input, second.latent_input),
            (first.routed_aggregate, second.routed_aggregate),
            (first.normalized_aggregate, second.normalized_aggregate),
            (first.routed_output, second.routed_output),
            (first.shared_output, second.shared_output),
            (first.output, second.output),
        ):
            errors.extend(
                abs(a - b) for a, b in zip(left_vector, right_vector, strict=True)
            )
    return max(errors, default=0.0)


def selected_sets(
    scores: ScoreMatrix = QB_SCORES,
    biases: Sequence[float] = RUNNING_BIASES,
    selected_count: int = RUNNING_DIMS.selected_experts,
) -> tuple[tuple[int, ...], ...]:
    """Return ordered selected experts for an already-scored matrix."""

    return tuple(
        route_from_scores(row, biases, selected_count).selected for row in scores
    )


def expert_loads(
    routes: Sequence[Sequence[int]], expert_count: int = 4
) -> tuple[int, ...]:
    """Count the selected-set membership of every expert."""

    if not routes:
        raise ValueError("routes must be nonempty")
    return tuple(
        sum(expert in route for route in routes)
        for expert in range(1, expert_count + 1)
    )


def exact_qb_update(
    scores: ScoreMatrix = QB_SCORES,
    biases: Sequence[float] = RUNNING_BIASES,
    selected_count: int = RUNNING_DIMS.selected_experts,
) -> dict[str, object]:
    """Apply one exact source-convention QB expert-coordinate update.

    Appendix C Algorithm 1 takes the zero-based ``q`` entry of a descending
    margin sort, i.e. the (q+1)-th largest margin.  The returned bias is the
    negative expert threshold and is mean-centered before the next step.
    """

    tokens = len(scores)
    experts = len(biases)
    routes = tokens * selected_count
    if routes % experts:
        raise ValueError("QB exact target load must be integral")
    target = routes // experts
    alpha = token_cutoffs(scores, biases, selected_count)
    beta: list[float] = []
    margin_columns: list[tuple[float, ...]] = []
    for expert in range(experts):
        margins = tuple(
            sorted(
                (scores[token][expert] - alpha[token] for token in range(tokens)),
                reverse=True,
            )
        )
        margin_columns.append(margins)
        beta.append(margins[target])
    uncentered = tuple(-value for value in beta)
    mean = sum(uncentered) / experts
    centered = tuple(value - mean for value in uncentered)
    # Compare each stored margin with its own selected endpoint.  Re-forming
    # s + b > alpha changes rounding at equality, and decimal fixture entries
    # such as 0.85 - 0.24 and 0.82 - 0.21 can differ by one float ulp.
    frozen_counts = tuple(
        sum(
            margin > beta[expert]
            and not isclose(margin, beta[expert], rel_tol=0.0, abs_tol=1e-12)
            for margin in margin_columns[expert]
        )
        for expert in range(experts)
    )
    equality_counts = tuple(
        sum(
            isclose(margin, beta[expert], rel_tol=0.0, abs_tol=1e-12)
            for margin in margin_columns[expert]
        )
        for expert in range(experts)
    )
    current_routes = selected_sets(scores, biases, selected_count)
    next_routes = selected_sets(scores, centered, selected_count)
    return {
        "target_load": target,
        "alpha": alpha,
        "beta": tuple(beta),
        "uncentered_bias": uncentered,
        "centered_bias": centered,
        "centered_mean": sum(centered) / experts,
        "current_loads": expert_loads(current_routes, experts),
        "frozen_cutoff_counts": frozen_counts,
        "endpoint_equality_counts": equality_counts,
        "recomputed_loads": expert_loads(next_routes, experts),
        "current_routes": current_routes,
        "next_routes": next_routes,
    }


def interior_qb_coordinate_witness(
    scores: ScoreMatrix = QB_SCORES,
    biases: Sequence[float] = RUNNING_BIASES,
    selected_count: int = RUNNING_DIMS.selected_experts,
) -> dict[str, object]:
    """Choose expert-coordinate midpoints with the endpoint token cutoffs.

    Midpoints remove endpoint equality only when the minimizer interval is
    nondegenerate.  The ten-token fixture has collapsed intervals, so this
    witness does not promise five strict acceptances in every expert column.
    """

    tokens = len(scores)
    experts = len(biases)
    routes = tokens * selected_count
    if routes % experts:
        raise ValueError("QB exact target load must be integral")
    target = routes // experts
    alpha = token_cutoffs(scores, biases, selected_count)
    beta: list[float] = []
    margin_columns: list[tuple[float, ...]] = []
    for expert in range(experts):
        margins = tuple(
            sorted(
                (scores[token][expert] - alpha[token] for token in range(tokens)),
                reverse=True,
            )
        )
        margin_columns.append(margins)
        beta.append((margins[target - 1] + margins[target]) / 2.0)
    uncentered = tuple(-value for value in beta)
    mean = sum(uncentered) / experts
    centered = tuple(value - mean for value in uncentered)
    frozen_counts = tuple(
        sum(
            margin > beta[expert]
            and not isclose(margin, beta[expert], rel_tol=0.0, abs_tol=1e-12)
            for margin in margin_columns[expert]
        )
        for expert in range(experts)
    )
    equality_counts = tuple(
        sum(
            isclose(margin, beta[expert], rel_tol=0.0, abs_tol=1e-12)
            for margin in margin_columns[expert]
        )
        for expert in range(experts)
    )
    return {
        "target_load": target,
        "alpha": alpha,
        "beta": tuple(beta),
        "uncentered_bias": uncentered,
        "centered_bias": centered,
        "centered_mean": sum(centered) / experts,
        "frozen_cutoff_counts": frozen_counts,
        "midpoint_equality_counts": equality_counts,
        "claim_boundary": (
            "expert midpoint coordinate-minimizer witness with the source endpoint "
            "token cutoffs frozen; collapsed intervals retain equality cases and "
            "one step does not guarantee balanced recomputed loads"
        ),
    }


def fault_detection_summary() -> dict[str, bool]:
    """Exercise the smallest oracle for six common semantic corruptions."""

    trace = stable_latent_moe_forward()
    x3 = trace.tokens[6]
    biased_denominator = sum(
        x3.routing.biased_scores[expert - 1] for expert in x3.routing.selected
    )
    biased_weights = tuple(
        x3.routing.biased_scores[expert - 1] / biased_denominator
        for expert in x3.routing.selected
    )
    bias_leakage = (
        max(
            abs(left - right)
            for left, right in zip(biased_weights, x3.routing.weights, strict=True)
        )
        > 1e-3
    )

    wrong_norm = rms_norm(
        tuple(sum(record.output[c] for record in x3.experts) for c in range(4))
    )
    normalization_placement = (
        max(
            abs(a - b) for a, b in zip(wrong_norm, x3.normalized_aggregate, strict=True)
        )
        > 1e-3
    )

    projection = ((101.0, 0.0, 0.0),)
    projected = matvec(projection, (100.0, 0.0, 0.0))[0]
    projection_bound = projected > situ_coordinate_bound()

    quantile_counterexample = (
        local_quantile_average_counterexample()["different"] is True
    )

    wrong_router_shape = False
    try:
        forward_token("bad", (0.0,) * 8, (0.0,) * 3)
    except ValueError:
        wrong_router_shape = True

    stale_step_mutation = RUNNING_BIASES != exact_qb_update()["centered_bias"]
    return {
        "wrong_router_shape_detected": wrong_router_shape,
        "bias_leakage_detected": bias_leakage,
        "misplaced_normalization_detected": normalization_placement,
        "post_projection_bound_overclaim_detected": projection_bound,
        "local_quantile_averaging_detected": quantile_counterexample,
        "same_step_bias_mutation_detected": stale_step_mutation,
    }


__all__ = [
    "add_histograms",
    "exact_histogram_targets",
    "exact_qb_update",
    "expert_loads",
    "fault_detection_summary",
    "histogram_counts",
    "histogram_quantile_estimate",
    "interior_qb_coordinate_witness",
    "max_forward_error",
    "partitioned_histogram_estimate",
    "recover_histogram_quantile",
    "required_bias_samples",
    "selected_sets",
    "token_cutoffs",
]
