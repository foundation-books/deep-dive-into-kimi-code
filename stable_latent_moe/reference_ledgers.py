"""Logical resource ledgers for Chapter 8 Stable LatentMoE."""

from __future__ import annotations

from dataclasses import asdict

from .reference_types import K3_DIMS, RUNNING_DIMS, Dimensions, ResourceLedger


def resource_ledger(dims: Dimensions = RUNNING_DIMS) -> ResourceLedger:
    """Count bias-free operator parameters and leading multiply-add units.

    Shared-expert parameters and work are intentionally excluded because the
    chapter treats each shared expert as an abstract full-width map.  Routed
    experts use three matrices: gate and up (h by lambda) plus output
    (lambda by h).  RMSNorm contributes one learned gain per latent channel.
    """

    d = dims.model
    latent = dims.latent
    hidden = dims.intermediate
    experts = dims.routed_experts
    selected = dims.selected_experts
    router = experts * d
    down = latent * d
    routed = experts * 3 * latent * hidden
    norm = latent
    up = d * latent
    multiply_adds = router + down + selected * 3 * latent * hidden + up
    payload = dims.route_records * latent
    return ResourceLedger(
        router_parameters=router,
        down_parameters=down,
        routed_expert_parameters=routed,
        aggregate_norm_parameters=norm,
        up_parameters=up,
        matrix_multiply_adds_per_token=multiply_adds,
        route_payload_scalars=payload,
    )


def payload_ledger(dims: Dimensions = RUNNING_DIMS) -> dict[str, int | float]:
    """Compare one-way full-width and latent logical routed payloads."""

    records = dims.route_records
    full = records * dims.model
    latent = records * dims.latent
    return {
        "route_records": records,
        "full_width_one_way_scalars": full,
        "latent_one_way_scalars": latent,
        "full_to_latent_ratio": full / latent,
    }


def activation_inventory(dims: Dimensions = RUNNING_DIMS) -> dict[str, int]:
    """Return materialized scalar counts, without asserting a peak schedule."""

    return {
        "model_input": dims.tokens * dims.model,
        "router_scores": dims.tokens * dims.routed_experts,
        "latent_input": dims.tokens * dims.latent,
        "selected_gate_pre": dims.tokens * dims.selected_experts * dims.intermediate,
        "selected_up_pre": dims.tokens * dims.selected_experts * dims.intermediate,
        "selected_situ_intermediate": dims.tokens
        * dims.selected_experts
        * dims.intermediate,
        "selected_expert_output": dims.tokens * dims.selected_experts * dims.latent,
        "routed_aggregate": dims.tokens * dims.latent,
        "normalized_aggregate": dims.tokens * dims.latent,
        "model_output": dims.tokens * dims.model,
    }


def ledger_bundle() -> dict[str, object]:
    """Return toy and K3 ledgers with explicit unsupported quantities."""

    return {
        "toy": {
            "parameters_and_work": asdict(resource_ledger(RUNNING_DIMS)),
            "payload": payload_ledger(RUNNING_DIMS),
            "activations": activation_inventory(RUNNING_DIMS),
        },
        "k3": {
            "parameters_and_work": asdict(resource_ledger(K3_DIMS)),
            "one_token_payload": payload_ledger(K3_DIMS),
        },
        "excluded": [
            "shared-expert parameters and arithmetic",
            "bias vectors not fixed by the typed matrix convention",
            "optimizer and gradient state",
            "padding, metadata, return traffic, collectives, and workspace",
            "physical bytes, latency, throughput, and peak memory",
        ],
    }


__all__ = [
    "K3_DIMS",
    "RUNNING_DIMS",
    "activation_inventory",
    "ledger_bundle",
    "payload_ledger",
    "resource_ledger",
]
