"""Logical resource ledgers for Chapter 7 Attention Residuals."""

from __future__ import annotations

from .reference_proofs import block_interaction_count, full_interaction_count
from .reference_types import K3_DIMS, RUNNING_DIMS, Dimensions, ResourceLedger


def full_resource_ledger(
    dims: Dimensions = RUNNING_DIMS,
    *,
    batch: int = 1,
    tokens: int | None = None,
) -> ResourceLedger:
    """Return Full AttnRes logical parameter, work, and live-state counts."""

    active_tokens = dims.tokens if tokens is None else tokens
    if batch <= 0 or active_tokens <= 0:
        raise ValueError("batch and token count must be positive")
    maximum_sources = dims.transformations + 1
    return ResourceLedger(
        pseudo_query_parameters=maximum_sources * dims.model,
        source_interactions=full_interaction_count(dims.transformations),
        maximum_live_sources=maximum_sources,
        live_source_scalars=batch * active_tokens * dims.model * maximum_sources,
    )


def block_resource_ledger(
    dims: Dimensions = RUNNING_DIMS,
    *,
    batch: int = 1,
    tokens: int | None = None,
) -> ResourceLedger:
    """Return Block AttnRes logical parameter, work, and live-state counts."""

    active_tokens = dims.tokens if tokens is None else tokens
    if batch <= 0 or active_tokens <= 0:
        raise ValueError("batch and token count must be positive")
    maximum_sources = dims.terminal_sources
    return ResourceLedger(
        pseudo_query_parameters=(dims.transformations + 1) * dims.model,
        source_interactions=block_interaction_count(
            dims.decoder_layers,
            dims.block_size,
            points_per_layer=dims.points_per_layer,
        ),
        maximum_live_sources=maximum_sources,
        live_source_scalars=batch * active_tokens * dims.model * maximum_sources,
    )


def k3_boundary_ledger() -> dict[str, object]:
    """Return exact K3 block and source counts from the pinned configuration."""

    full = full_resource_ledger(K3_DIMS)
    block = block_resource_ledger(K3_DIMS)
    return {
        "decoder_layers": K3_DIMS.decoder_layers,
        "points_per_decoder_layer": K3_DIMS.points_per_layer,
        "transformations": K3_DIMS.transformations,
        "decoder_layer_block_size": K3_DIMS.block_size,
        "block_lengths": list(K3_DIMS.block_lengths),
        "non_embedding_blocks": K3_DIMS.blocks,
        "terminal_sources": K3_DIMS.terminal_sources,
        "full_source_interactions": full.source_interactions,
        "block_source_interactions": block.source_interactions,
        "full_maximum_live_sources": full.maximum_live_sources,
        "block_maximum_live_sources": block.maximum_live_sources,
        "pseudo_query_parameters": block.pseudo_query_parameters,
    }


__all__ = [
    "block_resource_ledger",
    "full_resource_ledger",
    "k3_boundary_ledger",
]
