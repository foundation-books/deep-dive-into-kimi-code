"""Mutable workspace used during one integrated K3 reference forward pass."""

from __future__ import annotations

from dataclasses import dataclass

from .reference_types import (
    DepthState,
    GlobalLayerState,
    KdaLayerState,
    LayerCheckpoint,
    RouteRecord,
)


@dataclass
class ForwardAccumulator:
    """Mutable workspace for one otherwise immutable forward trace."""

    depth: DepthState
    kda_by_layer: dict[int, KdaLayerState]
    global_by_layer: dict[int, GlobalLayerState]
    checkpoints: list[LayerCheckpoint]
    routes: list[RouteRecord]
    point: int = 1


__all__ = ["ForwardAccumulator"]
