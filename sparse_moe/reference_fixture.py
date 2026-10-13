"""Canonical Chapter 3 sparse-MoE routing fixture."""

from __future__ import annotations


Vector = tuple[float, ...]

ORIGINS = (
    "v1",
    "v2",
    "v3",
    "v4",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
)
RUNNING_BIASES: Vector = (0.2, -0.1, 0.0, 0.1)
RUNNING_LOGITS: dict[str, Vector] = {
    "v1": (2.0, 2.0, -2.0, -2.0),
    "v2": (2.0, 2.0, -2.0, -2.0),
    "v3": (2.0, 2.0, -2.0, -2.0),
    "v4": (2.0, 2.0, -2.0, -2.0),
    "x1": (2.0, -2.0, 2.0, -2.0),
    "x2": (2.0, -2.0, 2.0, -2.0),
    "x3": (0.1, 0.7, 0.4, 0.6),
    "x4": (-2.0, 2.0, 2.0, -2.0),
    "x5": (-2.0, 2.0, -2.0, 2.0),
    "x6": (-2.0, -2.0, 2.0, 2.0),
}


__all__ = ["ORIGINS", "RUNNING_BIASES", "RUNNING_LOGITS", "Vector"]
