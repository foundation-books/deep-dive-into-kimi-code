"""Running constants for the dependency-free Stable LatentMoE reference."""

from __future__ import annotations

from .reference_proofs import sigmoid
from .reference_types import Matrix, ScoreMatrix, Vector


ORIGINS = ("v1", "v2", "v3", "v4", "x1", "x2", "x3", "x4", "x5", "x6")
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

# A no-repeated-row matrix used only for the QB derivation. With the current
# bias it preserves Chapter 3's route identities and load vector (7, 6, 4, 3).
QB_SCORES: ScoreMatrix = (
    (0.88, 0.86, 0.14, 0.11),
    (0.87, 0.85, 0.15, 0.12),
    (0.86, 0.84, 0.16, 0.13),
    (0.85, 0.83, 0.17, 0.14),
    (0.82, 0.12, 0.78, 0.11),
    (0.81, 0.13, 0.77, 0.10),
    tuple(sigmoid(value) for value in RUNNING_LOGITS["x3"]),
    (0.14, 0.81, 0.83, 0.12),
    (0.13, 0.82, 0.11, 0.84),
    (0.12, 0.10, 0.80, 0.83),
)

W_DOWN: Matrix = (
    (1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.25),
)
W_UP: Matrix = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.5, 0.0, 0.5, 0.0),
    (0.0, 0.5, 0.0, 0.5),
    (0.5, -0.5, 0.0, 0.0),
    (0.0, 0.0, 0.5, -0.5),
)


__all__ = [
    "ORIGINS",
    "QB_SCORES",
    "RUNNING_BIASES",
    "RUNNING_LOGITS",
    "W_DOWN",
    "W_UP",
]
