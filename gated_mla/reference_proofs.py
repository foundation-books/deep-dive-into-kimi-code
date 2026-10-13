"""Exact absorption identities for the Chapter 6 Gated MLA oracle."""

from __future__ import annotations

from fractions import Fraction


ExactVector = tuple[Fraction, ...]
ExactMatrix = tuple[ExactVector, ...]


def _require_width(vector: ExactVector, width: int, name: str) -> None:
    if len(vector) != width:
        raise ValueError(f"{name} must have width {width}, got {len(vector)}")


def _matvec(matrix: ExactMatrix, vector: ExactVector) -> ExactVector:
    if not matrix:
        return ()
    _require_width(vector, len(matrix[0]), "exact matrix input")
    return tuple(
        sum(
            (value * item for value, item in zip(row, vector, strict=True)),
            Fraction(0),
        )
        for row in matrix
    )


def _transpose(matrix: ExactMatrix) -> ExactMatrix:
    if not matrix:
        return ()
    return tuple(
        tuple(row[column] for row in matrix) for column in range(len(matrix[0]))
    )


def _dot(left: ExactVector, right: ExactVector) -> Fraction:
    _require_width(left, len(right), "exact dot-product left operand")
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def exact_key_absorption(
    q: ExactVector, matrix: ExactMatrix, latent: ExactVector
) -> tuple[Fraction, Fraction]:
    """Return both sides of q^T W z using exact rational arithmetic."""

    direct = _dot(q, _matvec(matrix, latent))
    absorbed = _dot(_matvec(_transpose(matrix), q), latent)
    return direct, absorbed


def exact_value_linearity(
    probabilities: ExactVector,
    latents: tuple[ExactVector, ...],
    matrix: ExactMatrix,
) -> tuple[ExactVector, ExactVector]:
    """Return both sides of sum_j alpha_j W z_j = W sum_j alpha_j z_j."""

    if len(probabilities) != len(latents) or not latents:
        raise ValueError("one probability is required per latent")
    direct = tuple(
        sum(
            (
                probability * _matvec(matrix, latent)[channel]
                for probability, latent in zip(probabilities, latents, strict=True)
            ),
            Fraction(0),
        )
        for channel in range(len(matrix))
    )
    aggregate = tuple(
        sum(
            (
                probability * latent[channel]
                for probability, latent in zip(probabilities, latents, strict=True)
            ),
            Fraction(0),
        )
        for channel in range(len(latents[0]))
    )
    return direct, _matvec(matrix, aggregate)


__all__ = ["exact_key_absorption", "exact_value_linearity"]
