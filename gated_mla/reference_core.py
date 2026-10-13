"""Dependency-free reference semantics for Chapter 6 Gated MLA.

This module is deliberately small and explicit.  It implements the same
causal map twice: once by reconstructing per-token keys and values, and once
by absorbing the content-key projection and aggregating values in the latent
space.  The two paths share a compact logical record ``(z_t, s_t)`` where
``z_t`` is the normalized KV latent and ``s_t`` is K3's shared unrotated NoPE
key branch.  It is a correctness oracle, not a production cache kernel.
"""

from __future__ import annotations

from math import exp, sqrt
from typing import Sequence

from .reference_types import (
    DECODE_LABEL,
    RUNNING_LABELS,
    CompactRecord,
    Dimensions,
    Matrix,
    PositionTrace,
    Projection,
    RUNNING_DIMS,
    Vector,
    Weights,
)


def _rows(seed: int, rows: int, columns: int) -> Matrix:
    """Construct a deterministic dense matrix with small nonzero entries."""

    return tuple(
        tuple(
            (((seed + 3 * row + 5 * column) % 13) - 6) / 8 for column in range(columns)
        )
        for row in range(rows)
    )


def running_weights(dims: Dimensions = RUNNING_DIMS) -> Weights:
    """Return fixed dense weights for the running example."""

    return Weights(
        q_down=_rows(1, dims.query_rank, dims.model),
        q_up=_rows(2, dims.query_projection, dims.query_rank),
        kv_down=_rows(3, dims.compact_record, dims.model),
        k_up=tuple(
            _rows(11 + head, dims.content_key, dims.latent)
            for head in range(dims.heads)
        ),
        v_up=tuple(
            _rows(19 + head, dims.value, dims.latent) for head in range(dims.heads)
        ),
        gate=_rows(29, dims.attention_output, dims.model),
        output=_rows(37, dims.model, dims.attention_output),
    )


def _require_width(vector: Sequence[object], width: int, name: str) -> None:
    if len(vector) != width:
        raise ValueError(f"{name} must have width {width}, got {len(vector)}")


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    """Multiply a row-major matrix by a vector."""

    if not matrix:
        return ()
    width = len(matrix[0])
    _require_width(vector, width, "matrix input")
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have one width")
    return tuple(
        sum(value * item for value, item in zip(row, vector, strict=True))
        for row in matrix
    )


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Transpose a nonempty rectangular matrix."""

    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows must have one width")
    return tuple(tuple(row[column] for row in matrix) for column in range(width))


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of two vectors."""
    _require_width(left, len(right), "dot-product left operand")
    return sum(a * b for a, b in zip(left, right, strict=True))


def rms_norm(vector: Sequence[float], epsilon: float = 1e-6) -> Vector:
    """Apply the unweighted RMS normalization used by the toy contract."""

    if not vector:
        raise ValueError("RMSNorm requires a nonempty vector")
    scale = 1.0 / sqrt(sum(value * value for value in vector) / len(vector) + epsilon)
    return tuple(value * scale for value in vector)


def sigmoid(value: float) -> float:
    """Return the scalar logistic sigmoid."""
    if value >= 0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    direct = exp(value)
    return direct / (1.0 + direct)


def softmax(values: Sequence[float]) -> Vector:
    """Return a stable probability vector."""

    if not values:
        raise ValueError("softmax requires a nonempty vector")
    maximum = max(values)
    numerators = tuple(exp(value - maximum) for value in values)
    denominator = sum(numerators)
    return tuple(value / denominator for value in numerators)


def token_vector(position: int, dims: Dimensions = RUNNING_DIMS) -> Vector:
    """Create one deterministic model-width input for a one-indexed position."""

    if position < 1:
        raise ValueError("positions are one-indexed")
    return tuple(
        (((position * (coordinate + 2) + 3 * coordinate) % 17) - 8) / 8
        for coordinate in range(dims.model)
    )


def running_inputs() -> tuple[tuple[str, Vector], ...]:
    """Return the four visual and six text tokens in the shared request."""

    return tuple(
        (label, token_vector(position))
        for position, label in enumerate(RUNNING_LABELS, start=1)
    )


def decode_input() -> tuple[str, Vector]:
    """Return the first answer-token input at position eleven."""

    return DECODE_LABEL, token_vector(11)


def validate_weights(weights: Weights, dims: Dimensions = RUNNING_DIMS) -> None:
    """Reject shape drift before evaluating a layer."""

    expected = (
        (weights.q_down, dims.query_rank, dims.model, "q_down"),
        (weights.q_up, dims.query_projection, dims.query_rank, "q_up"),
        (weights.kv_down, dims.compact_record, dims.model, "kv_down"),
        (weights.gate, dims.attention_output, dims.model, "gate"),
        (weights.output, dims.model, dims.attention_output, "output"),
    )
    for matrix, rows, columns, name in expected:
        if len(matrix) != rows or any(len(row) != columns for row in matrix):
            raise ValueError(f"{name} must have shape ({rows}, {columns})")
    if len(weights.k_up) != dims.heads or len(weights.v_up) != dims.heads:
        raise ValueError("one key-up and value-up matrix is required per head")
    for head, matrix in enumerate(weights.k_up):
        if len(matrix) != dims.content_key or any(
            len(row) != dims.latent for row in matrix
        ):
            raise ValueError(f"k_up[{head}] has the wrong shape")
    for head, matrix in enumerate(weights.v_up):
        if len(matrix) != dims.value or any(len(row) != dims.latent for row in matrix):
            raise ValueError(f"v_up[{head}] has the wrong shape")


def project_token(
    hidden: Sequence[float],
    weights: Weights,
    dims: Dimensions = RUNNING_DIMS,
) -> Projection:
    """Project one hidden state into queries, one compact record, K/V, and gate."""

    validate_weights(weights, dims)
    _require_width(hidden, dims.model, "hidden state")
    query_latent = rms_norm(matvec(weights.q_down, hidden))
    query_flat = matvec(weights.q_up, query_latent)
    query_content: list[Vector] = []
    query_shared: list[Vector] = []
    for head in range(dims.heads):
        start = head * dims.query_head
        query_content.append(query_flat[start : start + dims.content_key])
        query_shared.append(
            query_flat[start + dims.content_key : start + dims.query_head]
        )
    kv_flat = matvec(weights.kv_down, hidden)
    latent = rms_norm(kv_flat[: dims.latent])
    shared = kv_flat[dims.latent :]
    record = CompactRecord(latent, shared)
    keys = tuple(matvec(weights.k_up[head], latent) for head in range(dims.heads))
    values = tuple(matvec(weights.v_up[head], latent) for head in range(dims.heads))
    gate = tuple(sigmoid(value) for value in matvec(weights.gate, hidden))
    return Projection(
        tuple(query_content), tuple(query_shared), record, keys, values, gate
    )


def validate_record(record: CompactRecord, dims: Dimensions = RUNNING_DIMS) -> None:
    """Validate the record contract."""
    _require_width(record.latent, dims.latent, "cache latent")
    _require_width(record.shared_key, dims.shared_key, "shared key")


def _score(
    projection: Projection,
    head: int,
    key_content: Sequence[float],
    shared_key: Sequence[float],
    dims: Dimensions,
) -> float:
    numerator = dot(projection.query_content[head], key_content) + dot(
        projection.query_shared[head], shared_key
    )
    return numerator / sqrt(dims.query_head)


def _finish_position(
    *,
    position: int,
    label: str,
    projection: Projection,
    probabilities: tuple[Vector, ...],
    contexts: tuple[Vector, ...],
    weights: Weights,
) -> PositionTrace:
    ungated = tuple(value for context in contexts for value in context)
    gated = tuple(
        value * gate for value, gate in zip(ungated, projection.gate, strict=True)
    )
    output = matvec(weights.output, gated)
    return PositionTrace(
        position,
        label,
        probabilities,
        ungated,
        projection.gate,
        output,
        projection.record,
    )
