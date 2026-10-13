"""Exact specialized witnesses for MoE, AttnRes, gradients, and Muon."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence


Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def add_vectors(vectors: Iterable[Vector]) -> Vector:
    """Add equal-length exact vectors elementwise."""
    rows = tuple(vectors)
    return tuple(sum(column, Fraction(0)) for column in zip(*rows, strict=True))


def scale_vector(scale: Fraction, vector: Vector) -> Vector:
    """Multiply an exact vector by a rational scalar."""
    return tuple(scale * value for value in vector)


def dot(left: Vector, right: Vector) -> Fraction:
    """Return the exact dot product of two equal-length vectors."""
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def moe_backward(
    weights: Sequence[Fraction], activations: Sequence[Vector], upstream: Vector
) -> dict[str, object]:
    """Evaluate the chapter's exact MoE output and local adjoints."""
    weighted = tuple(
        scale_vector(weight, activation)
        for weight, activation in zip(weights, activations, strict=True)
    )
    return {
        "output": add_vectors(weighted),
        "activation_grads": tuple(scale_vector(weight, upstream) for weight in weights),
        "route_grads": tuple(dot(upstream, activation) for activation in activations),
    }


def moe_fixture() -> dict[str, object]:
    """Evaluate the fixed three-route MoE running example."""
    return moe_backward(
        (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
        (
            (Fraction(1), Fraction(2)),
            (Fraction(-1), Fraction(3)),
            (Fraction(2), Fraction(0)),
        ),
        (Fraction(4), Fraction(-2)),
    )


def attnres_release(
    produced: int,
    local_consumers: Sequence[int],
    send_complete: int,
    remote_consumers: Sequence[int],
) -> tuple[int, int]:
    """Return earliest safe source and destination AttnRes releases."""
    del produced
    source_release = max((*local_consumers, send_complete))
    destination_release = max(remote_consumers)
    return source_release, destination_release


def pipeline_balanced_peaks(receiver_workspace: int = 0) -> tuple[int, int]:
    """Return rank peaks after the running example's remote loans."""
    return 36 - 4, 24 + 4 + receiver_workspace


def pipeline_schedule_feasible(receiver_workspace: int = 0) -> bool:
    """Test both balanced rank peaks against the 32-unit capacities."""
    rank0, rank1 = pipeline_balanced_peaks(receiver_workspace)
    return rank0 <= 32 and rank1 <= 32


def reduce_scatter_to_cpu(
    contributions: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reduce 16-element replica gradients into two CPU owner shards."""
    if not contributions or any(len(row) != 16 for row in contributions):
        raise ValueError("expected nonempty 16-element replica gradients")
    reduced = tuple(sum(column) for column in zip(*contributions, strict=True))
    return reduced[:8], reduced[8:]


def assemble(shards: Sequence[Matrix], row_ranges: Sequence[tuple[int, int]]) -> Matrix:
    """Assemble complete row shards in canonical contiguous order."""
    if len(shards) != len(row_ranges):
        raise ValueError("shard manifest length mismatch")
    ordered = sorted(zip(row_ranges, shards, strict=True), key=lambda item: item[0])
    cursor = 0
    rows: list[Vector] = []
    for (start, stop), shard in ordered:
        if start != cursor or stop - start != len(shard):
            raise ValueError("incomplete or overlapping row shard map")
        rows.extend(shard)
        cursor = stop
    return tuple(rows)


def transpose(matrix: Matrix) -> Matrix:
    """Transpose a rectangular exact matrix."""
    return tuple(tuple(column) for column in zip(*matrix, strict=True))


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two compatible exact matrices."""
    right_t = transpose(right)
    return tuple(tuple(dot(row, column) for column in right_t) for row in left)


def subtract(left: Matrix, right: Matrix) -> Matrix:
    """Subtract equal-shaped exact matrices elementwise."""
    return tuple(
        tuple(a - b for a, b in zip(lrow, rrow, strict=True))
        for lrow, rrow in zip(left, right, strict=True)
    )


def scale_matrix(scale: Fraction, matrix: Matrix) -> Matrix:
    """Multiply every entry of an exact matrix by a rational scalar."""
    return tuple(scale_vector(scale, row) for row in matrix)


def newton_schulz_step(matrix: Matrix) -> Matrix:
    """One exact cubic step: 3/2 X - 1/2 X X^T X."""
    cubic = matmul(matmul(matrix, transpose(matrix)), matrix)
    return subtract(
        scale_matrix(Fraction(3, 2), matrix), scale_matrix(Fraction(1, 2), cubic)
    )


def matrix_fixture(offset: int) -> Matrix:
    """Construct the fixed 4-by-4 exact matrix used by Muon checks."""
    return tuple(
        tuple(Fraction(offset + row * 4 + column + 1, 32) for column in range(4))
        for row in range(4)
    )


def owner_directed_result(matrix: Matrix) -> Matrix:
    """Assemble owner-directed shards and apply the finite Muon map."""
    shards = (matrix[:2], matrix[2:])
    assembled = assemble(shards, ((0, 2), (2, 4)))
    return newton_schulz_step(assembled)


def muon_memory_counts() -> dict[str, int]:
    """Return logical peak and receive counts for both Muon strategies."""
    return {
        "gather_peak_elements": 32,
        "gather_receive_elements": 16,
        "owner_peak_elements": 16,
        "owner_receive_elements": 8,
    }
