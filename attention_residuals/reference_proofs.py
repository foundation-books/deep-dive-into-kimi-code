"""Exact arithmetic and closed-form ledgers for Chapter 7."""

from __future__ import annotations

from fractions import Fraction


ExactVector = tuple[Fraction, ...]


def exact_residual_unroll(
    embedding: ExactVector,
    outputs: tuple[ExactVector, ...],
) -> tuple[ExactVector, ExactVector]:
    """Return iterative and directly unrolled residual states."""

    width = len(embedding)
    if width == 0 or any(len(output) != width for output in outputs):
        raise ValueError("embedding and outputs must share one nonzero width")
    iterative = embedding
    for output in outputs:
        iterative = tuple(
            left + right for left, right in zip(iterative, output, strict=True)
        )
    direct = tuple(
        embedding[channel]
        + sum(
            (output[channel] for output in outputs),
            Fraction(0),
        )
        for channel in range(width)
    )
    return iterative, direct


def exact_three_source_example() -> tuple[ExactVector, ExactVector]:
    """Return the exact AttnRes mixture and ordinary residual sum."""

    weights = (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2))
    values = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(2)),
        (Fraction(2), Fraction(2)),
    )
    mixture = tuple(
        sum(
            (
                weight * value[channel]
                for weight, value in zip(weights, values, strict=True)
            ),
            Fraction(0),
        )
        for channel in range(2)
    )
    residual = tuple(
        sum((value[channel] for value in values), Fraction(0)) for channel in range(2)
    )
    return mixture, residual


def partition_lengths(total: int, block_size: int) -> tuple[int, ...]:
    """Partition a positive count into complete blocks and one possible tail."""

    if total <= 0 or block_size <= 0:
        raise ValueError("total and block size must be positive")
    complete, remainder = divmod(total, block_size)
    return (block_size,) * complete + ((remainder,) if remainder else ())


def full_interaction_count(
    transformations: int,
    *,
    include_terminal: bool = True,
) -> int:
    """Count source-target pairs for Full AttnRes."""

    if transformations <= 0:
        raise ValueError("transformations must be positive")
    count = transformations * (transformations + 1) // 2
    return count + (transformations + 1 if include_terminal else 0)


def block_interaction_count(
    decoder_layers: int,
    block_size: int,
    *,
    points_per_layer: int = 2,
    include_terminal: bool = True,
) -> int:
    """Count direct Block AttnRes source-target pairs exactly."""

    if points_per_layer <= 0:
        raise ValueError("points per layer must be positive")
    count = 0
    lengths = partition_lengths(decoder_layers, block_size)
    for block_index, decoder_count in enumerate(lengths):
        targets = decoder_count * points_per_layer
        completed_sources = 1 + block_index
        count += completed_sources
        count += (targets - 1) * (completed_sources + 1)
    return count + (1 + len(lengths) if include_terminal else 0)


__all__ = [
    "block_interaction_count",
    "exact_residual_unroll",
    "exact_three_source_example",
    "full_interaction_count",
    "partition_lengths",
]
