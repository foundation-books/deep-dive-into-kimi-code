"""Running data and numerical-gradient helpers for the MoBA reference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


Vector = tuple[float, ...]
HeadRow = tuple[Vector, ...]
Tensor3 = tuple[HeadRow, ...]
AttentionFunction = Callable[[Tensor3, Tensor3, Tensor3, int, int], Tensor3]


@dataclass(frozen=True, order=True)
class BlockRoute:
    """One query-head route to one completed historical block."""

    query: int
    head: int
    block: int
    score: float


@dataclass(frozen=True)
class OnlineState:
    """Stable softmax summary for one nonempty token partition."""

    maximum: float
    denominator: float
    numerator: Vector


def partition_blocks(length: int, block_size: int) -> tuple[tuple[int, ...], ...]:
    """Partition real positions into contiguous blocks, including a short tail."""

    if length <= 0:
        raise ValueError("length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return tuple(
        tuple(range(start, min(start + block_size, length)))
        for start in range(0, length, block_size)
    )


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
RUNNING_BLOCK_SIZE = 2
RUNNING_TOTAL_BLOCK_BUDGET = 3

RUNNING_REPRESENTATIVES: tuple[HeadRow, ...] = (
    ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ((0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
    ((0.5, 0.5, 0.0, 0.0), (0.0, 0.5, 0.5, 0.0)),
)

_HEAD_ZERO_QUERIES = (
    (1.0, 0.2, 0.1, 0.0),
    (1.1, 0.2, 0.1, 0.0),
    (1.5, 0.5, 0.2, 0.1),
    (1.4, 0.6, 0.2, 0.1),
    (0.4, 1.8, 0.2, 0.1),
    (0.5, 1.7, 0.3, 0.1),
    (1.4, 0.3, 1.8, 0.2),
    (1.3, 0.4, 1.7, 0.2),
    (1.2, 0.4, 0.8, 2.0),
    (2.0, 1.5, 0.4, 0.2),
)

_HEAD_ONE_QUERIES = (
    (0.2, 0.1, 1.0, 0.0),
    (0.2, 0.1, 1.1, 0.0),
    (0.5, 0.2, 1.5, 0.1),
    (0.6, 0.2, 1.4, 0.1),
    (1.7, 0.5, 1.2, 0.1),
    (1.6, 0.6, 1.1, 0.1),
    (0.6, 1.8, 1.4, 0.2),
    (0.7, 1.7, 1.3, 0.2),
    (1.4, 0.7, 0.3, 2.0),
    (1.8, 2.0, 1.2, 0.2),
)


def running_tensors() -> tuple[Tensor3, Tensor3, Tensor3]:
    """Return the explicit ten-token, two-head, four-wide running tensors."""

    queries: Tensor3 = tuple(
        (_HEAD_ZERO_QUERIES[position], _HEAD_ONE_QUERIES[position])
        for position in range(len(ORIGINS))
    )
    offset = (0.10, -0.10, 0.05, -0.05)
    key_rows: list[HeadRow] = []
    value_rows: list[HeadRow] = []
    for position in range(len(ORIGINS)):
        block = position // RUNNING_BLOCK_SIZE
        direction = 1.0 if position % RUNNING_BLOCK_SIZE == 0 else -1.0
        key_rows.append(
            tuple(
                tuple(
                    representative[coordinate] + direction * offset[coordinate]
                    for coordinate in range(4)
                )
                for representative in RUNNING_REPRESENTATIVES[block]
            )
        )
        value_rows.append(
            tuple(
                tuple(
                    (position + 1) * (head + 1) * (coordinate + 1) / 20.0
                    for coordinate in range(4)
                )
                for head in range(2)
            )
        )
    return queries, tuple(key_rows), tuple(value_rows)


def _shape(tensor: Tensor3, name: str) -> tuple[int, int, int]:
    """Return a positive rectangular tensor shape."""

    if not tensor or not tensor[0] or not tensor[0][0]:
        raise ValueError(f"{name} must have positive length, heads, and width")
    heads = len(tensor[0])
    width = len(tensor[0][0])
    if any(len(row) != heads for row in tensor):
        raise ValueError(f"{name} must have a constant head count")
    if any(len(vector) != width for row in tensor for vector in row):
        raise ValueError(f"{name} must have a constant head width")
    return len(tensor), heads, width


def _validate_qkv(
    queries: Tensor3, keys: Tensor3, values: Tensor3
) -> tuple[int, int, int]:
    """Return the shared shape of three matching tensors."""

    query_shape = _shape(queries, "queries")
    if _shape(keys, "keys") != query_shape or _shape(values, "values") != query_shape:
        raise ValueError("queries, keys, and values must have identical shapes")
    return query_shape


def _objective(outputs: Tensor3) -> float:
    return sum(
        value * (1.0 + position + head / 10.0 + coordinate / 100.0)
        for position, row in enumerate(outputs)
        for head, vector in enumerate(row)
        for coordinate, value in enumerate(vector)
    )


def _replace_coordinate(
    tensor: Tensor3, position: int, head: int, coordinate: int, value: float
) -> Tensor3:
    rows = [[list(vector) for vector in row] for row in tensor]
    rows[position][head][coordinate] = value
    return tuple(tuple(tuple(vector) for vector in row) for row in rows)


def finite_difference_gradients(
    attention: AttentionFunction,
    queries: Tensor3,
    keys: Tensor3,
    values: Tensor3,
    block_size: int,
    total_budget: int,
    epsilon: float = 1e-5,
) -> tuple[Tensor3, Tensor3, Tensor3]:
    """Differentiate a fixed scalar objective by centered finite differences."""

    length, heads, width = _validate_qkv(queries, keys, values)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    tensors = (queries, keys, values)
    gradients: list[Tensor3] = []
    for tensor_index, tensor in enumerate(tensors):
        gradient_rows: list[HeadRow] = []
        for position in range(length):
            gradient_heads: list[Vector] = []
            for head in range(heads):
                gradient_vector = []
                for coordinate in range(width):
                    center = tensor[position][head][coordinate]
                    plus_tensor = _replace_coordinate(
                        tensor, position, head, coordinate, center + epsilon
                    )
                    minus_tensor = _replace_coordinate(
                        tensor, position, head, coordinate, center - epsilon
                    )
                    plus_inputs = list(tensors)
                    minus_inputs = list(tensors)
                    plus_inputs[tensor_index] = plus_tensor
                    minus_inputs[tensor_index] = minus_tensor
                    plus_value = _objective(
                        attention(
                            plus_inputs[0],
                            plus_inputs[1],
                            plus_inputs[2],
                            block_size,
                            total_budget,
                        )
                    )
                    minus_value = _objective(
                        attention(
                            minus_inputs[0],
                            minus_inputs[1],
                            minus_inputs[2],
                            block_size,
                            total_budget,
                        )
                    )
                    gradient_vector.append((plus_value - minus_value) / (2 * epsilon))
                gradient_heads.append(tuple(gradient_vector))
            gradient_rows.append(tuple(gradient_heads))
        gradients.append(tuple(gradient_rows))
    return gradients[0], gradients[1], gradients[2]


def mutate_suffix(tensor: Tensor3, first_position: int, delta: float) -> Tensor3:
    """Add a deterministic perturbation to every coordinate in a suffix."""

    length, _, _ = _shape(tensor, "tensor")
    if not 0 <= first_position <= length:
        raise ValueError("first_position must lie in or just after the sequence")
    return tuple(
        tuple(
            tuple(
                value + delta * (1.0 + coordinate)
                if position >= first_position
                else value
                for coordinate, value in enumerate(vector)
            )
            for vector in row
        )
        for position, row in enumerate(tensor)
    )
