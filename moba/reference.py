"""Dependency-free reference semantics for Chapter 11.

The routines in this module make a small causal Mixture of Block Attention
(MoBA) trace executable.  They are correctness and accounting oracles, not a
GPU kernel, training recipe, or latency model.
"""

from __future__ import annotations

from math import ceil, exp, sqrt
from typing import Sequence

from .reference_online import (
    finalize_online_state,
    merge_online_states,
    online_merge_microexample as online_merge_microexample,
    online_state,
)
from .reference_support import (
    ORIGINS as ORIGINS,
    RUNNING_BLOCK_SIZE as RUNNING_BLOCK_SIZE,
    RUNNING_REPRESENTATIVES as RUNNING_REPRESENTATIVES,
    RUNNING_TOTAL_BLOCK_BUDGET as RUNNING_TOTAL_BLOCK_BUDGET,
    BlockRoute,
    _shape,
    _validate_qkv,
    finite_difference_gradients as finite_difference_gradients,
    mutate_suffix as mutate_suffix,
    partition_blocks,
    running_tensors as running_tensors,
)


Vector = tuple[float, ...]
HeadRow = tuple[Vector, ...]
Tensor3 = tuple[HeadRow, ...]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a typed dot product."""

    if not left or len(left) != len(right):
        raise ValueError("dot-product vectors must have the same positive width")
    return sum(a * b for a, b in zip(left, right, strict=True))


def mean_key_representatives(keys: Tensor3, block_size: int) -> tuple[HeadRow, ...]:
    """Mean-pool only real keys in each sequence-local block."""

    length, heads, width = _shape(keys, "keys")
    representatives: list[HeadRow] = []
    for block in partition_blocks(length, block_size):
        representatives.append(
            tuple(
                tuple(
                    sum(keys[position][head][coordinate] for position in block)
                    / len(block)
                    for coordinate in range(width)
                )
                for head in range(heads)
            )
        )
    return tuple(representatives)


def block_score_table(
    queries: Tensor3, representatives: tuple[HeadRow, ...]
) -> tuple[tuple[Vector, ...], ...]:
    """Return raw query-to-block dot products for every query and head."""

    _, heads, width = _shape(queries, "queries")
    if not representatives or any(len(block) != heads for block in representatives):
        raise ValueError("representatives must match the query head count")
    if any(len(vector) != width for block in representatives for vector in block):
        raise ValueError("representatives must match the query head width")
    return tuple(
        tuple(
            tuple(dot(query_row[head], block[head]) for block in representatives)
            for head in range(heads)
        )
        for query_row in queries
    )


def select_historical_blocks(
    scores: Sequence[float], query_position: int, block_size: int, total_budget: int
) -> tuple[int, ...]:
    """Select completed history by descending score, then lower block index."""

    if query_position < 0:
        raise ValueError("query_position must be nonnegative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if total_budget <= 0:
        raise ValueError("total_budget must include at least the current block")
    current = query_position // block_size
    if current >= len(scores):
        raise ValueError("score table does not contain the current block")
    count = min(total_budget - 1, current)
    return tuple(
        sorted(range(current), key=lambda block: (-scores[block], block))[:count]
    )


def all_selections(
    queries: Tensor3, keys: Tensor3, block_size: int, total_budget: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return selected historical block identifiers for every query and head."""

    representatives = mean_key_representatives(keys, block_size)
    scores = block_score_table(queries, representatives)
    return tuple(
        tuple(
            select_historical_blocks(
                scores[position][head], position, block_size, total_budget
            )
            for head in range(len(scores[position]))
        )
        for position in range(len(scores))
    )


def accessible_indices(
    length: int,
    query_position: int,
    block_size: int,
    selected_history: Sequence[int],
) -> tuple[int, ...]:
    """Return the sorted current-prefix and selected-history token union."""

    blocks = partition_blocks(length, block_size)
    if not 0 <= query_position < length:
        raise ValueError("query_position is outside the sequence")
    current = query_position // block_size
    if len(set(selected_history)) != len(selected_history):
        raise ValueError("selected historical blocks must be distinct")
    if any(block < 0 or block >= current for block in selected_history):
        raise ValueError("selected blocks must be completed history")
    positions = {position for block in selected_history for position in blocks[block]}
    positions.update(
        position for position in blocks[current] if position <= query_position
    )
    return tuple(sorted(positions))


def _softmax_output(logits: Sequence[float], values: Sequence[Vector]) -> Vector:
    if not logits or len(logits) != len(values):
        raise ValueError("logits and values must have equal positive length")
    width = len(values[0])
    if width == 0 or any(len(value) != width for value in values):
        raise ValueError("values must have a constant positive width")
    maximum = max(logits)
    weights = tuple(exp(logit - maximum) for logit in logits)
    denominator = sum(weights)
    return tuple(
        sum(
            weight * value[coordinate]
            for weight, value in zip(weights, values, strict=True)
        )
        / denominator
        for coordinate in range(width)
    )


def dense_masked_attention(
    queries: Tensor3,
    keys: Tensor3,
    values: Tensor3,
    block_size: int,
    total_budget: int,
) -> Tensor3:
    """Evaluate MoBA semantics directly over each accessible token set."""

    length, heads, width = _validate_qkv(queries, keys, values)
    selections = all_selections(queries, keys, block_size, total_budget)
    scale = sqrt(width)
    outputs: list[HeadRow] = []
    for position in range(length):
        output_heads: list[Vector] = []
        for head in range(heads):
            accessible = accessible_indices(
                length,
                position,
                block_size,
                selections[position][head],
            )
            logits = tuple(
                dot(queries[position][head], keys[key_position][head]) / scale
                for key_position in accessible
            )
            output_heads.append(
                _softmax_output(
                    logits,
                    tuple(values[key_position][head] for key_position in accessible),
                )
            )
        outputs.append(tuple(output_heads))
    return tuple(outputs)


def full_causal_attention(
    queries: Tensor3,
    keys: Tensor3,
    values: Tensor3,
    block_size: int = 1,
    total_budget: int = 1,
) -> Tensor3:
    """Evaluate ordinary dense causal attention; extra arguments keep one API."""

    del block_size, total_budget
    length, heads, width = _validate_qkv(queries, keys, values)
    scale = sqrt(width)
    return tuple(
        tuple(
            _softmax_output(
                tuple(
                    dot(queries[position][head], keys[key_position][head]) / scale
                    for key_position in range(position + 1)
                ),
                tuple(
                    values[key_position][head] for key_position in range(position + 1)
                ),
            )
            for head in range(heads)
        )
        for position in range(length)
    )


def partitioned_attention(
    queries: Tensor3,
    keys: Tensor3,
    values: Tensor3,
    block_size: int,
    total_budget: int,
) -> Tensor3:
    """Evaluate current/history partitions and merge their exact softmax states."""

    length, heads, width = _validate_qkv(queries, keys, values)
    blocks = partition_blocks(length, block_size)
    selections = all_selections(queries, keys, block_size, total_budget)
    scale = sqrt(width)
    outputs: list[HeadRow] = []
    for position in range(length):
        current = position // block_size
        current_prefix = tuple(key for key in blocks[current] if key <= position)
        head_outputs: list[Vector] = []
        for head in range(heads):
            partitions = (current_prefix,) + tuple(
                blocks[block] for block in selections[position][head]
            )
            states = []
            for partition in partitions:
                logits = tuple(
                    dot(queries[position][head], keys[key_position][head]) / scale
                    for key_position in partition
                )
                states.append(
                    online_state(
                        logits,
                        tuple(values[key_position][head] for key_position in partition),
                    )
                )
            merged = states[0]
            for state in states[1:]:
                merged = merge_online_states(merged, state)
            head_outputs.append(finalize_online_state(merged))
        outputs.append(tuple(head_outputs))
    return tuple(outputs)


def max_abs_tensor_difference(left: Tensor3, right: Tensor3) -> float:
    """Return the maximum absolute coordinate difference between equal tensors."""

    if _shape(left, "left") != _shape(right, "right"):
        raise ValueError("tensors must have equal shapes")
    return max(
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for left_head, right_head in zip(left_row, right_row, strict=True)
        for a, b in zip(left_head, right_head, strict=True)
    )


def block_routes(
    queries: Tensor3, keys: Tensor3, block_size: int, total_budget: int
) -> tuple[BlockRoute, ...]:
    """Expand selected historical membership into auditable route records."""

    representatives = mean_key_representatives(keys, block_size)
    scores = block_score_table(queries, representatives)
    selections = all_selections(queries, keys, block_size, total_budget)
    return tuple(
        BlockRoute(position, head, block, scores[position][head][block])
        for position in range(len(queries))
        for head in range(len(queries[position]))
        for block in selections[position][head]
    )


def route_loads(routes: Sequence[BlockRoute], block_count: int) -> tuple[int, ...]:
    """Count historical routes received by each block."""

    if block_count <= 0:
        raise ValueError("block_count must be positive")
    if any(route.block < 0 or route.block >= block_count for route in routes):
        raise ValueError("route contains an out-of-range block")
    return tuple(
        sum(route.block == block for route in routes) for block in range(block_count)
    )


def pair_and_cost_ledger(
    queries: Tensor3, keys: Tensor3, block_size: int, total_budget: int
) -> dict[str, int | float]:
    """Return exact pair counts and a simple scalar-operation ledger."""

    length, heads, width = _shape(queries, "queries")
    if _shape(keys, "keys") != (length, heads, width):
        raise ValueError("queries and keys must have equal shapes")
    selections = all_selections(queries, keys, block_size, total_budget)
    accessible_pairs = sum(
        len(
            accessible_indices(length, position, block_size, selections[position][head])
        )
        for position in range(length)
        for head in range(heads)
    )
    dense_pairs = heads * length * (length + 1) // 2
    block_count = ceil(length / block_size)
    historical_routes = sum(
        len(selections[position][head])
        for position in range(length)
        for head in range(heads)
    )
    return {
        "dense_causal_query_key_pairs": dense_pairs,
        "moba_query_key_pairs": accessible_pairs,
        "realized_causal_density": accessible_pairs / dense_pairs,
        "realized_causal_sparsity": 1.0 - accessible_pairs / dense_pairs,
        "gate_query_block_pairs": length * heads * block_count,
        "gate_dot_product_multiply_adds": length * heads * block_count * width,
        "selected_attention_dot_product_multiply_adds": accessible_pairs * width,
        "historical_route_records": historical_routes,
        "mandatory_current_assignments": length * heads,
        "key_value_storage_scalars": 2 * length * heads * width,
        "representative_storage_scalars": block_count * heads * width,
    }


def prefix_difference(left: Tensor3, right: Tensor3, last_position: int) -> float:
    """Return a maximum error through one inclusive prefix position."""

    if not 0 <= last_position < len(left):
        raise ValueError("last_position is outside the first tensor")
    return max_abs_tensor_difference(
        left[: last_position + 1], right[: last_position + 1]
    )


def batch_attention(
    sequences: Sequence[tuple[Tensor3, Tensor3, Tensor3]],
    block_size: int,
    total_budget: int,
) -> tuple[Tensor3, ...]:
    """Evaluate sequences independently; no block or route crosses an example."""

    if not sequences:
        raise ValueError("sequences must be nonempty")
    return tuple(
        dense_masked_attention(queries, keys, values, block_size, total_budget)
        for queries, keys, values in sequences
    )
