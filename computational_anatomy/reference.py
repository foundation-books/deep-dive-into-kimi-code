"""CPU reference contracts for Chapter 2.

These small, dependency-free routines validate scaled-down arithmetic and state
contracts; they are not a production K3 implementation or performance model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log1p, sqrt
from typing import Sequence

Vector = list[float]
Matrix = list[list[float]]


def _check_rectangular(matrix: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not matrix:
        raise ValueError("matrix must have at least one row")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError("matrix must be nonempty and rectangular")
    return len(matrix), width


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    """Multiply two compatible matrices."""
    left_rows, left_cols = _check_rectangular(left)
    right_rows, right_cols = _check_rectangular(right)
    if left_cols != right_rows:
        raise ValueError(
            f"incompatible matrix shapes: {left_rows}x{left_cols} and {right_rows}x{right_cols}"
        )
    return [
        [
            sum(left[i][k] * right[k][j] for k in range(left_cols))
            for j in range(right_cols)
        ]
        for i in range(left_rows)
    ]


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Return the matrix transpose."""
    rows, cols = _check_rectangular(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def outer(left: Sequence[float], right: Sequence[float]) -> Matrix:
    """Return the outer product of two vectors."""
    if not left or not right:
        raise ValueError("outer-product vectors must be nonempty")
    return [[a * b for b in right] for a in left]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    """Multiply a matrix by a vector."""
    rows, cols = _check_rectangular(matrix)
    if cols != len(vector):
        raise ValueError(
            f"incompatible matrix/vector shapes: {rows}x{cols} and {len(vector)}"
        )
    return [sum(matrix[i][j] * vector[j] for j in range(cols)) for i in range(rows)]


def sigmoid(value: float) -> float:
    """Return the scalar logistic sigmoid."""
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    scaled = exp(value)
    return scaled / (1.0 + scaled)


def log_sigmoid(value: float) -> float:
    """Return the sigmoid score in log space without overflow or underflow."""

    if value >= 0.0:
        return -log1p(exp(-value))
    return value - log1p(exp(value))


def softmax(values: Sequence[float]) -> Vector:
    """Return numerically stable softmax probabilities."""
    if not values:
        raise ValueError("softmax input must be nonempty")
    maximum = max(values)
    exponentials = [exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def rms_norm(vector: Sequence[float], epsilon: float = 1e-12) -> Vector:
    """Return the vector after RMS normalization."""
    if not vector:
        raise ValueError("RMSNorm input must be nonempty")
    scale = sqrt(sum(value * value for value in vector) / len(vector) + epsilon)
    return [value / scale for value in vector]


def max_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the largest elementwise absolute difference."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def flatten_matrix(matrix: Sequence[Sequence[float]]) -> Vector:
    """Flatten a matrix in row-major order."""
    _check_rectangular(matrix)
    return [value for row in matrix for value in row]


@dataclass(frozen=True)
class KDAComponents:
    """Expose every retained, erased, written, state, and output term of a KDA step."""

    retained: Matrix
    erased: Matrix
    written: Matrix
    state: Matrix
    output: Vector


def kda_step(
    previous_state: Sequence[Sequence[float]],
    key: Sequence[float],
    value: Sequence[float],
    query: Sequence[float],
    beta: float,
    retention: Sequence[float],
) -> KDAComponents:
    """Apply the one-head KDA recurrence from K3 Equation 1."""

    key_width, value_width = _check_rectangular(previous_state)
    if len(key) != key_width or len(query) != key_width:
        raise ValueError("key and query widths must match the KDA key width")
    if len(value) != value_width or len(retention) != key_width:
        raise ValueError("value or retention width does not match the state")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if any(not 0.0 <= alpha <= 1.0 for alpha in retention):
        raise ValueError("retention factors must be in [0, 1]")

    retained = [
        [retention[i] * previous_state[i][j] for j in range(value_width)]
        for i in range(key_width)
    ]
    key_read = [
        sum(key[r] * retained[r][j] for r in range(key_width))
        for j in range(value_width)
    ]
    erased = [
        [beta * key[i] * key_read[j] for j in range(value_width)]
        for i in range(key_width)
    ]
    written = [
        [beta * key[i] * value[j] for j in range(value_width)] for i in range(key_width)
    ]
    state = [
        [retained[i][j] - erased[i][j] + written[i][j] for j in range(value_width)]
        for i in range(key_width)
    ]
    output = matvec(transpose(state), query)
    return KDAComponents(retained, erased, written, state, output)


def project(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> Vector:
    """Apply a row-major linear projection to a vector."""
    rows, cols = _check_rectangular(matrix)
    if cols != len(vector):
        raise ValueError(f"projection expects width {cols}, received {len(vector)}")
    return [sum(matrix[i][j] * vector[j] for j in range(cols)) for i in range(rows)]


def append_mla_cache(
    cache: Sequence[Sequence[float]],
    token: Sequence[float],
    compression: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Append one compressed token latent to an immutable MLA cache."""
    latent = project(token, compression)
    return tuple(tuple(row) for row in cache) + (tuple(latent),)


@dataclass(frozen=True)
class RouteDecision:
    """Capture router scores, selected experts, and normalized selected weights."""

    logits: tuple[float, ...]
    scores: tuple[float, ...]
    biased_scores: tuple[float, ...]
    selected: tuple[int, ...]
    weights: tuple[float, ...]


def route(logits: Sequence[float], biases: Sequence[float], k: int) -> RouteDecision:
    """Select top-k experts by biased score and normalize their weights."""
    if len(logits) != len(biases) or not logits:
        raise ValueError("router logits and biases must have the same nonzero length")
    if not 1 <= k <= len(logits):
        raise ValueError("k must select between one and all experts")
    if any(not isfinite(value) for value in (*logits, *biases)):
        raise ValueError("router logits and biases must be finite")
    scores = [sigmoid(value) for value in logits]
    log_scores = [log_sigmoid(value) for value in logits]
    biased = [score + bias for score, bias in zip(scores, biases, strict=True)]
    selected = sorted(
        range(len(logits)),
        key=lambda index: (-biased[index], -log_scores[index], index),
    )[:k]
    maximum = max(log_scores[index] for index in selected)
    numerators = [exp(log_scores[index] - maximum) for index in selected]
    denominator = sum(numerators)
    weights = [value / denominator for value in numerators]
    return RouteDecision(
        tuple(logits),
        tuple(scores),
        tuple(biased),
        tuple(selected),
        tuple(weights),
    )


def dispatch_and_combine(
    tokens: Sequence[Sequence[float]],
    decisions: Sequence[RouteDecision],
) -> tuple[Matrix, tuple[int, ...]]:
    """Round-trip tokens through deterministic expert-major buckets."""

    if len(tokens) != len(decisions):
        raise ValueError("every token needs one route decision")
    width = len(tokens[0])
    if width == 0 or any(len(token) != width for token in tokens):
        raise ValueError("tokens must be nonempty and share a width")

    expert_buckets: dict[int, list[tuple[int, float, Sequence[float]]]] = {}
    for token_index, (token, decision) in enumerate(
        zip(tokens, decisions, strict=True)
    ):
        for expert_index, weight in zip(
            decision.selected, decision.weights, strict=True
        ):
            expert_buckets.setdefault(expert_index, []).append(
                (token_index, weight, token)
            )

    combined = [[0.0] * width for _ in tokens]
    contribution_counts = [0] * len(tokens)
    for expert_index in sorted(expert_buckets):
        expert_scale = float(expert_index + 1)
        for token_index, weight, token in expert_buckets[expert_index]:
            for channel, value in enumerate(token):
                combined[token_index][channel] += weight * expert_scale * value
            contribution_counts[token_index] += 1
    return combined, tuple(contribution_counts)


def attnres(
    scores: Sequence[float], values: Sequence[Sequence[float]]
) -> tuple[Vector, Vector]:
    """Mix depth values using softmax-normalized Attention Residual scores."""
    if len(scores) != len(values) or not scores:
        raise ValueError("every depth source needs one score and one value")
    width = len(values[0])
    if width == 0 or any(len(value) != width for value in values):
        raise ValueError("depth values must be nonempty and share a width")
    weights = softmax(scores)
    output = [
        sum(
            weight * value[channel]
            for weight, value in zip(weights, values, strict=True)
        )
        for channel in range(width)
    ]
    return weights, output


def multimodal_permission_matrix(
    visual_tokens: int, text_tokens: int
) -> list[list[bool]]:
    """Build bidirectional-visual and causal-text attention permissions."""
    if visual_tokens < 1 or text_tokens < 1:
        raise ValueError("both modalities must contribute at least one token")
    total = visual_tokens + text_tokens
    permission = [[False] * total for _ in range(total)]
    for query in range(total):
        if query < visual_tokens:
            for key in range(visual_tokens):
                permission[query][key] = True
        else:
            for key in range(visual_tokens):
                permission[query][key] = True
            for key in range(visual_tokens, query + 1):
                permission[query][key] = True
    return permission


@dataclass(frozen=True)
class ToyState:
    """Hold recurrent KDA, MLA-cache, depth-sum, and token-count state."""

    kda_state: tuple[tuple[float, ...], ...]
    mla_cache: tuple[tuple[float, ...], ...]
    depth_sum: tuple[float, ...]
    token_count: int

    @staticmethod
    def zero() -> "ToyState":
        return ToyState(
            kda_state=((0.0, 0.0), (0.0, 0.0)),
            mla_cache=(),
            depth_sum=(0.0,) * 6,
            token_count=0,
        )


def toy_token_step(token: Sequence[float], state: ToyState) -> tuple[Vector, ToyState]:
    """Advance a small stateful trace by one token.

    This composes a KDA recurrence, an MLA-shaped latent-cache append, a depth
    summary, and a deterministic readout. Its sole purpose is to test the
    prefix-state contract.
    """

    if len(token) != 6:
        raise ValueError("toy trace tokens have width six")
    kda = kda_step(
        state.kda_state,
        key=token[0:2],
        value=token[2:4],
        query=token[4:6],
        beta=0.25,
        retention=(0.9, 0.8),
    )
    compression = (
        (0.5, 0.0, 0.5, 0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0, 0.5, 0.0, 0.0),
    )
    cache = append_mla_cache(state.mla_cache, token, compression)
    depth_sum = tuple(
        previous + current
        for previous, current in zip(state.depth_sum, token, strict=True)
    )
    logits = [
        kda.output[0] + cache[-1][0] + 0.1 * depth_sum[0],
        kda.output[1] + cache[-1][1] + 0.1 * depth_sum[1],
        0.05 * sum(latent[0] for latent in cache) + 0.1 * depth_sum[2],
    ]
    next_state = ToyState(
        kda_state=tuple(tuple(row) for row in kda.state),
        mla_cache=cache,
        depth_sum=depth_sum,
        token_count=state.token_count + 1,
    )
    return logits, next_state


def evaluate_tokens(
    tokens: Sequence[Sequence[float]], initial_state: ToyState | None = None
) -> tuple[Vector, ToyState]:
    """Run the stateful toy trace and return its final logits and state."""
    if not tokens:
        raise ValueError("at least one token is required")
    state = initial_state or ToyState.zero()
    logits: Vector = []
    for token in tokens:
        logits, state = toy_token_step(token, state)
    return logits, state


def state_vector(state: ToyState) -> Vector:
    """Flatten every logical state component into one comparison vector."""
    return (
        flatten_matrix(state.kda_state)
        + [value for row in state.mla_cache for value in row]
        + list(state.depth_sum)
        + [float(state.token_count)]
    )


def resource_ledger() -> dict[str, int | float]:
    """Return scalar counts for the chapter's fixed resource example."""
    batch = 1
    tokens = 10
    model_width = 8
    heads = 2
    key_width = 2
    value_width = 2
    kda_layers = 3
    latent_cache_width = 4
    full_kv_head_width = 2
    live_depth_sources = 2
    selected_experts = 2
    latent_expert_width = 4

    full_kv_scalars = 2 * batch * heads * tokens * full_kv_head_width
    mla_cache_scalars = batch * tokens * latent_cache_width
    return {
        "prompt_activation_scalars": batch * tokens * model_width,
        "kda_state_scalars": kda_layers * heads * key_width * value_width,
        "full_kv_cache_scalars": full_kv_scalars,
        "mla_latent_cache_scalars": mla_cache_scalars,
        "full_to_latent_cache_ratio": full_kv_scalars / mla_cache_scalars,
        "block_attnres_live_scalars": live_depth_sources * batch * tokens * model_width,
        "routed_payload_scalars_per_token_round_trip": 2
        * selected_experts
        * latent_expert_width,
    }
