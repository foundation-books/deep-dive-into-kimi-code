"""Exact, dependency-free reference semantics for Chapter 4 KDA examples.

The transparent algebra uses ``[key, value]`` states and post-update reads.
Fractions preserve the running trace; the same functions accept floats.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Callable, Iterable, Sequence

from .reference_linear import (
    E1,
    E2,
    Matrix,
    Number,
    Q34,
    Q4M3,
    Vector,
    _identity,
    _matrix_add,
    _matrix_multiply,
    _matrix_vector_transpose_read,
    _norm_is_one,
    _outer,
    _scale_matrix,
    _zero_matrix,
    f,
    matrix_as_floats as matrix_as_floats,
    vector_as_floats as vector_as_floats,
)
from .reference_types import KdaInput, StepResult, Trace

TOKEN_LABELS = ("v1", "v2", "v3", "v4", "x1", "x2", "x3", "x4", "x5", "x6")
DECODE_LABEL = "square"
RUNNING_HEADS, RUNNING_KEY_WIDTH, RUNNING_VALUE_WIDTH = (2, 2, 2)
RUNNING_CHUNK_SIZE = 4


def validate_input(item: KdaInput, state: Matrix) -> None:
    """Enforce the chapter's dimensions, normalization, and gate domains."""

    key_width = len(item.k)
    value_width = len(item.v)
    if key_width == 0 or value_width == 0:
        raise ValueError("KDA vectors must be nonempty")
    if len(item.q) != key_width or len(item.alpha) != key_width:
        raise ValueError("q, k, and alpha must share the key width")
    if len(state) != key_width or any(len(row) != value_width for row in state):
        raise ValueError("state must have shape [key_width, value_width]")
    if not _norm_is_one(item.q) or not _norm_is_one(item.k):
        raise ValueError("the frozen chapter inputs require unit q and k")
    if not all(0 < float(alpha) < 1 for alpha in item.alpha):
        raise ValueError(
            "every retention factor must lie strictly between zero and one"
        )
    if not 0 < float(item.beta) < 1:
        raise ValueError("write strength must lie strictly between zero and one")


def kda_step(state: Matrix, item: KdaInput) -> StepResult:
    """Apply retain, predict, correct, and post-update read for one head."""

    validate_input(item, state)
    retained = tuple(
        tuple(item.alpha[row] * value for value in state[row])
        for row in range(len(state))
    )
    prediction = _matrix_vector_transpose_read(retained, item.k)
    error = tuple(
        value - predicted for value, predicted in zip(item.v, prediction, strict=True)
    )
    correction = _scale_matrix(item.beta, _outer(item.k, error))
    new_state = _matrix_add(retained, correction)
    output = _matrix_vector_transpose_read(new_state, item.q)
    return StepResult(retained, prediction, error, correction, new_state, output)


def _sequence_states(
    sequence: Sequence[Sequence[KdaInput]],
    initial_states: Sequence[Matrix] | None,
) -> tuple[int, list[Matrix]]:
    """Validate sequence head shape and return one initial state per head."""

    if not sequence:
        raise ValueError("a KDA sequence must contain at least one position")
    head_count = len(sequence[0])
    if head_count == 0 or any(len(position) != head_count for position in sequence):
        raise ValueError("every position must contain the same positive head count")
    first = sequence[0][0]
    if initial_states is None:
        states = [_zero_matrix(len(first.k), len(first.v)) for _ in range(head_count)]
    else:
        if len(initial_states) != head_count:
            raise ValueError("initial-state count must match head count")
        states = list(initial_states)
    return head_count, states


def run_recurrent(
    sequence: Sequence[Sequence[KdaInput]],
    initial_states: Sequence[Matrix] | None = None,
) -> Trace:
    """Evaluate a sequence position by position with isolated per-head states."""

    _, states = _sequence_states(sequence, initial_states)
    positions: list[tuple[StepResult, ...]] = []
    for position in sequence:
        results: list[StepResult] = []
        for head, item in enumerate(position):
            result = kda_step(states[head], item)
            states[head] = result.state
            results.append(result)
        positions.append(tuple(results))
    return Trace(tuple(positions), tuple(states))


def transition_affine(item: KdaInput) -> tuple[Matrix, Matrix]:
    """Return ``M, b`` such that one update is ``S_new = M S + b``."""

    width = len(item.k)
    diagonal = tuple(
        tuple(item.alpha[row] if row == column else f(0) for column in range(width))
        for row in range(width)
    )
    erase = _scale_matrix(item.beta, _outer(item.k, item.k))
    householder = _matrix_add(_identity(width), _scale_matrix(f(-1), erase))
    transition = _matrix_multiply(householder, diagonal)
    write = _scale_matrix(item.beta, _outer(item.k, item.v))
    return transition, write


def compose_affine(
    later: tuple[Matrix, Matrix], earlier: tuple[Matrix, Matrix]
) -> tuple[Matrix, Matrix]:
    """Compose adjacent affine state maps in execution order."""

    later_a, later_b = later
    earlier_a, earlier_b = earlier
    return (
        _matrix_multiply(later_a, earlier_a),
        _matrix_add(_matrix_multiply(later_a, earlier_b), later_b),
    )


def segment_summary(items: Sequence[KdaInput]) -> tuple[Matrix, Matrix]:
    """Summarize a nonempty token segment as one affine state map."""

    if not items:
        raise ValueError("an affine segment must be nonempty")
    key_width = len(items[0].k)
    value_width = len(items[0].v)
    summary = (_identity(key_width), _zero_matrix(key_width, value_width))
    for item in items:
        summary = compose_affine(transition_affine(item), summary)
    return summary


def apply_affine(summary: tuple[Matrix, Matrix], state: Matrix) -> Matrix:
    """Apply an affine summary to a recurrent state."""
    transition, write = summary
    return _matrix_add(_matrix_multiply(transition, state), write)


def run_chunked(
    sequence: Sequence[Sequence[KdaInput]],
    chunk_size: int,
    initial_states: Sequence[Matrix] | None = None,
) -> Trace:
    """Check each chunk boundary against causal recurrent outputs.

    This is a correctness oracle, not a performance implementation.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    head_count, states = _sequence_states(sequence, initial_states)
    positions: list[tuple[StepResult, ...]] = []
    for start in range(0, len(sequence), chunk_size):
        chunk = sequence[start : start + chunk_size]
        incoming = tuple(states)
        for position in chunk:
            results: list[StepResult] = []
            for head, item in enumerate(position):
                result = kda_step(states[head], item)
                states[head] = result.state
                results.append(result)
            positions.append(tuple(results))
        for head in range(head_count):
            items = tuple(position[head] for position in chunk)
            expected = apply_affine(segment_summary(items), incoming[head])
            boundary_error = max(
                abs(float(observed) - float(target))
                for observed_row, target_row in zip(states[head], expected, strict=True)
                for observed, target in zip(observed_row, target_row, strict=True)
            )
            if boundary_error > 1e-12:
                raise AssertionError("chunk boundary disagrees with affine summary")
    return Trace(tuple(positions), tuple(states))


def _item(
    q: Vector,
    k: Vector,
    v: Vector,
    alpha: Vector,
    beta: Number,
) -> KdaInput:
    return KdaInput(q=q, k=k, v=v, alpha=alpha, beta=beta)


def running_inputs() -> tuple[tuple[KdaInput, KdaInput], ...]:
    """Return the frozen ten-position, two-head Chapter 5 trace."""

    a12 = (f(1, 2), f(3, 4))
    a21 = (f(3, 4), f(1, 2))
    a33 = (f(3, 4), f(3, 4))
    items = (
        _item(E1, E1, (f(1), f(0)), a21, f(1, 2)),
        _item(E2, E2, (f(0), f(1)), a12, f(1, 2)),
        _item(E2, E2, (f(0), f(1)), a33, f(1, 2)),
        _item(E1, E1, (f(1), f(0)), a21, f(3, 4)),
        _item(Q34, E1, (f(1, 2), f(1, 2)), a12, f(1, 2)),
        _item(Q4M3, E2, (f(1, 2), f(-1, 2)), a33, f(1, 4)),
        _item(E1, E2, (f(1), f(-1, 2)), a21, f(1, 4)),
        _item(E2, E1, (f(1, 2), f(1)), a12, f(1, 2)),
        _item(E2, E2, (f(-1, 2), f(1)), a12, f(3, 4)),
        _item(E1, E1, (f(-1, 2), f(1)), a21, f(1, 2)),
        _item(Q4M3, E1, (f(0), f(1)), a21, f(1, 2)),
        _item(Q34, E2, (f(1), f(1, 2)), a12, f(3, 4)),
        _item(E1, E1, (f(1, 2), f(-1, 2)), a12, f(1, 4)),
        _item(E2, E2, (f(0), f(-1, 2)), a21, f(1, 4)),
        _item(E2, E2, (f(1), f(1, 2)), a21, f(1, 2)),
        _item(E1, E1, (f(1, 2), f(1, 2)), a12, f(1, 2)),
        _item(Q34, E1, (f(-1, 2), f(1, 2)), a12, f(3, 4)),
        _item(Q4M3, E2, (f(1), f(-1, 2)), a21, f(1, 2)),
        _item(E1, E2, (f(1), f(0)), a21, f(1, 2)),
        _item(E2, E1, (f(-1, 2), f(1)), a12, f(3, 4)),
    )
    return tuple(zip(items[::2], items[1::2], strict=True))


def decode_inputs() -> tuple[KdaInput, KdaInput]:
    """Return the frozen recurrent update for accepted answer token 11."""

    return (
        _item(E2, E1, (f(1, 2), f(1)), (f(3, 4), f(3, 4)), f(1, 2)),
        _item(E1, E2, (f(1), f(1, 2)), (f(1, 2), f(3, 4)), f(1, 2)),
    )


def one_step_example() -> tuple[Matrix, KdaInput, StepResult]:
    """Return M4.1 and all exact retain-correct-read intermediates."""

    state = ((f(1), f(2)), (f(3), f(4)))
    item = _item(Q34, E1, (f(2), f(-1)), (f(1, 2), f(1, 4)), f(1, 2))
    return state, item, kda_step(state, item)


def repeated_key_example() -> Trace:
    """Return M4.2 with the strictly interior gates used in the outline."""

    first = _item(E1, E1, (f(1), f(0)), (f(3, 4), f(3, 4)), f(1, 2))
    second = _item(E2, E2, (f(0), f(1)), (f(3, 4), f(3, 4)), f(1, 2))
    third = _item(E1, E1, (f(0), f(1)), (f(1, 2), f(3, 4)), f(1, 2))
    return run_recurrent(((first,), (second,), (third,)))


def as_float_input(item: KdaInput) -> KdaInput:
    """Convert one exact input to an equivalent binary-float input."""

    return KdaInput(
        q=tuple(float(value) for value in item.q),
        k=tuple(float(value) for value in item.k),
        v=tuple(float(value) for value in item.v),
        alpha=tuple(float(value) for value in item.alpha),
        beta=float(item.beta),
    )


def as_float_sequence(
    sequence: Sequence[Sequence[KdaInput]],
) -> tuple[tuple[KdaInput, ...], ...]:
    """Convert every exact KDA input in a sequence to binary floats."""
    return tuple(
        tuple(as_float_input(item) for item in position) for position in sequence
    )


def replace_value(
    sequence: Sequence[Sequence[KdaInput]],
    position: int,
    head: int,
    coordinate: int,
    value: float,
) -> tuple[tuple[KdaInput, ...], ...]:
    """Return a float sequence with one value coordinate replaced."""

    copied = [list(row) for row in sequence]
    item = copied[position][head]
    values = list(item.v)
    values[coordinate] = value
    copied[position][head] = replace(item, v=tuple(values))
    return tuple(tuple(row) for row in copied)


def mutate_suffix_values(
    sequence: Sequence[Sequence[KdaInput]], start: int, delta: float
) -> tuple[tuple[KdaInput, ...], ...]:
    """Perturb only value vectors at and after ``start``."""

    copied: list[tuple[KdaInput, ...]] = []
    for index, position in enumerate(sequence):
        if index < start:
            copied.append(tuple(position))
            continue
        copied.append(
            tuple(
                replace(item, v=tuple(float(value) + delta for value in item.v))
                for item in position
            )
        )
    return tuple(copied)


def flatten_outputs(trace: Trace) -> tuple[float, ...]:
    """Flatten every output vector in a recurrent trace."""
    return tuple(
        float(value)
        for position in trace.outputs
        for head in position
        for value in head
    )


def max_abs_difference(left: Iterable[Number], right: Iterable[Number]) -> float:
    """Return the largest elementwise absolute difference."""
    pairs = tuple(zip(left, right, strict=True))
    return max((abs(float(a) - float(b)) for a, b in pairs), default=0.0)


def scalar_objective(trace: Trace) -> float:
    """Smooth scalar objective used only for finite-difference comparison."""

    values = flatten_outputs(trace)
    return sum((index + 1) * value * value for index, value in enumerate(values))


def finite_difference_value_gradients(
    evaluator: Callable[[Sequence[Sequence[KdaInput]]], Trace],
    sequence: Sequence[Sequence[KdaInput]],
    step: float = 1e-6,
) -> tuple[float, ...]:
    """Differentiate the toy objective with respect to every value coordinate."""

    gradients: list[float] = []
    for position, heads in enumerate(sequence):
        for head, item in enumerate(heads):
            for coordinate, original in enumerate(item.v):
                plus = replace_value(
                    sequence, position, head, coordinate, float(original) + step
                )
                minus = replace_value(
                    sequence, position, head, coordinate, float(original) - step
                )
                gradient = (
                    scalar_objective(evaluator(plus))
                    - scalar_objective(evaluator(minus))
                ) / (2 * step)
                gradients.append(gradient)
    return tuple(gradients)


def logical_state_elements(
    batch: int, heads: int, key_width: int, value_width: int
) -> int:
    """Return the product of all logical recurrent-state dimensions."""
    if min(batch, heads, key_width, value_width) <= 0:
        raise ValueError("logical-state dimensions must be positive")
    return batch * heads * key_width * value_width


def kda_chunk_flops(length: int, chunk_size: int, head_width: int) -> int:
    """Evaluate the equal-width operator formula reported by Kimi Linear."""

    if min(length, chunk_size, head_width) <= 0:
        raise ValueError("FLOP arguments must be positive")
    return (
        6 * length * head_width * head_width
        + 3 * length * chunk_size * head_width
        + length * chunk_size * chunk_size
    )


def full_attention_flops(length: int, head_width: int) -> int:
    """Return the dense-attention operation estimate."""
    if min(length, head_width) <= 0:
        raise ValueError("FLOP arguments must be positive")
    return 2 * length * length * head_width


def formal_crossover(chunk_size: int, head_width: int) -> Fraction:
    """Solve the two reported operator formulas for their positive crossover."""

    numerator = (
        6 * head_width * head_width
        + 3 * chunk_size * head_width
        + chunk_size * chunk_size
    )
    return Fraction(numerator, 2 * head_width)
