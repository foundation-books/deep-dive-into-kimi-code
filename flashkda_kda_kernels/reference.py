"""Dependency-free semantic oracles for Chapter 33.

The module models exact KDA state transitions, affine segment composition,
packed sequence boundaries, layout conversion, dispatch, and recovery.  It is
not a CUDA kernel or a performance model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

Number = int | Fraction
Vector = tuple[Number, ...]
Matrix = tuple[tuple[Number, ...], ...]
StateSet = tuple[tuple[Matrix, ...], ...]
Summary = tuple[Matrix, Matrix]

PACK_LENGTHS = (19, 33, 7)
PACK_OFFSETS = (0, 19, 52, 59)
PACK_CHUNK_SIZE = 16
PACK_HEADS = 2
PACK_KEY_WIDTH = 3
PACK_VALUE_WIDTH = 2


@dataclass(frozen=True)
class Token:
    """One head's projected data for the post-update KDA recurrence."""

    q: Vector
    k: Vector
    v: Vector
    alpha: Vector
    beta: Number


@dataclass(frozen=True)
class Step:
    """All semantic outputs of one token update."""

    transition: Matrix
    write: Matrix
    state: Matrix
    output: Vector


@dataclass(frozen=True)
class Workload:
    """Versioned dispatch coordinates used by the teaching policy."""

    phase: str
    gradient_enabled: bool
    dtype: str
    key_width: int
    value_width: int
    state_v_first: bool
    safe_gate: bool
    fused_gate_inputs: bool
    device_sm: int
    context_parallel: bool = False
    intermediate_states: bool = False


def f(numerator: int, denominator: int = 1) -> Fraction:
    """Construct an exact fraction from integer terms."""
    return Fraction(numerator, denominator)


def zero(rows: int, columns: int) -> Matrix:
    """Return a zero matrix with the requested shape."""
    return tuple(tuple(f(0) for _ in range(columns)) for _ in range(rows))


def identity(width: int) -> Matrix:
    """Return an identity matrix with the requested width."""
    return tuple(
        tuple(f(1) if row == column else f(0) for column in range(width))
        for row in range(width)
    )


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    """Add two same-shaped matrices."""
    return tuple(
        tuple(a + b for a, b in zip(lrow, rrow, strict=True))
        for lrow, rrow in zip(left, right, strict=True)
    )


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two compatible matrices."""
    right_t = tuple(zip(*right, strict=True))
    return tuple(
        tuple(
            sum(a * b for a, b in zip(row, column, strict=True)) for column in right_t
        )
        for row in left
    )


def outer(left: Vector, right: Vector) -> Matrix:
    """Return the outer product of two vectors."""
    return tuple(tuple(a * b for b in right) for a in left)


def scale(value: Number, matrix: Matrix) -> Matrix:
    """Scale every value by the supplied factor."""
    return tuple(tuple(value * item for item in row) for row in matrix)


def transpose(matrix: Matrix) -> Matrix:
    """Return the matrix transpose."""
    return tuple(tuple(column) for column in zip(*matrix, strict=True))


def matvec_transpose(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a transposed matrix by a vector."""
    return tuple(
        sum(a * b for a, b in zip(column, vector, strict=True))
        for column in zip(*matrix, strict=True)
    )


def validate_token(state: Matrix, token: Token) -> None:
    """Validate the token contract."""
    key_width = len(token.k)
    value_width = len(token.v)
    if key_width == 0 or value_width == 0:
        raise ValueError("KDA widths must be positive")
    if len(token.q) != key_width or len(token.alpha) != key_width:
        raise ValueError("q, k, and alpha must share the key width")
    if len(state) != key_width or any(len(row) != value_width for row in state):
        raise ValueError("state must have shape [key_width, value_width]")
    if not all(0 <= item <= 1 for item in token.alpha):
        raise ValueError("retention factors must lie in [0, 1]")
    if not 0 <= token.beta <= 1:
        raise ValueError("beta must lie in [0, 1]")


def token_summary(token: Token) -> Summary:
    """Return M,E for S_new = M S + E."""

    width = len(token.k)
    diagonal = tuple(
        tuple(token.alpha[row] if row == column else f(0) for column in range(width))
        for row in range(width)
    )
    erase = matrix_add(identity(width), scale(-token.beta, outer(token.k, token.k)))
    return matrix_multiply(erase, diagonal), scale(token.beta, outer(token.k, token.v))


def kda_step(state: Matrix, token: Token) -> Step:
    """Update the state and perform the chapter's post-update read."""

    validate_token(state, token)
    transition, write = token_summary(token)
    updated = matrix_add(matrix_multiply(transition, state), write)
    return Step(transition, write, updated, matvec_transpose(updated, token.q))


def compose(later: Summary, earlier: Summary) -> Summary:
    """Compose summaries in token order: later after earlier."""

    later_phi, later_psi = later
    earlier_phi, earlier_psi = earlier
    return (
        matrix_multiply(later_phi, earlier_phi),
        matrix_add(matrix_multiply(later_phi, earlier_psi), later_psi),
    )


def apply_summary(summary: Summary, state: Matrix) -> Matrix:
    """Apply a segment summary to a recurrent state."""
    phi, psi = summary
    return matrix_add(matrix_multiply(phi, state), psi)


def segment_summary(tokens: Sequence[Token], value_width: int | None = None) -> Summary:
    """Compose tokens into one affine segment summary."""
    if not tokens:
        if value_width is None:
            raise ValueError("empty summaries require an explicit value width")
        return identity(0), zero(0, value_width)
    result = (identity(len(tokens[0].k)), zero(len(tokens[0].k), len(tokens[0].v)))
    for token in tokens:
        result = compose(token_summary(token), result)
    return result


def run_recurrent(
    tokens: Sequence[Token], initial_state: Matrix
) -> tuple[tuple[Step, ...], Matrix]:
    """Run the recurrent reference trace from an initial state."""
    state = initial_state
    steps: list[Step] = []
    for token in tokens:
        current = kda_step(state, token)
        steps.append(current)
        state = current.state
    return tuple(steps), state


def cumulative_offsets(lengths: Sequence[int]) -> tuple[int, ...]:
    """Return exclusive cumulative offsets for packed lengths."""
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("packed lengths must be nonempty and positive")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def packed_chunks(
    lengths: Sequence[int], chunk_size: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return half-open local chunk bounds without crossing a request."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return tuple(
        tuple(
            (start, min(start + chunk_size, length))
            for start in range(0, length, chunk_size)
        )
        for length in lengths
    )


def make_tokens(length: int, head: int) -> tuple[Token, ...]:
    """Generate fixed dyadic projected inputs for the running example."""

    keys = ((f(1), f(0), f(0)), (f(0), f(1), f(0)), (f(0), f(0), f(1)))
    tokens: list[Token] = []
    for index in range(length):
        key = keys[(index + head) % len(keys)]
        q = keys[(2 * index + head) % len(keys)]
        value = (f((index + head) % 5 - 2, 4), f((2 * index + head) % 7 - 3, 4))
        alpha = (f(3, 4), f(1, 2), f(7, 8))
        beta = (f(1, 4), f(1, 2), f(3, 4))[(index + 2 * head) % 3]
        tokens.append(Token(q, key, value, alpha, beta))
    return tuple(tokens)


def running_initial_states() -> StateSet:
    """Return the initial recurrent states for packed requests A, B, and C."""
    a = (
        ((f(1, 2), f(-1, 4)), (f(0), f(1, 2)), (f(1, 4), f(0))),
        ((f(-1, 4), f(1, 2)), (f(1, 4), f(1, 4)), (f(0), f(-1, 2))),
    )
    z = tuple(zero(PACK_KEY_WIDTH, PACK_VALUE_WIDTH) for _ in range(PACK_HEADS))
    return a, z, z


def run_pack32() -> tuple[StateSet, tuple[tuple[tuple[Vector, ...], ...], ...]]:
    """Evaluate A, B, and C independently despite physical concatenation."""

    finals: list[tuple[Matrix, ...]] = []
    outputs: list[tuple[tuple[Vector, ...], ...]] = []
    for request, length in enumerate(PACK_LENGTHS):
        head_states: list[Matrix] = []
        head_outputs: list[tuple[Vector, ...]] = []
        for head in range(PACK_HEADS):
            steps, final = run_recurrent(
                make_tokens(length, head), running_initial_states()[request][head]
            )
            head_states.append(final)
            head_outputs.append(tuple(step.output for step in steps))
        finals.append(tuple(head_states))
        outputs.append(tuple(head_outputs))
    return tuple(finals), tuple(outputs)


def logical_to_v_first(states: StateSet) -> StateSet:
    """Convert logical states to value-major storage order."""
    return tuple(tuple(transpose(head) for head in request) for request in states)


def v_first_to_logical(states: StateSet) -> StateSet:
    """Convert value-major states to logical storage order."""
    return tuple(tuple(transpose(head) for head in request) for request in states)


def exclusive_scan(
    summaries: Sequence[Summary], initial_state: Matrix
) -> tuple[Matrix, ...]:
    """Return the state entering every ordered segment."""

    states = [initial_state]
    for summary in summaries[:-1]:
        states.append(apply_summary(summary, states[-1]))
    return tuple(states)


def m32_composition_example() -> tuple[Summary, Matrix, Matrix]:
    """Return M32's composed summary plus correct and naive final states."""
    phi_a = ((f(1, 2), f(0)), (f(0), f(1)))
    psi_a = ((f(1), f(0)), (f(0), f(1)))
    phi_b = ((f(1), f(1, 2)), (f(0), f(1, 2)))
    psi_b = ((f(0), f(1)), (f(1), f(0)))
    initial = ((f(1), f(0)), (f(0), f(0)))
    combined = compose((phi_b, psi_b), (phi_a, psi_a))
    return combined, apply_summary(combined, initial), matrix_add(psi_a, psi_b)


def choose_schedule(workload: Workload) -> str:
    """Choose a semantics-preserving path for the pinned public envelope."""

    if workload.phase not in {"training", "prefill", "decode"}:
        raise ValueError("phase must be training, prefill, or decode")
    if workload.key_width <= 0 or workload.value_width <= 0:
        raise ValueError("head widths must be positive")
    if workload.phase == "decode":
        return "recurrent_decode"
    if workload.context_parallel:
        return "generic_kda_context_parallel"
    public_flash = (
        not workload.gradient_enabled
        and workload.dtype == "bf16"
        and workload.key_width == 128
        and workload.value_width == 128
        and workload.state_v_first
        and workload.safe_gate
        and workload.fused_gate_inputs
        and workload.device_sm >= 90
        and not workload.intermediate_states
    )
    return "public_flashkda_forward" if public_flash else "generic_chunkwise_kda"


def state_bytes(
    requests: int, heads: int, key_width: int, value_width: int, bytes_per_item: int
) -> int:
    """Return the byte footprint of the recurrent state."""
    values = (requests, heads, key_width, value_width, bytes_per_item)
    if any(value <= 0 for value in values):
        raise ValueError("state-byte coordinates must be positive")
    return requests * heads * key_width * value_width * bytes_per_item


def fragment_bytes(
    ranks: int, heads: int, key_width: int, value_width: int, bytes_per_item: int
) -> dict[str, int]:
    """Count one (Phi,Psi) fragment and an all-gather payload per rank."""

    if min(ranks, heads, key_width, value_width, bytes_per_item) <= 0:
        raise ValueError("fragment-byte coordinates must be positive")
    local = heads * (key_width * key_width + key_width * value_width) * bytes_per_item
    return {
        "local_fragment": local,
        "received_per_rank": (ranks - 1) * local,
        "materialized_per_rank": ranks * local,
        "aggregate_received": ranks * (ranks - 1) * local,
    }


def recover_accepted_state(
    initial_state: Matrix, candidates: Sequence[Token], accepted: int
) -> Matrix:
    """Replay projected inputs to reconstruct a partially accepted draft state."""

    if not 0 <= accepted <= len(candidates):
        raise ValueError("accepted count must lie within the draft window")
    return run_recurrent(candidates[:accepted], initial_state)[1]


def flatten_numbers(matrices: Iterable[Matrix]) -> tuple[Number, ...]:
    """Flatten a sequence of matrices in encounter order."""
    return tuple(value for matrix in matrices for row in matrix for value in row)


def pass_status(condition: bool) -> str:
    """Map one deterministic predicate to the evidence record vocabulary."""

    return "pass" if condition else "fail"


def check_record(condition: bool, **details: object) -> dict[str, object]:
    """Build one compact deterministic-check record."""

    return {"status": pass_status(condition), **details}


def all_checks() -> dict[str, dict[str, object]]:
    """Return the ten deterministic evidence outcomes used by the campaign."""

    from .checks import all_checks as collect_checks

    return collect_checks()
