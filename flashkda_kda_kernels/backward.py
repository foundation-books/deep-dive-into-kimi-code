"""Exact recurrence adjoints and chunk replay; no accelerator implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from . import reference as r


@dataclass(frozen=True)
class Reverse:
    """Input adjoints and states crossing logical chunk boundaries."""

    tokens: tuple[r.Token, ...]
    initial: r.Matrix
    boundaries: tuple[tuple[int, r.Matrix], ...]


def pullback(
    before: r.Matrix, token: r.Token, output_grad: r.Vector, later: r.Matrix
) -> tuple[r.Token, r.Matrix]:
    """Differentiate update-then-read, including the later state's contribution."""
    after = r.kda_step(before, token).state
    decayed = tuple(
        tuple(token.alpha[i] * x for x in row) for i, row in enumerate(before)
    )
    prediction = r.matvec_transpose(decayed, token.k)
    error = tuple(v - p for v, p in zip(token.v, prediction, strict=True))
    state_grad = r.matrix_add(later, r.outer(token.q, output_grad))
    dq = tuple(
        sum(x * g for x, g in zip(row, output_grad, strict=True)) for row in after
    )
    dv = tuple(token.beta * x for x in r.matvec_transpose(state_grad, token.k))
    dk = tuple(
        token.beta * sum(g * e for g, e in zip(row, error, strict=True))
        - sum(x * g for x, g in zip(decayed[i], dv, strict=True))
        for i, row in enumerate(state_grad)
    )
    db = sum(
        token.k[i] * sum(g * e for g, e in zip(row, error, strict=True))
        for i, row in enumerate(state_grad)
    )
    ddecayed = r.matrix_add(state_grad, r.scale(-1, r.outer(token.k, dv)))
    da = tuple(
        sum(x * g for x, g in zip(row, ddecayed[i], strict=True))
        for i, row in enumerate(before)
    )
    previous = tuple(
        tuple(token.alpha[i] * g for g in row) for i, row in enumerate(ddecayed)
    )
    return r.Token(dq, dk, dv, da, db), previous


def reverse_chunks(
    tokens: Sequence[r.Token],
    initial: r.Matrix,
    output_grads: Sequence[r.Vector],
    final_grad: r.Matrix,
    chunk_size: int = 16,
    recompute: bool = False,
) -> Reverse:
    """Save states or replay each chunk; carry its incoming adjoint to the left."""
    if len(tokens) != len(output_grads) or not tokens or chunk_size < 1:
        raise ValueError(
            "nonempty matching token/adjoint rows and positive chunks required"
        )
    steps, _ = r.run_recurrent(tokens, initial)
    starts = tuple(range(0, len(tokens), chunk_size))
    boundaries = {s: initial if s == 0 else steps[s - 1].state for s in starts}
    saved = None if recompute else (initial, *(step.state for step in steps))
    del steps
    grads: dict[int, r.Token] = {}
    incoming: list[tuple[int, r.Matrix]] = []
    later = final_grad
    for start in reversed(starts):
        stop = min(start + chunk_size, len(tokens))
        if saved is None:
            local, _ = r.run_recurrent(tokens[start:stop], boundaries[start])
            states = (boundaries[start], *(step.state for step in local))
        else:
            states = saved[start : stop + 1]
        for index in range(stop - 1, start - 1, -1):
            grads[index], later = pullback(
                states[index - start], tokens[index], output_grads[index], later
            )
        incoming.append((start, later))
    return Reverse(tuple(grads[i] for i in range(len(tokens))), later, tuple(incoming))


def loss(
    tokens: Sequence[r.Token],
    initial: r.Matrix,
    output_grads: Sequence[r.Vector],
    final_grad: r.Matrix,
) -> r.Number:
    """Independent forward loss used for centered-difference checks."""
    steps, final = r.run_recurrent(tokens, initial)
    value = sum(
        sum(x * g for x, g in zip(step.output, grad, strict=True))
        for step, grad in zip(steps, output_grads, strict=True)
    )
    return value + sum(
        x * g
        for row, grow in zip(final, final_grad, strict=True)
        for x, g in zip(row, grow, strict=True)
    )


def l2_pullback(raw: Sequence[float], grad: r.Vector) -> tuple[float, ...]:
    """Return to raw coordinates for nonzero, unclamped L2 normalization."""
    norm = sqrt(sum(x * x for x in raw))
    if norm == 0:
        raise ValueError("this smooth fixture excludes the normalization floor")
    unit = tuple(x / norm for x in raw)
    radial = sum(x * float(g) for x, g in zip(unit, grad, strict=True))
    return tuple(
        (float(g) - x * radial) / norm for x, g in zip(unit, grad, strict=True)
    )


# Each entry identifies a conceptual stage, its public tensor, and its last consumer.
# Shapes in the prose include batch/head axes; these names do not simulate launches.
STAGE_LEDGER = (
    ("reconstruct", "w,u,qg,kg,h,v_new", "state and fused WY backward"),
    ("output", "dAqk,dv", "intra-chunk and reverse state backward"),
    ("reverse state", "dh,dh0,dv", "fused WY backward and caller"),
    ("WY", "dq,dk,dv,db,dg,dAkk", "intra-chunk accumulation"),
    ("intra-chunk", "dq,dk,db,dg", "head reduction and gate pullback"),
    ("raw coordinates", "dQraw,dKraw,dZ,dZbeta,dA,dbias", "caller"),
)
