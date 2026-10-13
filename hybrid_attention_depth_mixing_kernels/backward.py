"""Floating-point AttnRes derivatives with explicit per-token contributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

Vector = tuple[float, ...]
Rows = tuple[Vector, ...]
Sources = tuple[Rows, ...]  # token, source, feature


def dot(a: Vector, b: Vector) -> float:
    """Contract two feature vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def add(a: Vector, b: Vector) -> Vector:
    """Accumulate two contributions to the same feature vector."""
    return tuple(x + y for x, y in zip(a, b, strict=True))


def rms(v: Vector, eps: float) -> float:
    """Return the inverse root-mean-square statistic with positive epsilon."""
    return 1 / sqrt(dot(v, v) / len(v) + eps)


@dataclass(frozen=True)
class RowContext:
    """Per-token statistics and an optional retained forward mixture."""

    inverse_rms: Vector
    probabilities: Vector
    mixture: Vector | None


@dataclass(frozen=True)
class Gradients:
    """Complete adjoints plus the two source paths and shared reduction rows."""

    sources: Sources
    query: Vector
    weight: Vector
    output_weight: Vector | None
    value_paths: Sources
    score_paths: Sources
    query_partials: Rows


def mix(values: Rows, probabilities: Vector) -> Vector:
    """Sum raw source values using their source-axis probabilities."""
    return tuple(
        sum(p * v[j] for p, v in zip(probabilities, values, strict=True))
        for j in range(len(values[0]))
    )


def forward(
    sources: Sources,
    query: Vector,
    weight: Vector,
    output_weight: Vector | None = None,
    checkpoint_level: int = 0,
    eps: float = 1e-5,
) -> tuple[Rows, tuple[RowContext, ...]]:
    """Normalize keys, aggregate raw values, then optionally normalize output."""
    if checkpoint_level not in (0, 1) or not sources or eps <= 0:
        raise ValueError(
            "positive epsilon, token rows, and checkpoint level 0 or 1 required"
        )
    outputs, contexts = [], []
    for values in sources:
        inv = tuple(rms(v, eps) for v in values)
        logits = tuple(
            dot(query, tuple(w * x for w, x in zip(weight, v, strict=True))) * r
            for v, r in zip(values, inv, strict=True)
        )
        exps = tuple(exp(a - max(logits)) for a in logits)
        probs = tuple(a / sum(exps) for a in exps)
        pre = mix(values, probs)
        out = (
            pre
            if output_weight is None
            else tuple(
                x * w * rms(pre, eps) for x, w in zip(pre, output_weight, strict=True)
            )
        )
        outputs.append(out)
        contexts.append(RowContext(inv, probs, pre if checkpoint_level == 0 else None))
    return tuple(outputs), tuple(contexts)


def output_pullback(
    pre: Vector, grad: Vector, weight: Vector, eps: float
) -> tuple[Vector, Vector]:
    """Differentiate optional output RMSNorm and its learned scale."""
    inv = rms(pre, eps)
    xhat = tuple(x * inv for x in pre)
    weighted = tuple(w * g for w, g in zip(weight, grad, strict=True))
    correction = dot(xhat, weighted) / len(pre)
    return (
        tuple(inv * (g - x * correction) for g, x in zip(weighted, xhat, strict=True)),
        tuple(x * g for x, g in zip(xhat, grad, strict=True)),
    )


def source_pullback(
    values: Rows,
    query: Vector,
    weight: Vector,
    grad: Vector,
    context: RowContext,
) -> tuple[Rows, Rows, Vector]:
    """Keep direct value and indirect score contributions separately visible."""
    pre = (
        context.mixture
        if context.mixture is not None
        else mix(values, context.probabilities)
    )
    delta = dot(grad, pre)
    dqw = [0.0] * len(query)
    direct, indirect = [], []
    qw = tuple(q * w for q, w in zip(query, weight, strict=True))
    for v, inv, probability in zip(
        values, context.inverse_rms, context.probabilities, strict=True
    ):
        ds = probability * (dot(grad, v) - delta)
        logit = dot(v, qw) * inv
        direct.append(tuple(probability * g for g in grad))
        indirect.append(
            tuple(
                ds * inv * (a - x * inv * logit / len(v))
                for a, x in zip(qw, v, strict=True)
            )
        )
        for j, x in enumerate(v):
            dqw[j] += ds * x * inv
    return tuple(direct), tuple(indirect), tuple(dqw)


def backward(
    sources: Sources,
    query: Vector,
    weight: Vector,
    grads: Rows,
    contexts: tuple[RowContext, ...],
    output_weight: Vector | None = None,
    eps: float = 1e-5,
) -> Gradients:
    """Accumulate source paths locally and shared parameters across tokens."""
    direct, indirect, total, partials, output_partials = [], [], [], [], []
    for values, upstream, context in zip(sources, grads, contexts, strict=True):
        mixture_grad = upstream
        if output_weight is not None:
            pre = (
                context.mixture
                if context.mixture is not None
                else mix(values, context.probabilities)
            )
            mixture_grad, dow = output_pullback(pre, upstream, output_weight, eps)
            output_partials.append(dow)
        value, score, dqw = source_pullback(
            values, query, weight, mixture_grad, context
        )
        direct.append(value)
        indirect.append(score)
        total.append(tuple(add(a, b) for a, b in zip(value, score, strict=True)))
        partials.append(dqw)
    shared = tuple(sum(row[j] for row in partials) for j in range(len(query)))
    dq = tuple(g * w for g, w in zip(shared, weight, strict=True))
    dw = tuple(g * q for g, q in zip(shared, query, strict=True))
    ow = (
        None
        if output_weight is None
        else tuple(sum(row[j] for row in output_partials) for j in range(len(query)))
    )
    return Gradients(
        tuple(total), dq, dw, ow, tuple(direct), tuple(indirect), tuple(partials)
    )


def assemble_blocks(contributors: Rows, groups: tuple[tuple[int, ...], ...]) -> Rows:
    """Wrapper-level sums: the fused operator itself knows no Block ownership."""
    return tuple(
        tuple(
            sum(contributors[i][j] for i in group) for j in range(len(contributors[0]))
        )
        for group in groups
    )


def scatter_blocks(
    grads: Rows, groups: tuple[tuple[int, ...], ...], count: int
) -> Rows:
    """Every member of a sum receives the source adjoint; overlapping uses add."""
    return tuple(
        tuple(
            sum(g[j] for g, group in zip(grads, groups, strict=True) if i in group)
            for j in range(len(grads[0]))
        )
        for i in range(count)
    )
