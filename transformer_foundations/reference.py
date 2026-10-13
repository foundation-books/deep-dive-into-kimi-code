"""Complete one-block, two-head decoder for Chapter 1 foundations.

All weights are visible, with no training data, framework, or random seed.
The zero normalization epsilon is specific to this nonzero finite fixture.
Full-prefix attention and cached decode use separate attention loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, exp, log, pi, sin, sqrt


class Scalar:
    """A scalar value and local derivatives; graphs are rebuilt each forward pass."""

    def __init__(
        self, value: float, parents: tuple[tuple[Scalar, float], ...] = ()
    ) -> None:
        self.value = float(value)
        self.grad = 0.0
        self.parents = parents

    def __add__(self, other: Scalar | float) -> Scalar:
        rhs = scalar(other)
        return Scalar(self.value + rhs.value, ((self, 1.0), (rhs, 1.0)))

    def __radd__(self, other: Scalar | float) -> Scalar:
        return self + other

    def __neg__(self) -> Scalar:
        return self * -1.0

    def __sub__(self, other: Scalar | float) -> Scalar:
        return self + -scalar(other)

    def __mul__(self, other: Scalar | float) -> Scalar:
        rhs = scalar(other)
        return Scalar(self.value * rhs.value, ((self, rhs.value), (rhs, self.value)))

    def __rmul__(self, other: Scalar | float) -> Scalar:
        return self * other

    def __pow__(self, exponent: float) -> Scalar:
        return Scalar(
            self.value**exponent,
            ((self, exponent * self.value ** (exponent - 1.0)),),
        )

    def __truediv__(self, other: Scalar | float) -> Scalar:
        return self * scalar(other) ** -1.0

    def exp(self) -> Scalar:
        value = exp(self.value)
        return Scalar(value, ((self, value),))

    def log(self) -> Scalar:
        return Scalar(log(self.value), ((self, 1.0 / self.value),))

    def backward(self) -> None:
        """Accumulate each shared parent's contributions exactly once per edge."""
        order: list[Scalar] = []
        seen: set[Scalar] = set()
        stack = [(self, False)]
        while stack:
            node, ready = stack.pop()
            if ready:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent, _ in node.parents)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            for parent, derivative in node.parents:
                parent.grad += node.grad * derivative


def scalar(value: Scalar | float) -> Scalar:
    """Lift a constant into the same arithmetic as a differentiable scalar."""
    return value if isinstance(value, Scalar) else Scalar(value)


Vector = list[Scalar]
Matrix = list[Vector]
TOKENS = (0, 1, 2)
TARGETS = (1, 2, 3)


def linear(row: Vector, weight: Matrix) -> Vector:
    """Apply the row-vector convention xW."""
    return [
        sum((x * weight[i][j] for i, x in enumerate(row)), scalar(0.0))
        for j in range(len(weight[0]))
    ]


def rmsnorm(row: Vector, epsilon: float = 0.0) -> Vector:
    """Unit-gain RMSNorm across one token's features."""
    inverse = (sum((x * x for x in row), scalar(epsilon * len(row))) / len(row)) ** -0.5
    return [x * inverse for x in row]


def softmax(row: Vector) -> Vector:
    """Subtract a common constant to avoid overflowing exponentials."""
    peak = max(x.value for x in row)
    numerators = [(x - peak).exp() for x in row]
    denominator = sum(numerators, scalar(0.0))
    return [x / denominator for x in numerators]


def rotate(row: Vector, position: int) -> Vector:
    """Rotate each width-2 head by position*pi/2 (a teaching frequency)."""
    c, s = cos(position * pi / 2), sin(position * pi / 2)
    out: Vector = []
    for start in (0, 2):
        a, b = row[start : start + 2]
        out.extend((a * c - b * s, a * s + b * c))
    return out


@dataclass
class Cache:
    """Processed keys and values for exactly one fixed model version."""

    keys: Matrix = field(default_factory=list)
    values: Matrix = field(default_factory=list)


class Decoder:
    """Four tokens/features/FFN channels; all nine matrices start diagonal."""

    def __init__(self) -> None:
        scales = {
            "E": 1.0,
            "Q": 1.0,
            "K": 1.0,
            "V": 1.0,
            "O": 0.5,
            "G": 1.0,
            "U": 1.0,
            "D": 0.25,
            "R": 1.0,
        }
        self.weights = {
            name: [
                [Scalar(scale if i == j else 0.0) for j in range(4)] for i in range(4)
            ]
            for name, scale in scales.items()
        }

    def parameters(self) -> list[Scalar]:
        """Embedding and readout are independent, even when initially equal."""
        return [x for matrix in self.weights.values() for row in matrix for x in row]

    def project(
        self, token: int, position: int
    ) -> tuple[Vector, Vector, Vector, Vector]:
        if not 0 <= token < 4:
            raise ValueError("token must be in the four-token vocabulary")
        x = self.weights["E"][token]
        n = rmsnorm(x)
        return (
            x,
            rotate(linear(n, self.weights["Q"]), position),
            rotate(linear(n, self.weights["K"]), position),
            linear(n, self.weights["V"]),
        )

    def finish(self, x: Vector, context: Vector) -> tuple[Vector, dict[str, Vector]]:
        attention = linear(context, self.weights["O"])
        residual = [a + b for a, b in zip(x, attention, strict=True)]
        n = rmsnorm(residual)
        gate, up = linear(n, self.weights["G"]), linear(n, self.weights["U"])
        hidden = [
            g / (scalar(1.0) + (-g).exp()) * u for g, u in zip(gate, up, strict=True)
        ]
        ffn = linear(hidden, self.weights["D"])
        h = [a + b for a, b in zip(residual, ffn, strict=True)]
        final = rmsnorm(h)
        logits = linear(final, self.weights["R"])
        return logits, {
            "attention": attention,
            "residual": residual,
            "ffn": ffn,
            "h": h,
            "final": final,
        }

    def full(self, tokens: tuple[int, ...]) -> tuple[Matrix, list[dict[str, Vector]]]:
        """Materialize all projected rows, then perform causal attention."""
        rows = [self.project(token, t) for t, token in enumerate(tokens)]
        logits, traces = [], []
        for t, (x, query, _, _) in enumerate(rows):
            context: Vector = []
            for start in (0, 2):
                scores = [
                    sum(
                        (query[j] * rows[s][2][j] for j in range(start, start + 2)),
                        scalar(0.0),
                    )
                    / sqrt(2)
                    for s in range(t + 1)
                ]
                probabilities = softmax(scores)
                context.extend(
                    sum(
                        (probabilities[s] * rows[s][3][j] for s in range(t + 1)),
                        scalar(0.0),
                    )
                    for j in range(start, start + 2)
                )
            output, trace = self.finish(x, context)
            logits.append(output)
            traces.append(trace)
        return logits, traces

    def decode(self, token: int, cache: Cache) -> Vector:
        """Append one token's rotated key and value; positions never restart."""
        if len(cache.keys) != len(cache.values):
            raise ValueError("key and value prefix lengths must agree")
        x, q, k, v = self.project(token, len(cache.keys))
        cache.keys.append(k)
        cache.values.append(v)
        contexts: Vector = []
        for start in (0, 2):
            weights = softmax(
                [
                    (q[start] * key[start] + q[start + 1] * key[start + 1]) / sqrt(2)
                    for key in cache.keys
                ]
            )
            for channel in (start, start + 1):
                contexts.append(
                    sum(
                        (
                            a * value[channel]
                            for a, value in zip(weights, cache.values, strict=True)
                        ),
                        scalar(0.0),
                    )
                )
        return self.finish(x, contexts)[0]


def loss(logits: Matrix, targets: tuple[int, ...]) -> Scalar:
    """Mean next-token cross-entropy, computed by stable log-sum-exp."""
    if not targets or len(logits) != len(targets):
        raise ValueError("one target is required for every logit row")
    terms = []
    for row, target in zip(logits, targets, strict=True):
        if not 0 <= target < len(row):
            raise ValueError("target is outside the vocabulary")
        peak = max(x.value for x in row)
        normalizer = sum(((x - peak).exp() for x in row), scalar(0.0)).log() + peak
        terms.append(normalizer - row[target])
    return sum(terms, scalar(0.0)) / len(terms)


def adamw_step(
    parameters: list[Scalar],
    moments: list[tuple[float, float]],
    step: int,
    *,
    learning_rate: float = 0.01,
    decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> list[tuple[float, float]]:
    """Update all fixture matrices with decoupled decay and bias correction."""
    if step < 1 or len(parameters) != len(moments):
        raise ValueError("positive step and one moment pair per parameter required")
    updated = []
    for parameter, (old_m, old_v) in zip(parameters, moments, strict=True):
        g = parameter.grad
        m = beta1 * old_m + (1 - beta1) * g
        v = beta2 * old_v + (1 - beta2) * g * g
        direction = (m / (1 - beta1**step)) / (sqrt(v / (1 - beta2**step)) + epsilon)
        parameter.value = (
            1 - learning_rate * decay
        ) * parameter.value - learning_rate * direction
        updated.append((m, v))
    return updated
