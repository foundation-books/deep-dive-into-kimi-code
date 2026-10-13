"""Ragged row-major SiTU expert derivatives; no GPU or trainer claims."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, tanh

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


def dot(a: Vector, b: Vector) -> float:
    """Contract one feature axis."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def project(x: Vector, weight: Matrix) -> Vector:
    """Apply a row-major weight matrix to one token row."""
    return tuple(
        sum(x[i] * weight[i][j] for i in range(len(x))) for j in range(len(weight[0]))
    )


def outer(x: Vector, y: Vector) -> Matrix:
    """Form a single-row contribution to a weight gradient."""
    return tuple(tuple(a * b for b in y) for a in x)


def zeros(rows: int, columns: int) -> Matrix:
    """Allocate an immutable zero matrix of the declared dimensions."""
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def add_matrix(a: Matrix, b: Matrix) -> Matrix:
    """Accumulate equal-shaped parameter-gradient contributions."""
    return tuple(
        tuple(x + y for x, y in zip(ar, br, strict=True))
        for ar, br in zip(a, b, strict=True)
    )


@dataclass(frozen=True)
class Expert:
    """Row-major gate, up, and down projection matrices for one expert."""

    gate: Matrix
    up: Matrix
    down: Matrix


@dataclass(frozen=True)
class Route:
    """Immutable token, slot, expert, and continuous execution coefficient."""

    token: int
    slot: int
    expert: int
    weight: float


@dataclass(frozen=True)
class Gradients:
    """Restored token gradients, grouped parameter gradients, and route scalars."""

    inputs: Matrix
    experts: tuple[Expert, ...]
    weights: Vector


def activation(g: float, u: float) -> tuple[float, float, float]:
    """SiTU with both cap scales fixed to one: value and two derivatives."""
    sigmoid = 1 / (1 + exp(-g))
    tg, tu = tanh(g), tanh(u)
    gate = tg * sigmoid
    gate_derivative = (1 - tg * tg) * sigmoid + tg * sigmoid * (1 - sigmoid)
    return gate * tu, gate_derivative * tu, gate * (1 - tu * tu)


def expert_forward(x: Vector, expert: Expert) -> tuple[Vector, Vector, Vector, Vector]:
    """Evaluate capped SiTU and return its gate, up, hidden, and output values."""
    gate, up = project(x, expert.gate), project(x, expert.up)
    hidden = tuple(activation(g, u)[0] for g, u in zip(gate, up, strict=True))
    return gate, up, hidden, project(hidden, expert.down)


def expert_backward(
    x: Vector, expert: Expert, p: float, dy: Vector
) -> tuple[Vector, Expert, float]:
    """Apply the route coefficient once before the complete expert pullback."""
    gate, up, hidden, output = expert_forward(x, expert)
    da = tuple(p * g for g in dy)  # establish this dependency before expert backward
    dh = tuple(dot(row, da) for row in expert.down)
    derivatives = tuple(activation(g, u) for g, u in zip(gate, up, strict=True))
    dg = tuple(g * partials[1] for g, partials in zip(dh, derivatives, strict=True))
    du = tuple(g * partials[2] for g, partials in zip(dh, derivatives, strict=True))
    dx = tuple(
        dot(g, dg) + dot(u, du) for g, u in zip(expert.gate, expert.up, strict=True)
    )
    dw = Expert(outer(x, dg), outer(x, du), outer(hidden, da))
    return dx, dw, dot(dy, output)


def zero_expert(expert: Expert) -> Expert:
    """Create parameter-gradient storage with the same matrix shapes."""
    return Expert(
        zeros(len(expert.gate), len(expert.gate[0])),
        zeros(len(expert.up), len(expert.up[0])),
        zeros(len(expert.down), len(expert.down[0])),
    )


def add_expert(a: Expert, b: Expert) -> Expert:
    """Accumulate the gate, up, and down parameter contributions."""
    return Expert(
        add_matrix(a.gate, b.gate), add_matrix(a.up, b.up), add_matrix(a.down, b.down)
    )


def forward(
    inputs: Matrix, routes: tuple[Route, ...], experts: tuple[Expert, ...]
) -> Matrix:
    """Independent token-major sum; selected route membership is held fixed."""
    rows = [[0.0] * len(experts[0].down[0]) for _ in inputs]
    for route in routes:
        output = expert_forward(inputs[route.token], experts[route.expert])[3]
        for j, value in enumerate(output):
            rows[route.token][j] += route.weight * value
    return tuple(map(tuple, rows))


def dispatch(
    routes: tuple[Route, ...], experts: int, alignment: int = 1
) -> tuple[int | None, ...]:
    """Expert-major indices; ownerless padding never enters a contraction."""
    if alignment < 1 or any(not 0 <= r.expert < experts for r in routes):
        raise ValueError("invalid grouping or alignment")
    grouped: list[int | None] = []
    for expert in range(experts):
        indices = [i for i, r in enumerate(routes) if r.expert == expert]
        grouped.extend(indices)
        grouped.extend([None] * ((-len(indices)) % alignment))
    return tuple(grouped)


def grouped_backward(
    inputs: Matrix,
    routes: tuple[Route, ...],
    experts: tuple[Expert, ...],
    upstream: Matrix,
    saved_order: tuple[int | None, ...] | None = None,
    alignment: int = 1,
) -> Gradients:
    """Reduce real route rows and restore input adjoints to token origins."""
    order = (
        dispatch(routes, len(experts), alignment)
        if saved_order is None
        else saved_order
    )
    if sorted(i for i in order if i is not None) != list(range(len(routes))):
        raise ValueError("every route must contribute exactly once")
    dx = [[0.0] * len(inputs[0]) for _ in inputs]
    dw = [zero_expert(e) for e in experts]
    dp = [0.0] * len(routes)
    for index in order:
        if index is None:
            continue
        route = routes[index]
        xgrad, wgrad, dp[index] = expert_backward(
            inputs[route.token],
            experts[route.expert],
            route.weight,
            upstream[route.token],
        )
        for j, value in enumerate(xgrad):
            dx[route.token][j] += value
        dw[route.expert] = add_expert(dw[route.expert], wgrad)
    return Gradients(tuple(map(tuple, dx)), tuple(dw), tuple(dp))


def fixture() -> tuple[Matrix, tuple[Route, ...], tuple[Expert, ...], Matrix]:
    """The twenty-route fixture at width four, plus an empty fifth expert."""
    from .reference import ORIGINS, toy_routes

    inputs = tuple(
        tuple((i + 1) * (j + 1) / 30 - 0.3 for j in range(4)) for i in range(10)
    )
    routes = tuple(
        Route(ORIGINS.index(r.origin), r.slot, r.expert - 1, float(r.weight))
        for r in toy_routes()
    )
    experts = tuple(
        Expert(
            tuple(tuple((i - j + e + 1) / 11 for j in range(2)) for i in range(4)),
            tuple(tuple((i + 2 * j - e + 1) / 13 for j in range(2)) for i in range(4)),
            tuple(tuple((i - j + e + 1) / 9 for j in range(4)) for i in range(2)),
        )
        for e in range(5)
    )
    upstream = tuple(tuple((i - j + 1) / 17 for j in range(4)) for i in range(10))
    return inputs, routes, experts, upstream
