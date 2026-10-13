"""Differentiable two-coordinate CPU expert with temporary-gradient closure."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from . import reference as r
from .runtime import dispatch_buffers

Vector = tuple[Fraction, Fraction]


def add(a: Vector, b: Vector) -> Vector:
    """Sum two exact feature contributions."""
    return a[0] + b[0], a[1] + b[1]


@dataclass(frozen=True)
class Derivatives:
    """Token, route, and execution-rank parameter-gradient contributions."""

    inputs: dict[str, Vector]
    weights: dict[r.RouteId, Fraction]
    partials: dict[tuple[int, int], Vector]


def inputs(plan: r.Plan, microbatch: int) -> dict[str, Vector]:
    """Construct distinct canonical inputs for the current microbatch."""
    return {
        route.origin: (r.f(route.token + microbatch, 7), r.f(2 - route.token, 9))
        for route in plan.route_snapshot
    }


def objective(
    plan: r.Plan, rows: dict[str, Vector], parameters: tuple[Vector, ...]
) -> Fraction:
    """Token-major weighted sum of E_e(x)=(theta_e0*x0²,theta_e1*x1²)."""
    value = r.f(0)
    for route in plan.route_snapshot:
        x = rows[route.origin]
        w = parameters[route.expert - 1]
        value += route.weight * (w[0] * x[0] ** 2 - 2 * w[1] * x[1] ** 2)
    return value


def backward(
    plan: r.Plan, rows: dict[str, Vector], parameters: tuple[Vector, ...]
) -> Derivatives:
    """Dispatch upstream (1,-2), apply p once, and ignore ownerless padding."""
    dx: dict[str, Vector] = {origin: (r.f(0), r.f(0)) for origin in rows}
    dp: dict[r.RouteId, Fraction] = {}
    partials: dict[tuple[int, int], Vector] = {}
    for rank, buffer in enumerate(dispatch_buffers(plan)):
        for route in buffer:
            if route is None:
                continue
            x, w = rows[route.origin], parameters[route.expert - 1]
            da = (route.weight, -2 * route.weight)
            dx[route.origin] = add(
                dx[route.origin], (2 * x[0] * w[0] * da[0], 2 * x[1] * w[1] * da[1])
            )
            dp[route.identity] = w[0] * x[0] ** 2 - 2 * w[1] * x[1] ** 2
            key = (route.expert, rank)
            partials[key] = add(
                partials.get(key, (r.f(0), r.f(0))),
                (x[0] ** 2 * da[0], x[1] ** 2 * da[1]),
            )
    return Derivatives(dx, dp, partials)


def home_reduce(
    partials: dict[tuple[int, int], Vector], temporary_copies: int = 1
) -> dict[int, Vector]:
    """A fault multiplier exposes missing/doubled return; zero consumed slots."""
    homes = {e: (r.f(0), r.f(0)) for e in range(1, r.EXPERTS + 1)}
    for (expert, rank), value in partials.items():
        factor = 1 if r.HOME[expert - 1] == rank else temporary_copies
        homes[expert] = add(homes[expert], (factor * value[0], factor * value[1]))
    for key in partials:
        partials[key] = (r.f(0), r.f(0))
    return homes


def check_lifecycle() -> None:
    """Check two microbatches, exactly-once home return, clearing, and aliases."""
    from dataclasses import replace

    plan = r.balanced_plan()
    parameters = tuple((r.f(e + 1, 5), r.f(2 - e, 7)) for e in range(4))
    running = {e: (r.f(0), r.f(0)) for e in range(1, 5)}
    delta = r.f(1, 1000)
    for microbatch in (0, 1):
        current = replace(plan, microbatch=microbatch, epoch=microbatch + 1)
        rows = inputs(current, microbatch)
        grad = backward(current, rows, parameters)
        expected = home_reduce(dict(grad.partials))
        for e in range(4):
            for j in range(2):
                losses = []
                for sign in (-1, 1):
                    changed = [list(w) for w in parameters]
                    changed[e][j] += sign * delta
                    losses.append(
                        objective(current, rows, tuple((w[0], w[1]) for w in changed))
                    )
                assert (losses[1] - losses[0]) / (2 * delta) == expected[e + 1][j]
        for origin, row in rows.items():
            for j in range(2):
                changed_rows = dict(rows)
                plus, minus = list(row), list(row)
                plus[j] += delta
                minus[j] -= delta
                changed_rows[origin] = (plus[0], plus[1])
                high = objective(current, changed_rows, parameters)
                changed_rows[origin] = (minus[0], minus[1])
                low = objective(current, changed_rows, parameters)
                assert (high - low) / (2 * delta) == grad.inputs[origin][j]
        for i, route in enumerate(current.route_snapshot):
            losses = []
            for sign in (-1, 1):
                routes = list(current.route_snapshot)
                routes[i] = replace(route, weight=route.weight + sign * delta)
                losses.append(
                    objective(
                        replace(current, route_snapshot=tuple(routes)), rows, parameters
                    )
                )
            assert (losses[1] - losses[0]) / (2 * delta) == grad.weights[route.identity]
        assert home_reduce(dict(grad.partials), 0) != expected
        assert home_reduce(dict(grad.partials), 2) != expected
        actual = home_reduce(grad.partials)
        assert all(v == (0, 0) for v in grad.partials.values())
        for e, value in actual.items():
            running[e] = add(running[e], value)
    assert running[1] != expected[1]
    # A saved plan does not snapshot payload storage: the next call overwrites it.
    communication = [r.f(2), r.f(3)]
    view = communication
    retained = tuple(communication)
    communication[:] = [r.f(7), r.f(11)]
    assert tuple(view) != (r.f(2), r.f(3)) and retained == (r.f(2), r.f(3))
