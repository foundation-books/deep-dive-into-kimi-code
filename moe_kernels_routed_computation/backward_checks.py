"""All-coordinate differences for nonlinear ragged input/weight/route gradients."""

from dataclasses import replace
from math import isclose, isfinite

from . import backward as b


def objective(
    inputs: b.Matrix,
    routes: tuple[b.Route, ...],
    experts: tuple[b.Expert, ...],
    upstream: b.Matrix,
) -> float:
    """Evaluate a token-major weighted-output loss independently of grouping."""
    return sum(
        b.dot(y, g)
        for y, g in zip(b.forward(inputs, routes, experts), upstream, strict=True)
    )


def check_all() -> None:
    """Check all derivative families, replay, empty experts, padding, and faults."""
    inputs, routes, experts, upstream = b.fixture()
    saved = b.dispatch(routes, len(experts), 4)
    grad = b.grouped_backward(inputs, routes, experts, upstream, saved_order=saved)
    assert grad == b.grouped_backward(inputs, routes, experts, upstream, alignment=4)
    assert grad == b.grouped_backward(inputs, routes, experts, upstream)
    assert grad.experts[-1] == b.zero_expert(experts[-1])
    h = 1e-5
    for i, row in enumerate(inputs):
        for j in range(len(row)):
            losses = []
            for sign in (-1, 1):
                changed = [list(x) for x in inputs]
                changed[i][j] += sign * h
                losses.append(
                    objective(tuple(map(tuple, changed)), routes, experts, upstream)
                )
            assert isclose(
                (losses[1] - losses[0]) / (2 * h),
                grad.inputs[i][j],
                rel_tol=2e-7,
                abs_tol=2e-9,
            )
    for e, expert in enumerate(experts):
        for name in ("gate", "up", "down"):
            matrix = getattr(expert, name)
            for i, row in enumerate(matrix):
                for j in range(len(row)):
                    losses = []
                    for sign in (-1, 1):
                        changed = [list(x) for x in matrix]
                        changed[i][j] += sign * h
                        parameters = list(experts)
                        parameters[e] = replace(
                            expert, **{name: tuple(map(tuple, changed))}
                        )
                        losses.append(
                            objective(inputs, routes, tuple(parameters), upstream)
                        )
                    derivative = (losses[1] - losses[0]) / (2 * h)
                    assert isfinite(derivative) and isclose(
                        derivative,
                        getattr(grad.experts[e], name)[i][j],
                        rel_tol=2e-7,
                        abs_tol=2e-9,
                    )
    for i, route in enumerate(routes):
        losses = []
        for sign in (-1, 1):
            changed_routes = list(routes)
            changed_routes[i] = replace(route, weight=route.weight + sign * h)
            losses.append(objective(inputs, tuple(changed_routes), experts, upstream))
        assert isclose(
            (losses[1] - losses[0]) / (2 * h),
            grad.weights[i],
            rel_tol=2e-7,
            abs_tol=2e-9,
        )
    wrong = list(routes)
    wrong[0] = replace(wrong[0], token=9)
    assert (
        b.grouped_backward(inputs, tuple(wrong), experts, upstream).inputs
        != grad.inputs
    )
    doubled = tuple(replace(r, weight=r.weight * r.weight) for r in routes)
    assert b.grouped_backward(inputs, doubled, experts, upstream).inputs != grad.inputs
