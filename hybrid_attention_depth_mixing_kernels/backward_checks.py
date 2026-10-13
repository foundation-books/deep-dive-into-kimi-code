"""Finite differences for source, shared-parameter, and output-norm paths."""

from math import isclose, isfinite

from . import backward as b

SOURCES: b.Sources = (
    ((1.0, -0.5), (0.2, 1.4), (-0.7, 0.3)),
    ((-0.4, 0.8), (1.3, 0.1), (0.5, -0.9)),
)
QUERY: b.Vector = (0.3, -0.2)
WEIGHT: b.Vector = (1.1, 0.7)
UPSTREAM: b.Rows = ((0.8, -0.3), (-0.2, 1.2))


def objective(
    sources: b.Sources, query: b.Vector, weight: b.Vector, ow: b.Vector | None
) -> float:
    """Evaluate an independently differentiated forward weighted-output loss."""
    outputs, _ = b.forward(sources, query, weight, ow)
    return sum(b.dot(out, grad) for out, grad in zip(outputs, UPSTREAM, strict=True))


def check_derivatives(ow: b.Vector | None) -> None:
    """Check all source and parameter coordinates with and without replay."""
    outputs, context = b.forward(SOURCES, QUERY, WEIGHT, ow)
    grads = b.backward(SOURCES, QUERY, WEIGHT, UPSTREAM, context, ow)
    replay_outputs, replay = b.forward(SOURCES, QUERY, WEIGHT, ow, 1)
    assert outputs == replay_outputs
    assert all(row.mixture is None for row in replay)
    assert grads == b.backward(SOURCES, QUERY, WEIGHT, UPSTREAM, replay, ow)
    h = 1e-5
    for row in range(2):
        for source in range(3):
            for feature in range(2):
                losses = []
                for sign in (-1, 1):
                    changed = [[list(v) for v in token] for token in SOURCES]
                    changed[row][source][feature] += sign * h
                    frozen = tuple(tuple(tuple(v) for v in token) for token in changed)
                    losses.append(objective(frozen, QUERY, WEIGHT, ow))
                observed = (losses[1] - losses[0]) / (2 * h)
                assert isfinite(observed) and isclose(
                    observed,
                    grads.sources[row][source][feature],
                    rel_tol=2e-7,
                    abs_tol=2e-9,
                )
    parameters = [(QUERY, grads.query), (WEIGHT, grads.weight)]
    if ow is not None:
        assert grads.output_weight is not None
        parameters.append((ow, grads.output_weight))
    for family, (values, analytic) in enumerate(parameters):
        for feature in range(2):
            losses = []
            for sign in (-1, 1):
                changed_param = list(values)
                changed_param[feature] += sign * h
                q, w, out_w = QUERY, WEIGHT, ow
                if family == 0:
                    q = tuple(changed_param)
                elif family == 1:
                    w = tuple(changed_param)
                else:
                    out_w = tuple(changed_param)
                losses.append(objective(SOURCES, q, w, out_w))
            assert isclose(
                (losses[1] - losses[0]) / (2 * h),
                analytic[feature],
                rel_tol=2e-7,
                abs_tol=2e-9,
            )
    assert any(
        abs(a - c) > 1e-3
        for token, correct in zip(grads.value_paths, grads.sources, strict=True)
        for v, g in zip(token, correct, strict=True)
        for a, c in zip(v, g, strict=True)
    )
    missing_row = tuple(
        x * w for x, w in zip(grads.query_partials[0], WEIGHT, strict=True)
    )
    assert max(abs(a - c) for a, c in zip(missing_row, grads.query, strict=True)) > 1e-3


def check_block_wrapper() -> None:
    """Differentiate assembled Block sources back to their original contributors."""
    contributors = ((1.0, -0.5), (0.2, 0.4), (-0.7, 0.3))
    groups = ((0, 1), (2,))  # one complete block and the current partial
    sources = (b.assemble_blocks(contributors, groups),)
    _, context = b.forward(sources, QUERY, WEIGHT)
    result = b.backward(sources, QUERY, WEIGHT, UPSTREAM[:1], context)
    restored = b.scatter_blocks(result.sources[0], groups, 3)
    assert restored[0] == restored[1]
    for row in range(3):
        for j in range(2):
            losses = []
            for sign in (-1, 1):
                changed = [list(v) for v in contributors]
                changed[row][j] += sign * 1e-5
                values = (b.assemble_blocks(tuple(map(tuple, changed)), groups),)
                output, _ = b.forward(values, QUERY, WEIGHT)
                losses.append(b.dot(output[0], UPSTREAM[0]))
            assert isclose(
                (losses[1] - losses[0]) / 2e-5,
                restored[row][j],
                rel_tol=2e-7,
                abs_tol=2e-9,
            )


def check_all() -> None:
    """Cover both output-norm modes, checkpoint levels, and Block assembly."""
    check_derivatives(None)
    check_derivatives((0.9, 1.2))
    check_block_wrapper()
