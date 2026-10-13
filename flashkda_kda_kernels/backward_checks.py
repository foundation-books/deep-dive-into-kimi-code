"""Independent differences and boundary faults for the KDA teaching derivative."""

from dataclasses import replace
from fractions import Fraction
from math import exp, isclose, sqrt

from . import backward as b
from . import reference as r


def perturb(token: r.Token, field: str, coordinate: int, delta: Fraction) -> r.Token:
    """Change one recurrence coordinate without altering route or chunk metadata."""
    if field == "beta":
        return replace(token, beta=token.beta + delta)
    vectors = {"q": token.q, "k": token.k, "v": token.v, "alpha": token.alpha}
    changed = list(vectors[field])
    changed[coordinate] += delta
    vectors[field] = tuple(changed)
    return r.Token(
        vectors["q"], vectors["k"], vectors["v"], vectors["alpha"], token.beta
    )


def check_differences() -> None:
    """Check every recurrence-ready coordinate of A, including its 16+3 tail."""
    tokens = r.make_tokens(19, 0)
    initial = r.running_initial_states()[0][0]
    dos = tuple((r.f(i % 3 + 1, 5), r.f(-1, 7)) for i in range(19))
    dfinal = ((r.f(1, 3), r.f(1, 5)), (r.f(-1, 4), r.f(2, 5)), (r.f(1, 6), r.f(-1, 8)))
    step = Fraction(1, 10000)
    for dy, dh in (
        (dos, r.zero(3, 2)),
        (tuple((r.f(0), r.f(0)) for _ in dos), dfinal),
        (dos, dfinal),
    ):
        expected = b.reverse_chunks(tokens, initial, dy, dh)
        assert expected == b.reverse_chunks(tokens, initial, dy, dh, recompute=True)
        for i, token in enumerate(tokens):
            for field in ("q", "k", "v", "alpha", "beta"):
                grads = (
                    (expected.tokens[i].beta,)
                    if field == "beta"
                    else getattr(expected.tokens[i], field)
                )
                for coordinate, grad in enumerate(grads):
                    losses = []
                    for sign in (-1, 1):
                        mutated = list(tokens)
                        mutated[i] = perturb(token, field, coordinate, sign * step)
                        losses.append(b.loss(mutated, initial, dy, dh))
                    assert (losses[1] - losses[0]) / (2 * step) == grad
        for row in range(3):
            for column in range(2):
                losses = []
                for sign in (-1, 1):
                    altered_state = [list(v) for v in initial]
                    altered_state[row][column] += sign * step
                    losses.append(
                        b.loss(tokens, tuple(map(tuple, altered_state)), dy, dh)
                    )
                assert (losses[1] - losses[0]) / (2 * step) == expected.initial[row][
                    column
                ]
    assert expected.boundaries[0][0] == 16
    assert any(v != 0 for row in expected.boundaries[0][1] for v in row)
    # Dropping the later chunk's state adjoint changes the earlier token gradients.
    broken = b.reverse_chunks(tokens[:16], initial, dos[:16], r.zero(3, 2))
    assert broken.tokens != expected.tokens[:16]
    assert broken.initial != expected.initial
    # B has its own initial state; changing A cannot change its derivative.
    btokens = r.make_tokens(7, 1)
    bgrad = b.reverse_chunks(btokens, r.zero(3, 2), dos[:7], dfinal)
    b.reverse_chunks(tokens, r.scale(2, initial), dos, dfinal)
    assert bgrad == b.reverse_chunks(btokens, r.zero(3, 2), dos[:7], dfinal)


def raw_loss(raw: tuple[float, ...]) -> float:
    """A separate smooth one-token raw-input forward map."""
    q, k = raw[:3], raw[3:6]
    qn, kn = sqrt(sum(x * x for x in q)), sqrt(sum(x * x for x in k))
    token = r.Token(
        tuple(Fraction(x / qn) for x in q),
        tuple(Fraction(x / kn) for x in k),
        (r.f(2, 5), r.f(-1, 3)),
        tuple(Fraction(exp(x)) for x in raw[6:9]),
        Fraction(1 / (1 + exp(-raw[9]))),
    )
    return float(
        b.loss(
            (token,),
            r.running_initial_states()[0][0],
            ((r.f(1), r.f(-2)),),
            r.zero(3, 2),
        )
    )


def check_raw_coordinates() -> None:
    """Compare smooth raw-input chain rules with a separate forward objective."""
    raw = (0.3, -0.4, 0.8, 0.7, 0.2, -0.5, -0.2, -0.6, -0.3, 0.4)
    qn = sqrt(sum(x * x for x in raw[:3]))
    kn = sqrt(sum(x * x for x in raw[3:6]))
    token = r.Token(
        tuple(Fraction(x / qn) for x in raw[:3]),
        tuple(Fraction(x / kn) for x in raw[3:6]),
        (r.f(2, 5), r.f(-1, 3)),
        tuple(Fraction(exp(x)) for x in raw[6:9]),
        Fraction(1 / (1 + exp(-raw[9]))),
    )
    grad, _ = b.pullback(
        r.running_initial_states()[0][0], token, (r.f(1), r.f(-2)), r.zero(3, 2)
    )
    analytic = (
        *b.l2_pullback(raw[:3], grad.q),
        *b.l2_pullback(raw[3:6], grad.k),
        *(float(g * a) for g, a in zip(grad.alpha, token.alpha, strict=True)),
        float(grad.beta * token.beta * (1 - token.beta)),
    )
    for i, expected in enumerate(analytic):
        plus, minus = list(raw), list(raw)
        plus[i] += 1e-5
        minus[i] -= 1e-5
        observed = (raw_loss(tuple(plus)) - raw_loss(tuple(minus))) / 2e-5
        assert isclose(observed, expected, rel_tol=1e-7, abs_tol=1e-9)


def check_all() -> None:
    """Run exact recurrence and smooth raw-coordinate derivative checks."""
    check_differences()
    check_raw_coordinates()
