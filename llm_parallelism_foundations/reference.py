"""Exact examples for DP, TP, EP, CP, and a four-rank composition."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction as F

from distributed_execution_k3_class_models.semantics import (
    kda_fixture,
    matvec,
    relu,
    tp_fixture,
    wrong_nonlinear_split,
)

from .collectives import Vector, all_gather, all_reduce, all_to_all, group_members


def mean_gradient(
    local_means: Mapping[int, Vector], masses: Mapping[int, int], group: Sequence[int]
) -> Vector:
    """Differentiate the mass-weighted mean of the stated local objectives."""
    group_members(group, local_means)
    group_members(group, masses)
    if any(masses[r] <= 0 for r in group):
        raise ValueError("each local objective must have positive mass")
    total = sum(masses.values())
    numerators = {r: tuple(masses[r] * g for g in local_means[r]) for r in group}
    return tuple(g / total for g in all_reduce(numerators, group)[group[0]])


def sgd(weights: Vector, gradient: Vector, rate: F = F(1, 10)) -> Vector:
    """One coordinatewise update after the complete gradient is available."""
    return tuple(w - rate * g for w, g in zip(weights, gradient, strict=True))


def tp_forward_backward(
    x: Vector, upstream: Vector = (F(1), F(1))
) -> dict[str, tuple[Vector, ...] | Vector]:
    """Trace the compatible TP partition at an explicitly selected input."""
    _, first, second = tp_fixture()
    z = matvec(first, x)
    local = (relu(z[:2]), relu(z[2:]))
    partials = tuple(
        matvec(tuple(row[j : j + 2] for row in second), local[j // 2]) for j in (0, 2)
    )
    output = all_reduce(dict(enumerate(partials)), (0, 1))[0]
    dh = tuple(
        sum((row[j] * dy for row, dy in zip(second, upstream, strict=True)), F(0))
        for j in range(4)
    )
    # At z=0 this selects multiplier zero, not an ordinary derivative.
    mask = tuple(F(value > 0) for value in z)
    dz = tuple(
        gradient * multiplier for gradient, multiplier in zip(dh, mask, strict=True)
    )
    input_partials = tuple(
        tuple(
            sum((first[j][i] * dz[j] for j in range(start, start + 2)), F(0))
            for i in range(2)
        )
        for start in (0, 2)
    )
    return {
        "preactivation": (z[:2], z[2:]),
        "local": local,
        "partials": partials,
        "output": output,
        "hidden_gradient": (dh[:2], dh[2:]),
        "relu_multiplier": (mask[:2], mask[2:]),
        "preactivation_gradient": (dz[:2], dz[2:]),
        "input_partials": input_partials,
        "input_gradient": all_reduce(dict(enumerate(input_partials)), (0, 1))[0],
    }


def tp_both_active() -> dict[str, tuple[Vector, ...] | Vector]:
    """Case A; retain the existing entry point for callers of this fixture."""
    return tp_forward_backward((F(1), F(1)))


@dataclass(frozen=True)
class Route:
    """A selected token/expert pair retains its origin and router weight."""

    token: str
    origin: int
    expert: int
    weight: F
    value: F
    ordinal: int = 0


def routes() -> tuple[Route, ...]:
    """Four selected routes, two of which stay at their origin."""
    return (
        Route("A", 0, 1, F(1, 4), F(2)),
        Route("A", 0, 3, F(3, 4), F(2), 1),
        Route("B", 1, 2, F(1, 2), F(1)),
        Route("B", 1, 4, F(1, 2), F(1), 1),
    )


def expert_home(expert: int) -> int:
    """Keep Chapter 3's one-based expert identifiers."""
    if expert not in (1, 2, 3, 4):
        raise ValueError("unknown expert")
    return (expert - 1) // 2


def combine(
    received: Sequence[tuple[Route, F]], expected: Sequence[Route]
) -> dict[tuple[int, str], F]:
    """Restore exactly one weighted contribution for every selected route."""
    if len(received) != len(expected) or {r for r, _ in received} != set(expected):
        raise ValueError("missing, duplicated, or misidentified route")
    result: dict[tuple[int, str], F] = {}
    for route, output in received:
        key = (route.origin, route.token)
        result[key] = result.get(key, F(0)) + route.weight * output
    return result


def expert_round_trip() -> dict[tuple[int, str], F]:
    """The constructed expert e multiplies its scalar input by e."""
    selected = routes()
    group = (0, 1)
    dispatch = {
        origin: {
            home: [
                r
                for r in selected
                if r.origin == origin and expert_home(r.expert) == home
            ]
            for home in group
        }
        for origin in group
    }
    destinations = all_to_all(dispatch, group)
    returns = {
        home: {
            origin: [
                (r, r.expert * r.value)
                for r in destinations[home]
                if r.origin == origin
            ]
            for origin in group
        }
        for home in group
    }
    at_origins = all_to_all(returns, group)
    returned = [item for origin in group for item in at_origins[origin]]
    return combine(returned, selected)


def causal_prefix_means(values: Vector) -> Vector:
    """Softmax attention with zero logits: each allowed key has equal weight."""
    return tuple(sum(values[: j + 1], F(0)) / (j + 1) for j in range(len(values)))


def attention_shards(
    documents: Mapping[str, Vector], pieces: Sequence[tuple[str, tuple[int, ...]]]
) -> dict[tuple[str, int], F]:
    """All-gather per document, then evaluate only locally owned causal queries."""
    keys = [(doc, pos) for doc, positions in pieces for pos in positions]
    expected = {
        (doc, j) for doc, values in documents.items() for j in range(len(values))
    }
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("query ownership does not cover each document position once")
    result = {}
    for doc, values in documents.items():
        local = {
            rank: tuple((pos, values[pos]) for pos in positions)
            for rank, (piece_doc, positions) in enumerate(pieces)
            if piece_doc == doc
        }
        if not local:
            continue
        gathered = all_gather(local, tuple(local))
        for rank, queries in local.items():
            for pos, _ in queries:
                allowed = [value for key_pos, value in gathered[rank] if key_pos <= pos]
                result[doc, pos] = sum(allowed, F(0)) / len(allowed)
    return result


COORDINATES = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def check_axis_group(group: Sequence[int], axis: str) -> None:
    """Equal group sizes do not make a TP and DP group interchangeable."""
    if axis not in {"TP", "DP"} or len(group) != 2 or len(set(group)) != 2:
        raise ValueError("expected two distinct ranks on a declared axis")
    coordinates = [COORDINATES[r] for r in group]
    fixed, varying = (0, 1) if axis == "TP" else (1, 0)
    if len({c[fixed] for c in coordinates}) != 1 or [
        c[varying] for c in coordinates
    ] != [0, 1]:
        raise ValueError("group has the wrong logical identities or order")


def four_rank_forward() -> dict[int, Vector]:
    """Reconstruct the 33.4 forward result independently inside each TP replica."""
    partials = ((F(3), F(7)), (F(-2), F(4)))
    outputs = {}
    for group in ((0, 1), (2, 3)):
        check_axis_group(group, "TP")
        outputs.update(
            all_reduce({r: partials[COORDINATES[r][1]] for r in group}, group)
        )
    return outputs


def four_rank_step() -> dict[int, Vector]:
    """Update matching shards across DP in the isolated two-parameter witness."""
    weights = {0: (F(10),), 1: (F(20),), 2: (F(10),), 3: (F(20),)}
    gradients = {0: (F(1),), 1: (F(3),), 2: (F(5),), 3: (F(7),)}
    for group in ((0, 1), (2, 3)):
        check_axis_group(group, "TP")
        assert all_gather({r: weights[r] for r in group}, group)[group[0]] == (10, 20)
    updated = {}
    for group in ((0, 2), (1, 3)):
        check_axis_group(group, "DP")
        gradient = mean_gradient(
            {r: gradients[r] for r in group}, dict.fromkeys(group, 1), group
        )
        updated.update({r: sgd(weights[r], gradient) for r in group})
    return updated


def main() -> None:
    """Print the numerical anchors used in the prose."""
    print("TP Case A, compatible split at (1,1):", tp_both_active())
    print("TP Case B, compatible split at (1,-1):", tp_forward_backward((F(1), F(-1))))
    print(
        "TP Case C, premature ReLU on input-column partials:", wrong_nonlinear_split()
    )
    print("EP:", expert_round_trip())
    print("CP:", causal_prefix_means(tuple(map(F, range(1, 9)))))
    print("Affine order:", kda_fixture())
    print("TP x DP forward:", four_rank_forward())
    print("TP x DP:", four_rank_step())


if __name__ == "__main__":
    main()
