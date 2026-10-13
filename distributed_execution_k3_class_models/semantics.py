"""Exact tensor, context, expert, and gradient semantics for Chapter 37."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence, TypedDict

Scalar = Fraction
Vector = tuple[Scalar, ...]
Matrix = tuple[tuple[Scalar, ...], ...]
ZERO = Fraction(0)


class TensorParallelResult(TypedDict):
    """Exact intermediate vectors and outputs of the tensor-parallel fixture."""

    preactivation: Vector
    activation: Vector
    local_activations: tuple[Vector, ...]
    partials: tuple[Vector, ...]
    dense: Vector
    reduced: Vector


def f(numerator: int, denominator: int = 1) -> Scalar:
    """Construct one exact scalar."""
    return Fraction(numerator, denominator)


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    """Multiply a rectangular exact matrix by a vector."""
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix and vector shapes do not match")
    return tuple(
        sum((a * b for a, b in zip(row, vector, strict=True)), f(0)) for row in matrix
    )


def relu(vector: Vector) -> Vector:
    """Apply exact elementwise ReLU."""
    return tuple(max(f(0), value) for value in vector)


def add_vectors(vectors: Sequence[Vector]) -> Vector:
    """Add equally shaped exact vectors."""
    if not vectors or len({len(vector) for vector in vectors}) != 1:
        raise ValueError("one or more equally shaped vectors are required")
    return tuple(
        sum((vector[index] for vector in vectors), f(0))
        for index in range(len(vectors[0]))
    )


def tp_fixture() -> tuple[Vector, Matrix, Matrix]:
    """Return the matrices used in the manuscript and Exercise 3."""
    x = (f(1), f(-1))
    first = ((f(1), f(2)), (f(3), f(4)), (f(-1), f(1)), (f(2), f(0)))
    second = ((f(1), f(0), f(2), f(-1)), (f(0), f(1), f(1), f(2)))
    return x, first, second


def tp_dense_and_sharded() -> TensorParallelResult:
    """Evaluate the compatible two-layer TP split exactly."""
    x, first, second = tp_fixture()
    preactivation = matvec(first, x)
    activation = relu(preactivation)
    dense = matvec(second, activation)
    local_activations = (activation[:2], activation[2:])
    second_shards = (
        tuple(tuple(row[:2]) for row in second),
        tuple(tuple(row[2:]) for row in second),
    )
    partials = tuple(
        matvec(shard, local)
        for shard, local in zip(second_shards, local_activations, strict=True)
    )
    return {
        "preactivation": preactivation,
        "activation": activation,
        "local_activations": local_activations,
        "partials": partials,
        "dense": dense,
        "reduced": add_vectors(partials),
    }


def wrong_nonlinear_split() -> Vector:
    """Apply ReLU before summing incompatible input-column partials."""
    x, first, _ = tp_fixture()
    partial_columns = tuple(
        tuple(row[column] * x[column] for row in first) for column in range(2)
    )
    return add_vectors(tuple(relu(partial) for partial in partial_columns))


@dataclass(frozen=True)
class AffineFragment:
    """One ordered scalar affine state transition ``s -> multiplier*s+shift``."""

    segment: int
    multiplier: Scalar
    shift: Scalar

    def apply(self, state: Scalar) -> Scalar:
        return self.multiplier * state + self.shift


def compose_fragments(fragments: Sequence[AffineFragment]) -> AffineFragment:
    """Compose fragments in document order."""
    if not fragments:
        return AffineFragment(-1, f(1), f(0))
    multiplier = f(1)
    shift = f(0)
    for fragment in fragments:
        shift = fragment.multiplier * shift + fragment.shift
        multiplier = fragment.multiplier * multiplier
    return AffineFragment(fragments[-1].segment, multiplier, shift)


def kda_fixture(incoming: Scalar = ZERO) -> dict[str, Scalar]:
    """Evaluate correct and reversed affine-fragment order."""
    fragments = (
        AffineFragment(0, f(2), f(1)),
        AffineFragment(1, f(3), f(-2)),
    )
    ordered = compose_fragments(fragments).apply(incoming)
    reversed_value = compose_fragments(tuple(reversed(fragments))).apply(incoming)
    return {"incoming": incoming, "ordered": ordered, "reversed": reversed_value}


@dataclass(frozen=True, order=True)
class Route:
    """One immutable token-to-expert route record."""

    origin_rank: int
    token: str
    slot: int
    expert: int
    weight: Scalar

    @property
    def identity(self) -> tuple[int, str, int]:
        return self.origin_rank, self.token, self.slot


def canonical_routes() -> tuple[Route, ...]:
    """Return eight routes with distinct weights and counts ``(3,2,2,1)``."""
    records = (
        Route(0, "a1", 0, 0, f(1, 9)),
        Route(0, "a1", 1, 1, f(8, 9)),
        Route(0, "a2", 0, 0, f(2, 9)),
        Route(0, "a2", 1, 2, f(7, 9)),
        Route(1, "b1", 0, 0, f(3, 9)),
        Route(1, "b1", 1, 2, f(6, 9)),
        Route(1, "b2", 0, 1, f(4, 9)),
        Route(1, "b2", 1, 3, f(5, 9)),
    )
    validate_routes(records)
    return records


def validate_routes(routes: Sequence[Route]) -> None:
    """Require unique identities, Top-2 completeness, and normalized weights."""
    if len({route.identity for route in routes}) != len(routes):
        raise ValueError("route identities must be unique")
    grouped: dict[tuple[int, str], list[Route]] = {}
    for route in routes:
        if route.expert not in range(4) or route.weight <= 0:
            raise ValueError("route field outside fixture domain")
        grouped.setdefault((route.origin_rank, route.token), []).append(route)
    for records in grouped.values():
        if len(records) != 2 or {route.slot for route in records} != {0, 1}:
            raise ValueError("each token must have two complete route slots")
        if len({route.expert for route in records}) != 2:
            raise ValueError("one token must select distinct experts")
        if sum((route.weight for route in records), f(0)) != 1:
            raise ValueError("route weights for one token must sum to one")


def semantic_counts(routes: Iterable[Route]) -> tuple[int, ...]:
    """Count immutable semantic routes assigned to each expert."""
    counts = [0, 0, 0, 0]
    for route in routes:
        counts[route.expert] += 1
    return tuple(counts)


def fixed_destinations(routes: Sequence[Route]) -> tuple[int, ...]:
    """Use homes ``{E0,E1}->0`` and ``{E2,E3}->1``."""
    return tuple(0 if route.expert < 2 else 1 for route in routes)


def balanced_destinations(routes: Sequence[Route]) -> tuple[int, ...]:
    """Move the ``b1/E0`` route to a fresh temporary E0 copy on rank 1."""
    destinations = list(fixed_destinations(routes))
    migrated = next(
        index
        for index, route in enumerate(routes)
        if route.token == "b1" and route.expert == 0
    )
    destinations[migrated] = 1
    return tuple(destinations)


def execution_loads(destinations: Sequence[int]) -> tuple[int, int]:
    """Count routed executions placed on each rank in the two-rank fixture."""
    return destinations.count(0), destinations.count(1)


def validate_placement(routes: Sequence[Route], destinations: Sequence[int]) -> None:
    """Validate placement without allowing it to mutate route semantics."""
    validate_routes(routes)
    if len(routes) != len(destinations) or any(
        rank not in {0, 1} for rank in destinations
    ):
        raise ValueError("each route needs one valid execution rank")


def expert_gradients(
    routes: Sequence[Route], destinations: Sequence[int]
) -> tuple[Scalar, ...]:
    """Repatriate every execution contribution to its semantic expert once."""
    validate_placement(routes, destinations)
    gradients = [f(0), f(0), f(0), f(0)]
    for route, _execution_rank in zip(routes, destinations, strict=True):
        gradients[route.expert] += route.weight
    return tuple(gradients)


def parameter_reductions(convention: str = "sum") -> dict[str, Scalar]:
    """Reduce the scalar fixture over class-specific replica sets."""
    if convention not in {"sum", "mean"}:
        raise ValueError("convention must be sum or mean")
    dense = (f(1), f(2), f(3), f(4))
    routed_e0 = (f(5), f(7))
    divisor_dense = len(dense) if convention == "mean" else 1
    divisor_routed = len(routed_e0) if convention == "mean" else 1
    return {
        "dense": sum(dense, f(0)) / divisor_dense,
        "routed_e0": sum(routed_e0, f(0)) / divisor_routed,
    }
