"""Exact worksheets for the printed Chapter 37 reader trace.

These CPU models distinguish route counts, payload reuse, physical backing,
and allocation identity from payload leases. They execute no MoonEP kernel.
The original conservative epoch oracle and historical evidence IDs stay intact.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, product

from . import reference as ref


def diagnostic_outputs(routes: Sequence[ref.Route]) -> dict[str, Fraction]:
    """Evaluate each origin independently with the printed scalar diagnostic."""
    result: dict[str, Fraction] = {}
    for route in routes:
        value = route.weight * (100 * route.token + route.expert)
        result[route.origin] = result.get(route.origin, Fraction(0)) + value
    return result


def source_execution_counts(plan: ref.Plan) -> ref.Matrix:
    """Count logical routes, with sources as rows and executions as columns."""
    matrix = [[0, 0], [0, 0]]
    for route, destination in zip(plan.route_snapshot, plan.destinations, strict=True):
        matrix[route.source_rank][destination] += 1
    return tuple(tuple(row) for row in matrix)


def remote_input_pairs(plan: ref.Plan) -> frozenset[tuple[str, int]]:
    """Deduplicate inputs by origin and destination, while retaining routes."""
    return frozenset(
        (route.origin, destination)
        for route, destination in zip(
            plan.route_snapshot, plan.destinations, strict=True
        )
        if route.source_rank != destination
    )


def encode_destination(rank: int, slot: int, capacity: int, duplicate: bool) -> int:
    """Encode a logical destination independently of physical padded stride."""
    if rank < 0 or capacity <= 0 or not 0 <= slot < capacity:
        raise ValueError("destination outside logical encoding capacity")
    raw = rank * capacity + slot
    return -raw - 1 if duplicate else raw


def decode_destination(stored: int, capacity: int) -> tuple[int, int]:
    """Recover a duplicate's real destination rather than deleting its route."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    raw = -stored - 1 if stored < 0 else stored
    return divmod(raw, capacity)


def physical_backing_bytes() -> dict[str, int]:
    """Sum backing once for the independent H=8, H'=3 worksheet."""
    ledger = ref.layout_bytes(4, 2, 2, 8, 3, 2, 4)
    return {
        "weight_view": ledger["symmetric_weight_view"],
        "local_weights": ledger["owned_home_weights"]
        + ledger["local_prefetch_weights"],
        "gradient_view": ledger["symmetric_gradient_view"],
        "temporary_gradient_alias": ledger["local_reduce_storage"],
        "local_gradients": ledger["owned_home_gradients"]
        + ledger["local_reduce_storage"],
    }


@dataclass(frozen=True)
class PhysicalBinding:
    """Caller identity in a CPU model; address tags stand for actual bindings."""

    plan_digest: str
    allocation_generation: int
    address_tags: tuple[str, ...]
    parameter_revision: int
    homes: tuple[int, ...]
    capacities: tuple[int, ...]
    payload_width: int
    dtype_policy: str


def validate_replay(saved: PhysicalBinding, current: PhysicalBinding) -> None:
    """Compare the complete binding; neither shape nor digest alone suffices."""
    if saved != current:
        raise ValueError("saved replay requires a freshly validated binding")


@dataclass(frozen=True)
class PayloadLease:
    """A view certificate binds one allocation identity and content generation."""

    binding: PhysicalBinding
    content_generation: int


@dataclass
class PayloadBuffer:
    """A finite lease checker, not a model of GPU synchronization."""

    binding: PhysicalBinding
    content_generation: int = 0
    values: tuple[int, ...] = ()

    def publish(self, values: tuple[int, ...]) -> PayloadLease:
        """Advance content generation; the caller owes final-reader completion."""
        self.content_generation += 1
        self.values = values
        return PayloadLease(self.binding, self.content_generation)

    def read(self, lease: PayloadLease, expected: PhysicalBinding) -> tuple[int, ...]:
        """Reject overwritten payloads or changed allocations without weakening P."""
        validate_replay(lease.binding, expected)
        validate_replay(expected, self.binding)
        if lease.content_generation != self.content_generation:
            raise ValueError("payload lease expired after overwrite")
        return self.values


def restore_forward_value(
    required_revision: int, available: dict[int, tuple[int, ...]]
) -> tuple[int, ...]:
    """Select the required forward value, never the newest available revision."""
    if required_revision not in available:
        raise ValueError("required forward revision is unavailable")
    return available[required_revision]


def weak_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Enumerate count instances without imposing token router realizability."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in weak_compositions(total - first, parts - 1):
            yield (first, *tail)


def distinct_pair_histograms(tokens: int) -> frozenset[tuple[int, ...]]:
    """Independently enumerate distinct Top-2 choices over four experts."""
    pairs = tuple(combinations(range(4), 2))
    histograms = set()
    for choices in combinations_with_replacement(pairs, tokens):
        histogram = [0] * 4
        for pair in choices:
            for expert in pair:
                histogram[expert] += 1
        histograms.add(tuple(histogram))
    return frozenset(histograms)


def first_source_split(counts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Reconstruct Batch 1's first lexicographic four-slot source split."""
    if len(counts) != 4 or sum(counts) != 8 or min(counts) < 0:
        raise ValueError("expected four nonnegative counts totaling eight")
    first = next(
        row for row in product(*(range(count + 1) for count in counts)) if sum(row) == 4
    )
    second = tuple(count - value for count, value in zip(counts, first, strict=True))
    return first, second


def sorted_source_slots(
    counts: tuple[int, ...], ranks: int, selected: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Reconstruct GPU producer arrays, including repeated experts per token."""
    flat = tuple(expert for expert, count in enumerate(counts, 1) for _ in range(count))
    if ranks <= 0 or selected <= 0 or len(flat) % (ranks * selected):
        raise ValueError("slot total must divide into rank/token blocks")
    block = len(flat) // ranks
    return tuple(
        tuple(
            flat[start : start + selected]
            for start in range(rank * block, (rank + 1) * block, selected)
        )
        for rank in range(ranks)
    )


def four_rank_worksheet() -> tuple[tuple[ref.Route, ...], tuple[int, ...], ref.Matrix]:
    """Realize Exercise 37.10 under its explicitly supplied K=1 contract."""
    homes = (0, 0, 1, 1, 2, 2, 3, 3)
    routes = tuple(
        ref.Route(
            f"t{token}",
            token,
            1,
            3 + (token - 1) // 4,
            Fraction(1),
            (token - 1) // 6,
        )
        for token in range(1, 25)
    )
    ref.validate_routes(routes, 1)
    destinations = []
    for route in routes:
        if route.token <= 6:
            destination = 0
        elif 9 <= route.token <= 12:
            destination = 1
        elif 17 <= route.token <= 18:
            destination = 2
        else:
            destination = homes[route.expert - 1]
        destinations.append(destination)
    matrix = ref.allocation_matrix(routes, destinations, 8, 4)
    ref.validate_allocation(matrix, (0, 0, 4, 4, 4, 4, 4, 4), 6)
    return routes, tuple(destinations), matrix
