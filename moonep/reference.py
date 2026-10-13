"""Exact semantic and accounting oracles for Chapter 36.

The module models route placement, bounded redundant expert types, aligned
segments, plan identity, zero-copy lifetimes, and gradient repatriation.  It
does not implement MoonEP kernels, communication, symmetric virtual memory,
or accelerator performance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256

Scalar = Fraction
RouteId = tuple[str, int]
Matrix = tuple[tuple[int, ...], ...]

RANKS = 2
EXPERTS = 4
TOKENS_PER_RANK = 5
SELECTED = 2
COPY_BUDGET = 2
ALIGNMENT = 4
HOME: tuple[int, ...] = (0, 0, 1, 1)
EXPECTED_COUNTS = (7, 6, 4, 3)
EXPECTED_HOME_LOADS = (13, 7)
EXPECTED_EXECUTION_LOADS = (10, 10)
EXPECTED_CHECKSUM = Fraction(5_521_654_629, 1_000_000)
MIGRATED = (("tau1", 1), ("tau2", 1), ("tau3", 2))


def f(numerator: int, denominator: int = 1) -> Fraction:
    """Construct one exact scalar."""
    return Fraction(numerator, denominator)


@dataclass(frozen=True, order=True)
class Route:
    """One immutable semantic route plus its input-rank origin."""

    origin: str
    token: int
    slot: int
    expert: int
    weight: Fraction
    source_rank: int

    @property
    def identity(self) -> RouteId:
        return (self.origin, self.slot)


@dataclass(frozen=True)
class Plan:
    """A route-level execution placement bound to one memory epoch."""

    route_snapshot: tuple[Route, ...]
    destinations: tuple[int, ...]
    microbatch: int
    layer: int
    epoch: int
    copy_budget: int = COPY_BUDGET
    alignment: int = ALIGNMENT

    @property
    def digest(self) -> str:
        """Return a canonical fingerprint of routes, placement, and lifetime."""
        return plan_digest(self)


@dataclass(frozen=True)
class Segment:
    """One real expert group inside an aligned rank-local physical buffer."""

    expert: int
    physical_row: int
    kind: str
    real_rows: int
    allocated_rows: int
    start: int
    end: int


@dataclass(frozen=True)
class PhysicalCopy:
    """A temporary expert value qualified by revision and readiness epoch."""

    expert: int
    rank: int
    parameter_revision: int
    ready_epoch: int


@dataclass(frozen=True)
class SupportEnvelope:
    """The conditions that select a semantic execution path."""

    phase: str
    ranks: int
    experts: int
    tokens_per_rank: int
    selected: int
    copy_budget: int
    alignment: int
    gradients: bool
    zero_copy: bool


def canonical_routes() -> tuple[Route, ...]:
    """Return the Chapter 35 twenty-route fixture with two source ranks."""
    half = f(1, 2)
    origins = (
        ("v1", 1),
        ("v2", 2),
        ("v3", 3),
        ("v4", 4),
        ("tau1", 5),
        ("tau2", 6),
        ("tau3", 7),
        ("tau4", 8),
        ("tau5", 9),
        ("tau6", 10),
    )
    choices = (
        ((1, half), (2, half)),
        ((1, half), (2, half)),
        ((1, half), (2, half)),
        ((1, half), (2, half)),
        ((1, half), (3, half)),
        ((1, half), (3, half)),
        ((4, f(551_543, 1_000_000)), (1, f(448_457, 1_000_000))),
        ((3, half), (2, half)),
        ((4, half), (2, half)),
        ((4, half), (3, half)),
    )
    records: list[Route] = []
    for (origin, token), selected in zip(origins, choices, strict=True):
        source_rank = 0 if token <= 5 else 1
        for slot, (expert, weight) in enumerate(selected, 1):
            records.append(Route(origin, token, slot, expert, weight, source_rank))
    validate_routes(records, selected_count=SELECTED)
    return tuple(records)


def validate_routes(routes: Sequence[Route], selected_count: int) -> None:
    """Validate exact identity, per-token multiplicity, and semantic weights."""
    identities = tuple(route.identity for route in routes)
    if len(set(identities)) != len(identities):
        raise ValueError("route identities must be unique")
    grouped: dict[str, list[Route]] = {}
    for route in routes:
        if route.expert < 1 or route.weight <= 0 or route.source_rank < 0:
            raise ValueError("route fields are outside their domains")
        grouped.setdefault(route.origin, []).append(route)
    for origin, records in grouped.items():
        if len(records) != selected_count:
            raise ValueError(f"{origin} must have exactly {selected_count} routes")
        if {record.slot for record in records} != set(range(1, selected_count + 1)):
            raise ValueError("route slots must be complete")
        if len({record.expert for record in records}) != selected_count:
            raise ValueError("selected experts must be distinct")
        if sum((record.weight for record in records), f(0)) != 1:
            raise ValueError("execution weights must sum to one")


def route_digest(routes: Sequence[Route]) -> str:
    """Hash every semantics-affecting route field in stable order."""
    payload = "|".join(
        (
            f"{route.origin}:{route.token}:{route.slot}:{route.expert}:"
            f"{route.weight.numerator}/{route.weight.denominator}:"
            f"{route.source_rank}"
        )
        for route in routes
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def plan_digest(plan: Plan) -> str:
    """Hash every field that determines this reference plan's use."""
    payload = "|".join(
        (
            f"routes={route_digest(plan.route_snapshot)}",
            "destinations=" + ",".join(str(rank) for rank in plan.destinations),
            f"microbatch={plan.microbatch}",
            f"layer={plan.layer}",
            f"epoch={plan.epoch}",
            f"copy_budget={plan.copy_budget}",
            f"alignment={plan.alignment}",
        )
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def semantic_counts(routes: Sequence[Route], expert_count: int) -> tuple[int, ...]:
    """Count immutable semantic expert selections."""
    counts = [0] * expert_count
    for route in routes:
        if not 1 <= route.expert <= expert_count:
            raise ValueError("route expert is outside the configured pool")
        counts[route.expert - 1] += 1
    return tuple(counts)


def home_loads(
    counts: Sequence[int], homes: Sequence[int], ranks: int
) -> tuple[int, ...]:
    """Aggregate semantic counts under fixed expert ownership."""
    if len(counts) != len(homes):
        raise ValueError("counts and homes must name the same experts")
    loads = [0] * ranks
    for count, home in zip(counts, homes, strict=True):
        if count < 0 or not 0 <= home < ranks:
            raise ValueError("counts and homes are outside their domains")
        loads[home] += count
    return tuple(loads)


def home_plan(routes: Sequence[Route] | None = None) -> Plan:
    """Place every route at its unique semantic expert home."""
    snapshot = tuple(routes or canonical_routes())
    destinations = tuple(HOME[route.expert - 1] for route in snapshot)
    return Plan(snapshot, destinations, microbatch=0, layer=0, epoch=0)


def balanced_plan(routes: Sequence[Route] | None = None) -> Plan:
    """Return the exact moonep-35-a placement."""
    snapshot = tuple(routes or canonical_routes())
    migrated = set(MIGRATED)
    destinations = tuple(
        1 if route.identity in migrated else HOME[route.expert - 1]
        for route in snapshot
    )
    plan = Plan(snapshot, destinations, microbatch=0, layer=35, epoch=1)
    validate_plan(plan, HOME, RANKS, TOKENS_PER_RANK * SELECTED)
    return plan


def allocation_matrix(
    routes: Sequence[Route], destinations: Sequence[int], expert_count: int, ranks: int
) -> Matrix:
    """Count semantic-expert rows by physical execution rank."""
    if len(routes) != len(destinations):
        raise ValueError("every route requires one destination")
    matrix = [[0 for _ in range(ranks)] for _ in range(expert_count)]
    for route, rank in zip(routes, destinations, strict=True):
        if not 0 <= rank < ranks:
            raise ValueError("destination rank is outside the EP group")
        matrix[route.expert - 1][rank] += 1
    return tuple(tuple(row) for row in matrix)


def column_sums(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return physical execution loads."""
    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("allocation matrix must be rectangular")
    return tuple(sum(row[rank] for row in matrix) for rank in range(width))


def redundant_sets(
    matrix: Sequence[Sequence[int]], homes: Sequence[int]
) -> tuple[frozenset[int], ...]:
    """Return remote semantic expert types active on each rank."""
    ranks = len(matrix[0]) if matrix else 0
    remote: list[set[int]] = [set() for _ in range(ranks)]
    for expert_index, (row, home) in enumerate(zip(matrix, homes, strict=True), 1):
        for rank, count in enumerate(row):
            if count and rank != home:
                remote[rank].add(expert_index)
    return tuple(frozenset(items) for items in remote)


def validate_allocation(
    matrix: Sequence[Sequence[int]], counts: Sequence[int], target: int
) -> None:
    """Check nonnegativity, semantic row conservation, and exact columns."""
    if len(matrix) != len(counts) or not matrix:
        raise ValueError("allocation and expert counts are incompatible")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("allocation counts must be nonnegative")
    if tuple(sum(row) for row in matrix) != tuple(counts):
        raise ValueError("allocation rows must preserve semantic counts")
    if any(load != target for load in column_sums(matrix)):
        raise ValueError("allocation columns must be perfectly balanced")


def validate_plan(plan: Plan, homes: Sequence[int], ranks: int, target: int) -> Matrix:
    """Validate a route-level plan and return its allocation matrix."""
    validate_routes(plan.route_snapshot, selected_count=SELECTED)
    matrix = allocation_matrix(
        plan.route_snapshot, plan.destinations, len(homes), ranks
    )
    validate_allocation(
        matrix, semantic_counts(plan.route_snapshot, len(homes)), target
    )
    if any(len(items) > plan.copy_budget for items in redundant_sets(matrix, homes)):
        raise ValueError("redundant expert budget exceeded")
    return matrix


def replace_route(
    routes: Sequence[Route], index: int, *, expert: int
) -> tuple[Route, ...]:
    """Return one changed snapshot for negative tests."""
    result = list(routes)
    result[index] = replace(result[index], expert=expert)
    return tuple(result)


def all_checks() -> dict[str, dict[str, object]]:
    """Return the twelve deterministic outcomes used by the evidence campaign."""
    from .checks import all_checks as collect_checks

    return collect_checks()


def check_record(passed: bool, **details: object) -> dict[str, object]:
    """Return one stable evidence row."""
    return {"status": "pass" if passed else "fail", **details}


from .planning import constructive_balance, exact_min_redundancy  # noqa: E402, F401
from .runtime import (  # noqa: E402, F401
    aligned_size,
    diagnostic_checksum,
    diagnostic_checksum_from_buffers,
    dispatch_buffers,
    gradient_partials,
    layout_bytes,
    physical_segments,
    repatriated_gradients,
    validate_bound_operation,
    validate_copy,
    validate_training_envelope,
    validate_view_epoch,
)
