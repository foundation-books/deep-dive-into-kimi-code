"""Two-batch placement and replica-consistency oracles, without GPU execution.

The fixed placement is a constructed witness, not output from EPLB. Replica
tags model caller checks, not quantization, migration, or synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from . import reference as ref


@dataclass(frozen=True)
class Replica:
    """The logical value currently bound to one physical expert slot."""

    expert: int
    parameter_revision: int = 7
    quantization: str = "bf16"
    placement_epoch: int = 1


@dataclass(frozen=True)
class ExecutionClaim:
    """Expected batch, values, placement publication, and selected backend."""

    microbatch: int
    parameter_revision: int = 7
    quantization: str = "bf16"
    placement_epoch: int = 1
    requested_backend: str = "constructed-reference"
    resolved_backend: str = "constructed-reference"
    exact_balance: bool = False


def two_batch_plans() -> tuple[ref.Plan, ref.Plan]:
    """Keep batch A intact and create a separately identified reflected batch."""
    batch_a = ref.balanced_plan()
    batch_b = ref.Plan(
        tuple(
            replace(route, origin="batch-b:" + route.origin, expert=5 - route.expert)
            for route in batch_a.route_snapshot
        ),
        tuple(1 - rank for rank in batch_a.destinations),
        microbatch=1,
        layer=batch_a.layer,
        epoch=batch_a.epoch + 1,
    )
    return batch_a, batch_b


def fixed_replicas() -> dict[tuple[int, int], Replica]:
    """Pre-position E4 on rank 0 and E1 on rank 1, plus the four homes."""
    replicas = {
        (rank, expert - 1): Replica(expert) for expert, rank in enumerate(ref.HOME, 1)
    }
    replicas[(0, ref.EXPERTS)] = Replica(4)
    replicas[(1, ref.EXPERTS)] = Replica(1)
    return replicas


def physical_rows(plan: ref.Plan) -> tuple[int, ...]:
    """Resolve each route into this fixture's home row or sole remote slot."""
    return tuple(
        route.expert - 1 if ref.HOME[route.expert - 1] == rank else ref.EXPERTS
        for route, rank in zip(plan.route_snapshot, plan.destinations, strict=True)
    )


def validate_execution(
    plan: ref.Plan,
    reference_routes: Sequence[ref.Route],
    rows: Sequence[int],
    replicas: Mapping[tuple[int, int], Replica],
    claim: ExecutionClaim,
) -> ref.Matrix:
    """Check common semantics, then enforce balance only when it is promised."""
    ref.validate_routes(reference_routes, ref.SELECTED)
    ref.validate_routes(plan.route_snapshot, ref.SELECTED)
    expected = {route.identity: route for route in reference_routes}
    actual = {route.identity: route for route in plan.route_snapshot}
    if actual != expected or plan.microbatch != claim.microbatch:
        raise ValueError("batch or immutable routes do not match")
    if claim.requested_backend != claim.resolved_backend:
        raise ValueError("requested backend was replaced by an undeclared fallback")
    matrix = ref.allocation_matrix(
        plan.route_snapshot, plan.destinations, ref.EXPERTS, ref.RANKS
    )
    for route, rank, row in zip(
        plan.route_snapshot, plan.destinations, rows, strict=True
    ):
        replica = replicas.get((rank, row))
        required = Replica(
            route.expert,
            claim.parameter_revision,
            claim.quantization,
            claim.placement_epoch,
        )
        if replica != required:
            raise ValueError("replica identity, value, layout, or placement is stale")
    if claim.exact_balance:
        ref.validate_plan(plan, ref.HOME, ref.RANKS, len(reference_routes) // ref.RANKS)
    return matrix
