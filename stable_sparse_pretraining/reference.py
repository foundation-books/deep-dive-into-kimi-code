"""Dependency-free semantic oracles for stable sparse pretraining.

The module fixes the tiny Chapter 14 fixture and makes its conservation,
ownership, and restart claims executable.  It is deliberately not a MoonEP,
distributed optimizer, checkpoint engine, or performance implementation.
Fractions keep routing, gradient, optimizer, and replay comparisons exact; a
small floating-point matrix routine is used only for the finite Per-Head Muon
example.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Mapping, Sequence, cast

from .reference_checkpoint import (
    OptimizerState,
    TrainingState as TrainingState,
    Vector as Vector,
    advance_state as advance_state,
    assemble_shards as assemble_shards,
    checkpoint_manifest as checkpoint_manifest,
    initial_training_state as initial_training_state,
    load_checkpoint as load_checkpoint,
    optimizer_step,
    restart_replay_fixture as restart_replay_fixture,
    shard_vector as shard_vector,
    state_digest as state_digest,
    state_payload as state_payload,
)
from .reference_matrix import (
    Matrix,
    global_stacked_muon as global_stacked_muon,
    max_abs_difference,
    per_head_fixture as per_head_fixture,
    per_head_muon as per_head_muon,
)
from .reference_qb import qb_fixture as qb_fixture
from .reference_qb import qb_select as qb_select
from .reference_routing import (
    ROUTE_SCORES as ROUTE_SCORES,
    ROUTE_WEIGHTS as ROUTE_WEIGHTS,
    TOP2_EXPERTS as TOP2_EXPERTS,
    RouteRecord as RouteRecord,
    balance_execution as balance_execution,
    home_rank as home_rank,
    route_identity as route_identity,
    routes_from_pairs as routes_from_pairs,
    running_routes as running_routes,
    validate_execution as validate_execution,
)


def placement_fault_fixture() -> dict[str, object]:
    """Exercise balanced/skewed plans and four route-integrity failures."""

    running = running_routes()
    balanced = balance_execution(running)
    validate_execution(
        balanced,
        route_identity(running),
        copy_versions={(0, 1): 1},
    )

    skew_free_pairs = ((0, 2), (0, 3), (1, 2), (1, 3)) * 2
    skew_free = routes_from_pairs(skew_free_pairs)
    validate_execution(skew_free, route_identity(skew_free))

    maximum_skew_home = routes_from_pairs(((0, 1),) * 8)
    maximum_skew_balanced = tuple(
        replace(record, execution_rank=1) if record.expert == 0 else record
        for record in maximum_skew_home
    )
    validate_execution(
        maximum_skew_balanced,
        route_identity(maximum_skew_home),
        copy_versions={(0, 1): 1},
    )

    failures: dict[str, bool] = {}
    corruptions = {
        "dropped_route": balanced[:-1],
        "duplicate_route": balanced[:-1] + (balanced[0],),
        "duplicate_top2_expert": (replace(balanced[0], expert=balanced[1].expert),)
        + balanced[1:],
    }
    for name, corrupted in corruptions.items():
        try:
            validate_execution(
                corrupted,
                route_identity(running),
                copy_versions={(0, 1): 1},
            )
            failures[name] = False
        except ValueError:
            failures[name] = True
    try:
        validate_execution(
            balanced,
            route_identity(running),
            copy_versions={(0, 1): 0},
        )
        failures["stale_copy"] = False
    except ValueError:
        failures["stale_copy"] = True

    return {
        "running_loads": rank_loads(balanced),
        "skew_free_loads": rank_loads(skew_free),
        "maximum_skew_home_loads": rank_loads(maximum_skew_home),
        "maximum_skew_balanced_loads": rank_loads(maximum_skew_balanced),
        "detected_failures": failures,
    }


def expert_counts(records: Sequence[RouteRecord]) -> tuple[int, ...]:
    """Count semantic routes by expert."""

    return tuple(
        sum(record.expert == expert for record in records) for expert in range(4)
    )


def rank_loads(records: Sequence[RouteRecord]) -> tuple[int, int]:
    """Count physical route executions by EP rank."""

    return tuple(
        sum(record.execution_rank == rank for record in records) for rank in range(2)
    )  # type: ignore[return-value]


def expert_value(expert: int, token: int) -> Fraction:
    """Deterministic scalar expert output for a route."""

    return Fraction(100 * expert + token + 1)


def combined_outputs(records: Sequence[RouteRecord]) -> tuple[Fraction, ...]:
    """Combine route outputs by semantic identity, independent of placement."""

    token_count = 1 + max(record.token for record in records)
    return tuple(
        sum(
            (
                record.weight * expert_value(record.expert, token)
                for record in records
                if record.token == token
            ),
            start=Fraction(0),
        )
        for token in range(token_count)
    )


def route_gradient(record: RouteRecord) -> Fraction:
    """Return one exact scalar contribution tagged by its semantic route."""

    return Fraction((record.dp_replica + 1) * (record.token + 1) * record.slot, 10)


def execution_partials(
    records: Sequence[RouteRecord],
) -> dict[tuple[int, int], Fraction]:
    """Accumulate gradient partials by execution rank and semantic expert."""

    partials: dict[tuple[int, int], Fraction] = {}
    for record in records:
        key = (record.execution_rank, record.expert)
        partials[key] = partials.get(key, Fraction(0)) + route_gradient(record)
    return partials


def repatriate_to_home(
    partials: Mapping[tuple[int, int], Fraction],
) -> tuple[Fraction, ...]:
    """Sum temporary-copy partials at each expert's unique home owner."""

    return tuple(
        sum(
            (
                value
                for (_rank, partial_expert), value in partials.items()
                if partial_expert == expert
            ),
            start=Fraction(0),
        )
        for expert in range(4)
    )


def direct_semantic_gradients(
    records: Sequence[RouteRecord],
) -> tuple[Fraction, ...]:
    """Compute the mathematical expert gradient without physical placement."""

    return tuple(
        sum(
            (route_gradient(record) for record in records if record.expert == expert),
            start=Fraction(0),
        )
        for expert in range(4)
    )


def dp_consensus(
    replica_gradients: Sequence[Sequence[Fraction]],
) -> tuple[Fraction, ...]:
    """Sum the same expert coordinate across data-parallel replicas."""

    if not replica_gradients or any(len(row) != 4 for row in replica_gradients):
        raise ValueError("each replica must supply four expert gradients")
    return tuple(
        sum((row[i] for row in replica_gradients), start=Fraction(0)) for i in range(4)
    )


def sharded_owner_step(
    state: OptimizerState, gradient: Sequence[Fraction]
) -> OptimizerState:
    """Update two disjoint owner shards and reassemble the logical state."""

    if len(state.weights) != 4 or len(gradient) != 4:
        raise ValueError("the running fixture has four coordinates")
    shards: list[OptimizerState] = []
    for start, stop in ((0, 2), (2, 4)):
        shards.append(
            optimizer_step(
                OptimizerState(
                    state.weights[start:stop], state.momentum[start:stop], state.step
                ),
                gradient[start:stop],
            )
        )
    return OptimizerState(
        tuple(value for shard in shards for value in shard.weights),
        tuple(value for shard in shards for value in shard.momentum),
        shards[0].step,
    )


def owner_equivalence_fixture() -> dict[str, object]:
    """Compare one unsharded update with two disjoint owner shards."""

    replica_records = tuple(
        balance_execution(running_routes(dp_replica=replica)) for replica in range(2)
    )
    replica_gradients = tuple(
        repatriate_to_home(execution_partials(records)) for records in replica_records
    )
    consensus = dp_consensus(replica_gradients)
    initial = OptimizerState(
        tuple(Fraction(value) for value in (10, 20, 30, 40)),
        (Fraction(0),) * 4,
        0,
    )
    return {
        "gradient": consensus,
        "unsharded": optimizer_step(initial, consensus),
        "sharded": sharded_owner_step(initial, consensus),
    }


def owner_fault_fixture() -> dict[str, bool]:
    """Detect duplicate owner updates and incomplete matrix transforms."""

    fixture = owner_equivalence_fixture()
    reference = fixture["unsharded"]
    if not isinstance(reference, OptimizerState):
        raise TypeError("owner fixture returned an invalid state")
    gradient = cast(Sequence[Fraction], fixture["gradient"])
    duplicate = optimizer_step(reference, gradient)

    complete_block: Matrix = ((4.0, 1.0), (2.0, 3.0))
    complete_transform = per_head_muon((complete_block,))[0]
    rowwise_transform: Matrix = tuple(
        per_head_muon(((row,),))[0][0] for row in complete_block
    )
    return {
        "double_update_detected": duplicate != reference,
        "incomplete_head_block_detected": (
            max_abs_difference(complete_transform, rowwise_transform) > 1e-6
        ),
    }
