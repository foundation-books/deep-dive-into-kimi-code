"""Twelve deterministic evidence checks for Chapter 36."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

from . import faults, reference as ref


def _route_fixture() -> dict[str, object]:
    routes = ref.canonical_routes()
    per_source = tuple(
        sum(route.source_rank == rank for route in routes) for rank in range(ref.RANKS)
    )
    return ref.check_record(
        len(routes) == 20
        and len({route.identity for route in routes}) == 20
        and per_source == (10, 10),
        logical_routes=len(routes),
        routes_per_source=list(per_source),
    )


def _semantic_and_home_loads() -> dict[str, object]:
    counts = ref.semantic_counts(ref.canonical_routes(), ref.EXPERTS)
    loads = ref.home_loads(counts, ref.HOME, ref.RANKS)
    return ref.check_record(
        counts == ref.EXPECTED_COUNTS and loads == ref.EXPECTED_HOME_LOADS,
        semantic_counts=list(counts),
        home_loads=list(loads),
    )


def _balanced_allocation() -> dict[str, object]:
    plan = ref.balanced_plan()
    matrix = ref.validate_plan(
        plan, ref.HOME, ref.RANKS, ref.TOKENS_PER_RANK * ref.SELECTED
    )
    expected = ((4, 3), (6, 0), (0, 4), (0, 3))
    return ref.check_record(
        matrix == expected and ref.column_sums(matrix) == ref.EXPECTED_EXECUTION_LOADS,
        allocation=[list(row) for row in matrix],
        execution_loads=list(ref.column_sums(matrix)),
        migrated=[list(identity) for identity in ref.MIGRATED],
    )


def _redundant_budget() -> dict[str, object]:
    plan = ref.balanced_plan()
    matrix = ref.allocation_matrix(
        plan.route_snapshot, plan.destinations, ref.EXPERTS, ref.RANKS
    )
    remote = ref.redundant_sets(matrix, ref.HOME)
    return ref.check_record(
        remote == (frozenset(), frozenset({1}))
        and all(len(items) <= ref.COPY_BUDGET for items in remote),
        redundant_sets=[sorted(items) for items in remote],
        budget=ref.COPY_BUDGET,
    )


def _constructive_upper_bound() -> dict[str, object]:
    cases = 0
    maximum = 0
    for first, second, third in product(range(9), repeat=3):
        fourth = 8 - first - second - third
        if fourth < 0:
            continue
        counts = (first, second, third, fourth)
        matrix = ref.constructive_balance(counts, ref.HOME, ref.RANKS)
        value = max(len(items) for items in ref.redundant_sets(matrix, ref.HOME))
        maximum = max(maximum, value)
        cases += 1
    return ref.check_record(
        cases == 165 and maximum <= ref.EXPERTS // ref.RANKS,
        enumerated_count_vectors=cases,
        maximum_redundant_types=maximum,
        theorem_bound=ref.EXPERTS // ref.RANKS,
    )


def _tightness_witness() -> dict[str, object]:
    counts = (0, 0, 4, 4, 4, 4, 4, 4)
    homes = (0, 0, 1, 1, 2, 2, 3, 3)
    matrix = ref.constructive_balance(counts, homes, 4)
    remote = ref.redundant_sets(matrix, homes)
    rank_zero_types = len(remote[0])
    return ref.check_record(
        ref.column_sums(matrix) == (6, 6, 6, 6)
        and rank_zero_types == 2
        and max(len(items) for items in remote) == 2,
        counts=list(counts),
        allocation=[list(row) for row in matrix],
        rank_zero_remote_types=rank_zero_types,
        lower_bound=2,
    )


def _exact_oracle() -> dict[str, object]:
    objective, matrix = ref.exact_min_redundancy(
        ref.EXPECTED_COUNTS, ref.HOME, ref.RANKS
    )
    constructive = ref.constructive_balance(ref.EXPECTED_COUNTS, ref.HOME, ref.RANKS)
    families = {
        "balanced": (5, 5, 5, 5),
        "home_skew": (7, 6, 4, 3),
        "zipf_like": (10, 5, 3, 2),
        "single_home": (10, 10, 0, 0),
    }
    comparisons: dict[str, dict[str, int]] = {}
    family_pass = True
    for name, counts in families.items():
        optimum, _ = ref.exact_min_redundancy(counts, ref.HOME, ref.RANKS)
        logical = ref.constructive_balance(counts, ref.HOME, ref.RANKS)
        logical_value = max(
            len(items) for items in ref.redundant_sets(logical, ref.HOME)
        )
        ref.validate_allocation(logical, counts, 10)
        family_pass = (
            family_pass and optimum <= logical_value <= ref.EXPERTS // ref.RANKS
        )
        comparisons[name] = {"optimum": optimum, "logical": logical_value}
    return ref.check_record(
        objective == 1
        and max(len(items) for items in ref.redundant_sets(constructive, ref.HOME))
        == objective
        and family_pass,
        optimum=objective,
        exact_allocation=[list(row) for row in matrix],
        constructive_allocation=[list(row) for row in constructive],
        family_comparisons=comparisons,
    )


def _aligned_segments() -> dict[str, object]:
    plan = ref.balanced_plan()
    matrix = ref.allocation_matrix(
        plan.route_snapshot, plan.destinations, ref.EXPERTS, ref.RANKS
    )
    segments = [
        ref.physical_segments(matrix, ref.HOME, rank, ref.ALIGNMENT)
        for rank in range(ref.RANKS)
    ]
    real = tuple(sum(segment.real_rows for segment in rank) for rank in segments)
    allocated = tuple(
        sum(segment.allocated_rows for segment in rank) for rank in segments
    )
    buffers = ref.dispatch_buffers(plan)
    masks = tuple(tuple(record is not None for record in buffer) for buffer in buffers)
    return ref.check_record(
        real == (10, 10)
        and allocated == (12, 12)
        and tuple(sum(mask) for mask in masks) == real
        and tuple(len(mask) for mask in masks) == allocated,
        real_rows=list(real),
        allocated_rows=list(allocated),
        padding_rows=[allocated[i] - real[i] for i in range(ref.RANKS)],
        real_slot_masks=[list(mask) for mask in masks],
    )


def _forward_checksum() -> dict[str, object]:
    home = ref.home_plan()
    balanced = ref.balanced_plan()
    home_checksum = ref.diagnostic_checksum(home)
    balanced_checksum = ref.diagnostic_checksum(balanced)
    buffers = ref.dispatch_buffers(balanced)
    dispatched_checksum = ref.diagnostic_checksum_from_buffers(buffers, balanced)
    dispatched_identities = tuple(
        record.identity for buffer in buffers for record in buffer if record is not None
    )
    return ref.check_record(
        home_checksum
        == balanced_checksum
        == dispatched_checksum
        == ref.EXPECTED_CHECKSUM
        and len(dispatched_identities) == len(set(dispatched_identities)) == 20
        and set(dispatched_identities)
        == {route.identity for route in balanced.route_snapshot},
        home_checksum=str(home_checksum),
        balanced_checksum=str(balanced_checksum),
        dispatched_checksum=str(dispatched_checksum),
        restored_route_count=len(dispatched_identities),
    )


def _plan_and_alias_lifetime() -> dict[str, object]:
    plan = ref.balanced_plan()
    ref.validate_bound_operation(plan, plan.route_snapshot, 0, 35, 1, plan.digest)
    ref.validate_view_epoch(plan, view_epoch=1, current_epoch=1)
    stale_route = stale_destination = stale_alias = False
    try:
        changed = ref.replace_route(plan.route_snapshot, 0, expert=4)
        ref.validate_bound_operation(plan, changed, 0, 35, 1, plan.digest)
    except ValueError:
        stale_route = True
    changed_destinations = tuple(0 for _ in plan.destinations)
    try:
        mutated_plan = replace(plan, destinations=changed_destinations)
        ref.validate_bound_operation(
            mutated_plan, plan.route_snapshot, 0, 35, 1, plan.digest
        )
    except ValueError:
        stale_destination = True
    try:
        ref.validate_view_epoch(plan, view_epoch=1, current_epoch=2)
    except ValueError:
        stale_alias = True
    return ref.check_record(
        stale_route and stale_destination and stale_alias,
        stale_route=stale_route,
        stale_destination=stale_destination,
        stale_alias=stale_alias,
    )


def _gradient_repatriation() -> dict[str, object]:
    plan = ref.balanced_plan()
    partials = ref.gradient_partials(plan)
    gradients = ref.repatriated_gradients(plan)
    return ref.check_record(
        partials[(1, 0)] == 104 and partials[(1, 1)] == 184 and gradients[1] == 288,
        expert_1_home_partial=partials[(1, 0)],
        expert_1_temporary_partial=partials[(1, 1)],
        expert_1_home_gradient=gradients[1],
    )


def _support_and_layout() -> dict[str, object]:
    envelope = ref.SupportEnvelope(
        "training", 2, 4, 5, 2, 2, 4, gradients=True, zero_copy=True
    )
    ref.validate_training_envelope(envelope)
    byte_ledger = ref.layout_bytes(4, 2, 2, 8, 3, 2, 4)
    expected = {
        "symmetric_weight_view": 288,
        "owned_home_weights": 96,
        "local_prefetch_weights": 96,
        "symmetric_gradient_view": 576,
        "owned_home_gradients": 192,
        "local_temporary_gradients": 192,
        "mapped_reduce_view": 384,
        "local_reduce_storage": 192,
    }
    return ref.check_record(
        byte_ledger == expected
        and envelope.tokens_per_rank == ref.TOKENS_PER_RANK
        and envelope.zero_copy,
        **byte_ledger,
    )


def _fault_mutants() -> dict[str, object]:
    witnesses = faults.fault_witnesses()
    return ref.check_record(
        tuple(witnesses) == faults.FAULT_IDS and all(witnesses.values()),
        detected_faults=list(witnesses),
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Return C35-DET01 through C35-DET12 in stable order."""
    return {
        "route_fixture": _route_fixture(),
        "semantic_and_home_loads": _semantic_and_home_loads(),
        "balanced_allocation": _balanced_allocation(),
        "redundant_budget": _redundant_budget(),
        "constructive_upper_bound": _constructive_upper_bound(),
        "tightness_witness": _tightness_witness(),
        "exact_oracle": _exact_oracle(),
        "aligned_segments": _aligned_segments(),
        "forward_checksum": _forward_checksum(),
        "plan_and_alias_lifetime": _plan_and_alias_lifetime(),
        "gradient_repatriation": _gradient_repatriation(),
        "support_layout_and_faults": ref.check_record(
            _support_and_layout()["status"] == "pass"
            and _fault_mutants()["status"] == "pass",
            layout=_support_and_layout(),
            faults=_fault_mutants(),
        ),
    }
