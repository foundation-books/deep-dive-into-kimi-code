"""Tests for the Chapter 36 dependency-free MoonEP semantic companion."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from . import checks, faults, reference as ref
from . import temporal_balance as temporal


def test_reader_walkthrough_worksheets() -> None:
    import unittest

    from .test_moonep_reader_walkthrough import ReaderWalkthroughTests

    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(ReaderWalkthroughTests).run(result)
    assert result.wasSuccessful(), (result.errors, result.failures)


def test_differentiable_backward_lifecycle() -> None:
    from .backward_lifecycle import check_lifecycle

    check_lifecycle()


def test_fixture_preserves_chapter_34_route_semantics() -> None:
    routes = ref.canonical_routes()
    assert len(routes) == 20
    assert len({route.identity for route in routes}) == 20
    assert ref.semantic_counts(routes, 4) == (7, 6, 4, 3)
    assert tuple(
        sum(route.source_rank == rank for route in routes) for rank in range(2)
    ) == (10, 10)


def test_home_and_balanced_load_surfaces_are_distinct() -> None:
    counts = ref.semantic_counts(ref.canonical_routes(), 4)
    assert ref.home_loads(counts, ref.HOME, 2) == (13, 7)
    matrix = ref.validate_plan(ref.balanced_plan(), ref.HOME, 2, 10)
    assert matrix == ((4, 3), (6, 0), (0, 4), (0, 3))
    assert ref.column_sums(matrix) == (10, 10)


def test_balanced_plan_moves_three_routes_but_one_expert_type() -> None:
    plan = ref.balanced_plan()
    matrix = ref.allocation_matrix(plan.route_snapshot, plan.destinations, 4, 2)
    assert ref.redundant_sets(matrix, ref.HOME) == (frozenset(), frozenset({1}))
    moved = tuple(
        route.identity
        for route, rank in zip(plan.route_snapshot, plan.destinations, strict=True)
        if rank != ref.HOME[route.expert - 1]
    )
    assert moved == ref.MIGRATED


def test_constructive_fill_obeys_general_upper_bound() -> None:
    matrix = ref.constructive_balance((9, 1, 1, 5), ref.HOME, 2)
    ref.validate_allocation(matrix, (9, 1, 1, 5), 8)
    assert max(len(items) for items in ref.redundant_sets(matrix, ref.HOME)) <= 2


def test_tightness_fixture_requires_two_types_on_rank_zero() -> None:
    counts = (0, 0, 4, 4, 4, 4, 4, 4)
    homes = (0, 0, 1, 1, 2, 2, 3, 3)
    matrix = ref.constructive_balance(counts, homes, 4)
    assert ref.column_sums(matrix) == (6, 6, 6, 6)
    assert len(ref.redundant_sets(matrix, homes)[0]) == 2


def test_exact_oracle_finds_one_remote_type_for_running_example() -> None:
    objective, matrix = ref.exact_min_redundancy((7, 6, 4, 3), ref.HOME, 2)
    ref.validate_allocation(matrix, (7, 6, 4, 3), 10)
    assert objective == 1


def test_aligned_segments_keep_padding_ownerless() -> None:
    matrix = ref.validate_plan(ref.balanced_plan(), ref.HOME, 2, 10)
    rank_zero = ref.physical_segments(matrix, ref.HOME, 0, 4)
    rank_one = ref.physical_segments(matrix, ref.HOME, 1, 4)
    assert tuple(segment.real_rows for segment in rank_zero) == (4, 6)
    assert tuple(segment.allocated_rows for segment in rank_zero) == (4, 8)
    assert tuple(segment.real_rows for segment in rank_one) == (4, 3, 3)
    assert tuple(segment.allocated_rows for segment in rank_one) == (4, 4, 4)
    buffers = ref.dispatch_buffers(ref.balanced_plan())
    assert tuple(len(buffer) for buffer in buffers) == (12, 12)
    assert tuple(
        sum(record is not None for record in buffer) for buffer in buffers
    ) == (
        10,
        10,
    )


def test_placement_preserves_exact_forward_checksum() -> None:
    plan = ref.balanced_plan()
    assert ref.diagnostic_checksum(plan) == ref.EXPECTED_CHECKSUM
    assert (
        ref.diagnostic_checksum_from_buffers(ref.dispatch_buffers(plan), plan)
        == ref.EXPECTED_CHECKSUM
    )


def test_buffer_combine_rejects_a_padding_leak() -> None:
    plan = ref.balanced_plan()
    buffers = [list(buffer) for buffer in ref.dispatch_buffers(plan)]
    padding_index = buffers[0].index(None)
    buffers[0][padding_index] = plan.route_snapshot[0]
    try:
        ref.diagnostic_checksum_from_buffers(
            tuple(tuple(buffer) for buffer in buffers), plan
        )
    except ValueError:
        pass
    else:
        raise AssertionError("padding exposed as a route was accepted")


def test_plan_copy_and_alias_epochs_are_checked_separately() -> None:
    plan = ref.balanced_plan()
    ref.validate_bound_operation(plan, plan.route_snapshot, 0, 35, 1, plan.digest)
    fresh = ref.PhysicalCopy(1, 1, parameter_revision=7, ready_epoch=1)
    ref.validate_copy(fresh, plan, parameter_revision=7)
    ref.validate_view_epoch(plan, view_epoch=1, current_epoch=1)
    try:
        ref.validate_copy(fresh, plan, parameter_revision=8)
    except ValueError:
        pass
    else:
        raise AssertionError("stale expert copy was accepted")


def test_plan_digest_binds_execution_destinations() -> None:
    plan = ref.balanced_plan()
    changed = replace(plan, destinations=ref.home_plan().destinations)
    assert changed.digest != plan.digest
    try:
        ref.validate_bound_operation(
            changed, plan.route_snapshot, 0, 35, 1, plan.digest
        )
    except ValueError:
        pass
    else:
        raise AssertionError("changed execution destinations were accepted")


def test_gradient_partials_repatriate_exactly_once() -> None:
    plan = ref.balanced_plan()
    partials = ref.gradient_partials(plan)
    assert partials[(1, 0)] == 104
    assert partials[(1, 1)] == 184
    assert ref.repatriated_gradients(plan)[1] == 288


def test_layout_ledger_separates_views_and_storage() -> None:
    ledger = ref.layout_bytes(4, 2, 2, 8, 3, 2, 4)
    assert ledger["symmetric_weight_view"] == 288
    assert ledger["owned_home_weights"] == 96
    assert ledger["local_prefetch_weights"] == 96
    assert ledger["mapped_reduce_view"] == 384
    assert ledger["local_reduce_storage"] == 192


def test_all_fault_mutants_are_detected() -> None:
    witnesses = faults.fault_witnesses()
    assert tuple(witnesses) == faults.FAULT_IDS
    assert all(witnesses.values())


def test_evidence_surface_has_twelve_passing_checks() -> None:
    outcomes = checks.all_checks()
    assert len(outcomes) == 12
    assert all(record["status"] == "pass" for record in outcomes.values())


def test_two_batches_preserve_routes_and_restore_each_output() -> None:
    plans = temporal.two_batch_plans()
    matrices = (((4, 3), (6, 0), (0, 4), (0, 3)), ((3, 0), (4, 0), (0, 6), (3, 4)))
    for plan, expected in zip(plans, matrices, strict=True):
        assert ref.validate_plan(plan, ref.HOME, 2, 10) == expected
        assert len({route.identity for route in plan.route_snapshot}) == 20
        assert tuple(
            sum(route.source_rank == rank for route in plan.route_snapshot)
            for rank in range(2)
        ) == (10, 10)
        restored = ref.diagnostic_checksum_from_buffers(
            ref.dispatch_buffers(plan), plan
        )
        assert restored == ref.diagnostic_checksum(ref.home_plan(plan.route_snapshot))
    assert plans[0].digest != plans[1].digest
    assert not {route.identity for route in plans[0].route_snapshot}.intersection(
        route.identity for route in plans[1].route_snapshot
    )


def test_average_balance_does_not_determine_batch_makespan() -> None:
    counts = tuple(
        ref.semantic_counts(plan.route_snapshot, 4)
        for plan in temporal.two_batch_plans()
    )
    assert counts == ((7, 6, 4, 3), (3, 4, 6, 7))
    assert tuple(Fraction(sum(pair), 2) for pair in zip(*counts, strict=True)) == (
        5,
        5,
        5,
        5,
    )
    loads = tuple(ref.home_loads(row, ref.HOME, 2) for row in counts)
    assert loads == ((13, 7), (7, 13))
    assert Fraction(sum(max(row) for row in loads), 2) == 13
    assert max(Fraction(sum(pair), 2) for pair in zip(*loads, strict=True)) == 10


def test_one_fixed_replica_placement_balances_both_batches() -> None:
    replicas = temporal.fixed_replicas()
    assert tuple(sum(rank == r for rank, _ in replicas) for r in range(2)) == (3, 3)
    for plan in temporal.two_batch_plans():
        matrix = temporal.validate_execution(
            plan,
            plan.route_snapshot,
            temporal.physical_rows(plan),
            replicas,
            temporal.ExecutionClaim(plan.microbatch, exact_balance=True),
        )
        assert ref.column_sums(matrix) == (10, 10)


def test_common_correctness_accepts_imbalance_without_a_balance_claim() -> None:
    plan = ref.home_plan()
    arguments = (
        plan,
        plan.route_snapshot,
        temporal.physical_rows(plan),
        temporal.fixed_replicas(),
    )
    matrix = temporal.validate_execution(*arguments, temporal.ExecutionClaim(0))
    assert ref.column_sums(matrix) == (13, 7)
    try:
        temporal.validate_execution(
            *arguments, temporal.ExecutionClaim(0, exact_balance=True)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unfulfilled perfect-balance claim was accepted")


def test_replica_faults_are_rejected_before_balance_or_timing() -> None:
    plan = temporal.two_batch_plans()[0]
    replicas = temporal.fixed_replicas()
    claim = temporal.ExecutionClaim(plan.microbatch)
    mutants = (
        replace(replicas[(1, 4)], expert=2),
        replace(replicas[(1, 4)], parameter_revision=6),
        replace(replicas[(1, 4)], quantization="mxfp4-other-scales"),
        replace(replicas[(1, 4)], placement_epoch=0),
    )
    for mutant in mutants:
        try:
            temporal.validate_execution(
                plan,
                plan.route_snapshot,
                temporal.physical_rows(plan),
                {**replicas, (1, 4): mutant},
                claim,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"inconsistent replica was accepted: {mutant}")


def test_wrong_batch_and_silent_backend_fallback_are_rejected() -> None:
    batch_a, batch_b = temporal.two_batch_plans()
    cases = (
        (batch_b.route_snapshot, temporal.ExecutionClaim(1)),
        (batch_a.route_snapshot, temporal.ExecutionClaim(1)),
        (
            batch_a.route_snapshot,
            temporal.ExecutionClaim(0, resolved_backend="different-backend"),
        ),
    )
    for routes, claim in cases:
        try:
            temporal.validate_execution(
                batch_a,
                routes,
                temporal.physical_rows(batch_a),
                temporal.fixed_replicas(),
                claim,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("wrong batch or undeclared fallback was accepted")
