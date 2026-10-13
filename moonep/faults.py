"""Single-fault mutants for Chapter 36 oracle sensitivity."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from . import reference as ref

FAULT_IDS = (
    "duplicate_route",
    "changed_semantic_expert",
    "missing_destination",
    "residual_imbalance",
    "copy_budget_overflow",
    "stale_plan",
    "stale_copy",
    "padding_leak",
    "expired_alias",
    "missing_gradient_reduce",
    "double_gradient_reduce",
    "unsupported_training_budget",
)


def _rejects(operation: Callable[[], object]) -> bool:
    try:
        operation()
    except ValueError:
        return True
    return False


def _duplicate_route(routes: tuple[ref.Route, ...]) -> bool:
    duplicate = list(routes)
    duplicate[1] = duplicate[0]
    return _rejects(lambda: ref.validate_routes(duplicate, ref.SELECTED))


def _changed_semantic_expert(routes: tuple[ref.Route, ...], plan: ref.Plan) -> bool:
    changed = ref.replace_route(routes, 0, expert=4)
    return _rejects(
        lambda: ref.validate_bound_operation(plan, changed, 0, 35, 1, plan.digest)
    )


def _missing_destination(routes: tuple[ref.Route, ...], plan: ref.Plan) -> bool:
    return _rejects(
        lambda: ref.allocation_matrix(
            routes, plan.destinations[:-1], ref.EXPERTS, ref.RANKS
        )
    )


def _residual_imbalance(routes: tuple[ref.Route, ...], plan: ref.Plan) -> bool:
    imbalanced = replace(plan, destinations=ref.home_plan(routes).destinations)
    return _rejects(
        lambda: ref.validate_plan(
            imbalanced, ref.HOME, ref.RANKS, ref.TOKENS_PER_RANK * ref.SELECTED
        )
    )


def _copy_budget_overflow() -> bool:
    overflow_matrix = (
        (4, 3),
        (5, 1),
        (1, 3),
        (0, 3),
    )
    reduced_budget = 1
    return len(ref.redundant_sets(overflow_matrix, ref.HOME)[1]) > reduced_budget


def _stale_plan(routes: tuple[ref.Route, ...], plan: ref.Plan) -> bool:
    stale_epoch = _rejects(
        lambda: ref.validate_bound_operation(plan, routes, 0, 35, 2, plan.digest)
    )
    changed_destinations = replace(
        plan, destinations=ref.home_plan(routes).destinations
    )
    stale_destination = _rejects(
        lambda: ref.validate_bound_operation(
            changed_destinations, routes, 0, 35, 1, plan.digest
        )
    )
    return stale_epoch and stale_destination


def _stale_copy(plan: ref.Plan) -> bool:
    stale = ref.PhysicalCopy(1, 1, parameter_revision=6, ready_epoch=1)
    return _rejects(lambda: ref.validate_copy(stale, plan, parameter_revision=7))


def _padding_leak(routes: tuple[ref.Route, ...], plan: ref.Plan) -> bool:
    buffers = [list(buffer) for buffer in ref.dispatch_buffers(plan)]
    padding_index = buffers[0].index(None)
    buffers[0][padding_index] = routes[0]
    dispatched = tuple(tuple(row) for row in buffers)
    return _rejects(lambda: ref.diagnostic_checksum_from_buffers(dispatched, plan))


def _expired_alias(plan: ref.Plan) -> bool:
    return _rejects(
        lambda: ref.validate_view_epoch(plan, view_epoch=1, current_epoch=2)
    )


def _gradient_reduction_faults(plan: ref.Plan) -> tuple[bool, bool]:
    partials = ref.gradient_partials(plan)
    direct = ref.repatriated_gradients(plan)[1]
    home_only = partials[(1, 0)]
    remote = partials[(1, 1)]
    return home_only != direct, home_only + 2 * remote != direct


def _unsupported_training_budget() -> bool:
    bad_envelope = ref.SupportEnvelope(
        "training", 2, 4, 5, 2, 1, 4, gradients=True, zero_copy=False
    )
    return _rejects(lambda: ref.validate_training_envelope(bad_envelope))


def fault_witnesses() -> dict[str, bool]:
    """Return True exactly when each injected defect is observable."""
    routes = ref.canonical_routes()
    plan = ref.balanced_plan(routes)
    ref.validate_plan(plan, ref.HOME, ref.RANKS, ref.TOKENS_PER_RANK * ref.SELECTED)
    missing_gradient, double_gradient = _gradient_reduction_faults(plan)
    outcomes = (
        _duplicate_route(routes),
        _changed_semantic_expert(routes, plan),
        _missing_destination(routes, plan),
        _residual_imbalance(routes, plan),
        _copy_budget_overflow(),
        _stale_plan(routes, plan),
        _stale_copy(plan),
        _padding_leak(routes, plan),
        _expired_alias(plan),
        missing_gradient,
        double_gradient,
        _unsupported_training_budget(),
    )
    return dict(zip(FAULT_IDS, outcomes, strict=True))
