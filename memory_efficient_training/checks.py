"""Twelve deterministic checks aligned with the Chapter 38 contracts."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from .faults import (
    duplicate_epoch,
    wrong_microbatch,
    wrong_route_order,
    wrong_scale,
    wrong_shard_order,
)
from .objects import ObjectKey, Residency, naive_peaks, running_object, sweep_peak
from .finite_fixture import complete_trace
from .end_to_end import check_end_to_end
from .numerical_fixture import (
    compression_witness,
    gradient_epochs,
    replay_panel,
    row_update_witness,
)
from .policies import (
    Lease,
    Policy,
    Representation,
    compose_policies,
    completion_ms,
    dropout_fixture,
    max_error,
    quantize,
    readiness_slack,
    replay,
    validate_leases,
    validate_representation,
)
from .specialized import (
    attnres_release,
    matrix_fixture,
    moe_backward,
    moe_fixture,
    muon_memory_counts,
    newton_schulz_step,
    owner_directed_result,
    pipeline_balanced_peaks,
    pipeline_schedule_feasible,
    reduce_scatter_to_cpu,
)


def result(passed: bool, **details: object) -> dict[str, object]:
    """Build one JSON-ready deterministic check result."""
    return {"status": "pass" if passed else "fail", **details}


def check_identity() -> dict[str, object]:
    """Check that a stale microbatch key cannot identify the live object."""
    obj = running_object(1, "routed-input", 4, (12, 18))
    mutant = wrong_microbatch(obj.key)
    accepted_keys = {obj.key}
    stale_rejected = mutant not in accepted_keys
    return result(
        stale_rejected and obj.lifetime == (0, 18),
        key=obj.key.__dict__,
        stale_microbatch_rejected=stale_rejected,
    )


def check_peak() -> dict[str, object]:
    """Check ordered event sweeps, peak values, and the first violation."""
    small = tuple(
        Residency(ObjectKey(name, 0, 0, "toy", name, 0), 0, "hbm", start, stop, units)
        for name, start, stop, units in (
            ("a", 0, 5, 6),
            ("b", 1, 3, 4),
            ("c", 2, 7, 5),
            ("d", 3, 6, 7),
            ("e", 5, 8, 3),
        )
    )
    release_first = sweep_peak(8, small, rank=0, tier="hbm", capacity=24)
    allocate_first = sweep_peak(
        8, small, rank=0, tier="hbm", capacity=24, release_before_allocate=False
    )
    naive = naive_peaks()
    first_violation = release_first.first_violation
    return result(
        naive == (57, 31)
        and release_first.peak == 26
        and allocate_first.peak == 30
        and first_violation == (3, "+d"),
        naive_peaks=naive,
        toy_peaks=(release_first.peak, allocate_first.peak),
        first_violation=first_violation,
    )


def check_policy_types() -> dict[str, object]:
    """Check typed policy composition and required decoder metadata."""
    legal = compose_policies(
        "full",
        (
            Policy("compress", "full", "encoded", True),
            Policy("offload", "encoded", "remote", True),
        ),
    )
    discard_rejected = False
    try:
        compose_policies("full", (Policy("discard", "full", "none", False),))
    except ValueError:
        discard_rejected = True
    key = ObjectKey("x", 7, 2, "block11", "routed-input", 0)
    incomplete = Representation(
        key, "encoded", (Fraction(1),), (("format", "toy8"),), False
    )
    metadata_rejected = False
    try:
        validate_representation(
            incomplete,
            expected_key=key,
            required_metadata=(("format", "toy8"), ("scale", "1/8")),
        )
    except ValueError:
        metadata_rejected = True
    return result(
        legal == "remote" and discard_rejected and metadata_rejected,
        legal_output=legal,
        discard_rejected=discard_rejected,
        missing_metadata_rejected=metadata_rejected,
    )


def check_recompute() -> dict[str, object]:
    """Check exact replay and reject mutations of the replay closure."""
    closure = ((2, 3, 4), 5, 7, "exact-int")
    expected = dropout_fixture(*closure)
    exact = replay(dropout_fixture, closure, closure)
    mutants = {
        "rng": ((2, 3, 4), 5, 8, "exact-int"),
        "weight": ((2, 3, 4), 6, 7, "exact-int"),
        "autocast": ((2, 3, 4), 5, 7, "truncated-int"),
    }
    rejected: dict[str, bool] = {}
    for name, mutant in mutants.items():
        try:
            replay(dropout_fixture, mutant, closure)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    return result(
        exact == expected
        and all(rejected.values())
        and replay_panel() == (11, 0, 21)
        and replay_panel(6) == (13, 0, 25)
        and replay_panel(mask=(0, 1, 0)) == (0, 16, 0),
        output=exact,
        closure_mutants_rejected=rejected,
        printed_mask_panel=replay_panel(),
        printed_weight_mutant=replay_panel(6),
        printed_mask_mutant=replay_panel(mask=(0, 1, 0)),
    )


def check_compression() -> dict[str, object]:
    """Check the declared compression tolerance and versioned scale data."""
    witness = compression_witness()
    original = witness["inputs"]
    assert isinstance(original, tuple)
    decoded = quantize(original, Fraction(1, 8))
    error = max_error(original, decoded)
    wrong_metadata_error = max_error(original, quantize(original, Fraction(1, 2)))
    key = ObjectKey("x", 7, 2, "block11", "routed-input", 0)
    record = Representation(
        key,
        "encoded",
        decoded,
        (("format", "toy8"), ("scale", "1/8")),
        False,
    )
    validate_representation(
        record,
        expected_key=key,
        required_metadata=(("format", "toy8"), ("scale", "1/8")),
    )
    scale_rejected = False
    stale = replace(
        record, metadata=(("format", "toy8"), ("scale", wrong_scale("1/8")))
    )
    try:
        validate_representation(
            stale,
            expected_key=key,
            required_metadata=(("format", "toy8"), ("scale", "1/8")),
        )
    except ValueError:
        scale_rejected = True
    return result(
        error <= Fraction(1, 16) and wrong_metadata_error > error and scale_rejected,
        max_error=error,
        mutant_error=wrong_metadata_error,
        stale_scale_rejected=scale_rejected,
        printed_witness=witness,
    )


def check_offload() -> dict[str, object]:
    """Check serial transfer deadlines and nonoverlapping buffer leases."""
    six_first = completion_ms(6, Fraction(0))
    six_then_ten = (six_first, completion_ms(10, six_first))
    ten_first = completion_ms(10, Fraction(0))
    ten_then_six = (ten_first, completion_ms(6, ten_first))
    key_a = ObjectKey("a", 0, 0, "x", "x", 0)
    key_b = ObjectKey("b", 0, 0, "x", "x", 0)
    validate_leases((Lease("link", key_a, 0, 3), Lease("link", key_b, 3, 6)))
    alias_rejected = False
    try:
        validate_leases((Lease("pool:A", key_a, 0, 4), Lease("pool:A", key_b, 3, 6)))
    except ValueError:
        alias_rejected = True
    slacks = {
        "six_then_ten": (
            readiness_slack(180, six_then_ten[0]),
            readiness_slack(330, six_then_ten[1]),
        ),
        "ten_then_six": (
            readiness_slack(330, ten_then_six[0]),
            readiness_slack(180, ten_then_six[1]),
        ),
    }
    return result(
        slacks["six_then_ten"] == (Fraction(1499, 50), Fraction(-1751, 25))
        and slacks["ten_then_six"] == (Fraction(3999, 50), Fraction(-5501, 25))
        and alias_rejected,
        completions_ms={"six_then_ten": six_then_ten, "ten_then_six": ten_then_six},
        readiness_slacks_ms=slacks,
        alias_overlap_rejected=alias_rejected,
    )


def check_moe() -> dict[str, object]:
    """Check the MoE backward witness and route-order sensitivity."""
    fixture = moe_fixture()
    weights = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
    activations = (
        (Fraction(1), Fraction(2)),
        (Fraction(-1), Fraction(3)),
        (Fraction(2), Fraction(0)),
    )
    mutant_order = wrong_route_order((0, 1, 2))
    mutant = moe_backward(
        weights,
        tuple(activations[index] for index in mutant_order),
        (Fraction(4), Fraction(-2)),
    )
    route_mutant_rejected = mutant["route_grads"] != fixture["route_grads"]
    return result(
        fixture["output"] == (Fraction(1, 2), Fraction(2))
        and fixture["route_grads"] == (Fraction(0), Fraction(-10), Fraction(8))
        and route_mutant_rejected,
        route_order_mutant_rejected=route_mutant_rejected,
        **fixture,
    )


def check_attnres() -> dict[str, object]:
    """Check source and destination release events for shared residual state."""
    release = attnres_release(5, (12,), 8, (15, 17))
    holders = sum(start <= 10 < stop for start, stop in ((5, 12), (8, 17)))
    early_release_rejected = 8 < release[0] or 8 < release[1]
    return result(
        release == (12, 17) and early_release_rejected,
        release_events=release,
        early_release_rejected=early_release_rejected,
        logical_keys=1,
        simultaneous_holders=holders,
        physical_payload_units=2 * holders,
    )


def check_pipeline_loans() -> dict[str, object]:
    """Check feasible remote loans and a receiver-workspace counterexample."""
    balanced = pipeline_balanced_peaks()
    physical = complete_trace(True)
    conflicted = pipeline_balanced_peaks(6)
    return result(
        balanced == (32, 28)
        and pipeline_schedule_feasible()
        and not pipeline_schedule_feasible(6),
        balanced=balanced,
        receiver_workspace=conflicted,
        full_physical_peaks=physical.peaks,
        complete_operations=len(physical.events),
    )


def check_gradients() -> dict[str, object]:
    """Check complete once-only gradient reduction into owner shards."""
    first = tuple(range(16))
    second = tuple(2 * value for value in range(16))
    left, right = reduce_scatter_to_cpu((first, second))
    complete = left + right
    expected = tuple(3 * value for value in range(16))
    missing = reduce_scatter_to_cpu((first,))[0] + reduce_scatter_to_cpu((first,))[1]
    duplicate_rejected = duplicate_epoch(left) != left
    missing_rejected = missing != expected
    epochs = gradient_epochs()
    shards = epochs["owner_shards"]
    assert isinstance(shards, tuple)
    return result(
        complete == expected
        and duplicate_rejected
        and missing_rejected
        and shards[0] + shards[1] == epochs["reference_gradient"],
        owner_shards=(left, right),
        duplicate_epoch_rejected=duplicate_rejected,
        missing_replica_rejected=missing_rejected,
        printed_epochs=epochs,
        storage_units={
            "reserved_full_input": 4,
            "rolling_outputs": 2,
            "scratch_in_arena": 1,
            "host_shard": 2,
        },
    )


def publications_match(item: dict[str, object]) -> bool:
    """Compare every reconstructed parameter replica with the finite reference."""
    publications = item["published_replicas"]
    assert isinstance(publications, tuple)
    return all(p == item["reference_parameters"] for p in publications)


def check_muon() -> dict[str, object]:
    """Check owner-directed Muon equivalence and canonical shard ordering."""
    matrix = matrix_fixture(0)
    gathered = newton_schulz_step(matrix)
    owner = owner_directed_result(matrix)
    counts = muon_memory_counts()
    shards = (matrix[:2], matrix[2:])
    reordered = tuple(row for shard in wrong_shard_order(shards) for row in shard)
    shard_order_rejected = newton_schulz_step(reordered) != gathered
    paths = tuple(row_update_witness(j) for j in (0, 1))
    all_rows_published = all(
        publications_match(item) and item["update_count"] == (1, 1, 1, 1)
        for item in paths
    )
    return result(
        owner == gathered
        and shard_order_rejected
        and all_rows_published
        and counts
        == {
            "gather_peak_elements": 32,
            "gather_receive_elements": 16,
            "owner_peak_elements": 16,
            "owner_receive_elements": 8,
        },
        counts=counts,
        shard_order_mutant_rejected=shard_order_rejected,
        all_rows_published=all_rows_published,
        row_paths=paths,
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Run C37-DET01 through C37-DET12 in stable identifier order."""
    checks = (
        check_identity,
        check_peak,
        check_policy_types,
        check_recompute,
        check_compression,
        check_offload,
        check_moe,
        check_attnres,
        check_pipeline_loans,
        check_gradients,
        check_muon,
        check_end_to_end,
    )
    return {f"C37-DET{index:02d}": check() for index, check in enumerate(checks, 1)}
