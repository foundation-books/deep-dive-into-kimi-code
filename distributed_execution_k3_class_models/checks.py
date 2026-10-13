"""Twelve named deterministic checks promised by the Chapter 37 plan."""

from __future__ import annotations

from dataclasses import replace

from .faults import (
    duplicate_group_member,
    dropped_route,
    reversed_collectives,
    stale_epoch,
)
from .schedule import (
    canonical_collective_trace,
    first_collective_mismatch,
    ideal_interleaved_bubble_fraction,
    legal_overlap,
    longest_dependency_path,
    pipeline_events,
    pipeline_resource_schedule,
    resource_fixture,
    trace_digest,
    validate_pipeline,
    validate_resource_schedule,
)
from .reader_interfaces import mla_fixture, scalar_owner_witness
from .semantics import (
    balanced_destinations,
    canonical_routes,
    execution_loads,
    expert_gradients,
    f,
    fixed_destinations,
    kda_fixture,
    parameter_reductions,
    semantic_counts,
    tp_dense_and_sharded,
    validate_routes,
    wrong_nonlinear_split,
)
from .topology import (
    AXES,
    WORLD_SIZE,
    Coordinate,
    LayoutSignature,
    all_coordinates,
    decode,
    encode,
    focus_parameter_group,
    gather_columns,
    group_partition,
    groups_for_rank,
    placement,
    shard_columns,
    validate_partition,
)


def passed(detail: object) -> dict[str, object]:
    """Wrap deterministic check evidence in the common passing-result schema."""
    return {"status": "pass", "detail": detail}


def det01_rank_bijection() -> dict[str, object]:
    """Check rank-coordinate bijection and the focus-rank placement fixture."""
    coordinates = all_coordinates()
    assert len(coordinates) == WORLD_SIZE
    assert tuple(encode(coordinate) for coordinate in coordinates) == tuple(
        range(WORLD_SIZE)
    )
    assert all(decode(encode(coordinate)) == coordinate for coordinate in coordinates)
    assert decode(23) == Coordinate(1, 0, 1, 1, 1)
    assert placement(23) == (2, 7)
    return passed(
        {"world_size": WORLD_SIZE, "rank_23": (1, 0, 1, 1, 1), "placement": (2, 7)}
    )


def det02_groups() -> dict[str, object]:
    """Check focus-rank groups, axis partitions, and duplicate rejection."""
    focus = groups_for_rank(23)
    expected = {
        "t": (22, 23),
        "e": (21, 23),
        "c": (19, 23),
        "p": (23, 31),
        "d": (7, 23),
    }
    assert {axis: group.members for axis, group in focus.items()} == expected
    for axis in AXES:
        validate_partition(group_partition(axis))
    mutant = duplicate_group_member(focus["c"])
    try:
        mutant.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate group member mutant was accepted")
    return passed(expected)


def det03_layouts() -> dict[str, object]:
    """Check column sharding and the layout-signature invariants."""
    matrix = ((1, 2, 3, 4), (5, 6, 7, 8))
    shards = shard_columns(matrix)
    assert gather_columns(shards) == matrix
    layout = LayoutSignature(
        "X", (2, 4), ("t",), ("d",), ("p",), (0, 3, "F"), "rational"
    )
    layout.validate()
    bad = replace(layout, shard_axes=("t",), replica_axes=("t",))
    try:
        bad.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping shard/replica axes were accepted")
    return passed({"shards": shards, "layout": layout.object_id})


def det04_tensor_parallel() -> dict[str, object]:
    """Check the exact tensor-parallel result and nonlinear counterexample."""
    result = tp_dense_and_sharded()
    assert result["dense"] == result["reduced"] == (f(-2), f(4))
    assert result["local_activations"] == ((f(0), f(0)), (f(0), f(2)))
    assert wrong_nonlinear_split() != result["activation"]
    return passed({"dense": result["dense"], "wrong_split": wrong_nonlinear_split()})


def det05_pipeline() -> dict[str, object]:
    """Check the pipeline DAG, complete resource schedule, and timing bounds."""
    events = pipeline_events()
    validate_pipeline(events)
    schedule = pipeline_resource_schedule()
    validate_resource_schedule(schedule, events)
    durations = {
        event_id: f(2) if event_id.logical_chunk == 3 else f(1) for event_id in events
    }
    critical = longest_dependency_path(events, durations)
    assert len(events) == 32 and ideal_interleaved_bubble_fraction() == f(1, 9)
    assert len(schedule[0]) == 18
    assert critical > f(8)
    return passed(
        {
            "events": len(events),
            "schedule_slots": len(schedule[0]),
            "bubble_fraction": f(1, 9),
            "heterogeneous_critical_path": critical,
        }
    )


def det06_context_parallel() -> dict[str, object]:
    """Check ordered KDA affine fragments and the reversed-order mutant."""
    zero = kda_fixture()
    nonzero = kda_fixture(f(-1))
    assert zero == {"incoming": f(0), "ordered": f(1), "reversed": f(-3)}
    assert nonzero["ordered"] == f(-5) and nonzero["reversed"] == f(-9)
    for document in (0, 1):
        attention = mla_fixture(document)
        assert attention["before"] == attention["restored"]
        assert attention["remote_qkv_scalars"] == 24
        assert attention["remote_output_scalars"] == 8
    return passed(
        {
            "zero_state": zero,
            "nonzero_state": nonzero,
            "mla_payload_type": "sequence_shard",
        }
    )


def det07_expert_parallel() -> dict[str, object]:
    """Check route semantics under fixed and balanced expert placement."""
    routes = canonical_routes()
    fixed = fixed_destinations(routes)
    balanced = balanced_destinations(routes)
    assert semantic_counts(routes) == (3, 2, 2, 1)
    assert execution_loads(fixed) == (5, 3)
    assert execution_loads(balanced) == (4, 4)
    assert expert_gradients(routes, fixed) == expert_gradients(routes, balanced)
    try:
        validate_routes(dropped_route(routes))
    except ValueError:
        pass
    else:
        raise AssertionError("lost route mutant was accepted")
    return passed(
        {
            "routes": len(routes),
            "loads": (execution_loads(fixed), execution_loads(balanced)),
        }
    )


def det08_parameter_reductions() -> dict[str, object]:
    """Check class-specific replica groups and sum-versus-mean conventions."""
    assert focus_parameter_group("dense") == (5, 7, 21, 23)
    assert focus_parameter_group("routed_e0") == (5, 21)
    assert parameter_reductions("sum") == {"dense": f(10), "routed_e0": f(12)}
    assert parameter_reductions("mean") == {"dense": f(5, 2), "routed_e0": f(6)}
    wrong_mean = (f(5) + f(7) + f(0) + f(0)) / 4
    assert wrong_mean == f(3) and wrong_mean != f(6)
    owner = scalar_owner_witness()
    assert owner["replica_numerators"] == (f(160), f(224))
    assert owner["normalizer"] == 64 and owner["gradient"] == f(6)
    assert owner["updated"] == f(47, 5)
    assert scalar_owner_witness(False)["updated"] == owner["updated"]
    return passed(
        {"sum": parameter_reductions("sum"), "mean": parameter_reductions("mean")}
    )


def det09_collective_matching() -> dict[str, object]:
    """Check collective-trace agreement and two mismatch classifications."""
    trace = canonical_collective_trace()
    assert first_collective_mismatch({19: trace, 23: trace}) is None
    assert first_collective_mismatch({19: trace, 23: reversed_collectives(trace)}) == (
        23,
        0,
        "order",
    )
    assert first_collective_mismatch({19: trace, 23: stale_epoch(trace)}) == (
        23,
        0,
        "epoch",
    )
    return passed({"events": len(trace), "mutants": ("order", "epoch")})


def det10_overlap() -> dict[str, object]:
    """Check legal and contended overlap against one shared-link budget."""
    assert legal_overlap(f(10), (f(4), f(5)))
    assert not legal_overlap(f(10), (f(6), f(7)))
    return passed({"budget": f(10), "legal": (f(4), f(5)), "contended": (f(6), f(7))})


def det11_resources() -> dict[str, object]:
    """Check that logical payload can map to different physical link loads."""
    main = resource_fixture(False)
    alternative = resource_fixture(True)
    assert main.collective_payload == alternative.collective_payload
    assert main.physical_link_load != alternative.physical_link_load
    assert main.observed_time is alternative.observed_time is None
    assert main.link_load_bound == f(4) and alternative.link_load_bound == f(18)
    return passed(
        {
            "main_link_bound": main.link_load_bound,
            "alternative_link_bound": alternative.link_load_bound,
        }
    )


def det12_end_to_end() -> dict[str, object]:
    """Check the serial checksum, distributed checksum, and trace digest."""
    tensor = tp_dense_and_sharded()
    routes = canonical_routes()
    balanced = balanced_destinations(routes)
    updates = parameter_reductions("sum")
    serial_checksum = (
        sum(tensor["dense"], f(0))
        + sum(expert_gradients(routes, balanced), f(0))
        + sum(updates.values(), f(0))
    )
    distributed_checksum = (
        sum(tensor["reduced"], f(0))
        + sum(expert_gradients(routes, fixed_destinations(routes)), f(0))
        + sum(updates.values(), f(0))
    )
    assert serial_checksum == distributed_checksum == f(28)
    digest = trace_digest(pipeline_events())
    assert len(digest) == 64
    update_owners = {"dense": 1, "routed_e0": 1}
    assert set(update_owners.values()) == {1}
    return passed(
        {
            "checksum": serial_checksum,
            "trace_digest": digest,
            "updates_per_owner": update_owners,
        }
    )


CHECKS = (
    ("C36-DET01", det01_rank_bijection),
    ("C36-DET02", det02_groups),
    ("C36-DET03", det03_layouts),
    ("C36-DET04", det04_tensor_parallel),
    ("C36-DET05", det05_pipeline),
    ("C36-DET06", det06_context_parallel),
    ("C36-DET07", det07_expert_parallel),
    ("C36-DET08", det08_parameter_reductions),
    ("C36-DET09", det09_collective_matching),
    ("C36-DET10", det10_overlap),
    ("C36-DET11", det11_resources),
    ("C36-DET12", det12_end_to_end),
)


def all_checks() -> dict[str, object]:
    """Run and return all twelve checks in declared order."""
    return {check_id: check() for check_id, check in CHECKS}
