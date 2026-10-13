"""Deterministic evidence checks for the Chapter 34 semantic oracle."""

from __future__ import annotations

from . import faults, reference as ref


def _owner_rejection() -> dict[str, object]:
    expected = ref.OwnerTag(
        "A", "prefix:5", 4, None, "model-r1", "token-major", "bf16", "accepted"
    )
    foreign = ref.OwnerTag(
        "B", "prefix:3", 4, None, "model-r1", "token-major", "bf16", "accepted"
    )
    rejected = False
    fields: tuple[str, ...] = ()
    try:
        ref.validate_owner(expected, foreign)
    except ValueError as error:
        rejected = True
        fields = tuple(str(error).split(": ", 1)[1].split(", "))
    expected_fixture = (
        expected.endpoint_or_forward == "prefix:5" and expected.mixing_point is None
    )
    return ref.check_record(
        expected_fixture
        and rejected
        and {"request", "endpoint_or_forward"} <= set(fields),
        rejected=rejected,
        differing_fields=list(fields),
    )


def _layout_round_trip() -> dict[str, object]:
    logical = (
        (ref.f(1), ref.f(2), ref.f(3), ref.f(4)),
        (ref.f(5), ref.f(6), ref.f(7), ref.f(8)),
        (ref.f(9), ref.f(10), ref.f(11), ref.f(12)),
    )
    physical = ref.logical_to_feature_major(logical)
    return ref.check_record(
        ref.feature_major_to_logical(physical) == logical
        and physical[1][2] == logical[2][1],
        logical_shape=[3, 4],
        physical_shape=[4, 3],
        non_symmetric_witness=str(physical[1][2]),
    )


def _packed_isolation() -> dict[str, object]:
    requests = {
        "A": tuple((ref.f(index), ref.f(index + 1)) for index in range(1, 6)),
        "B": tuple((ref.f(-index), ref.f(2 * index)) for index in range(1, 4)),
    }
    gate = (ref.f(1, 2), ref.f(3, 4))
    baseline = ref.packed_gated_causal(requests, gate)
    perturbed = dict(requests)
    perturbed["A"] = requests["A"][:-1] + ((ref.f(10_000), ref.f(-10_000)),)
    changed = ref.packed_gated_causal(perturbed, gate)
    return ref.check_record(
        baseline["B"] == changed["B"] and baseline["A"] != changed["A"],
        offsets=list(ref.PACK_OFFSETS),
        unchanged_request="B",
        perturbed_request="A",
    )


def _hybrid_dispatch() -> dict[str, object]:
    schedule = ref.hybrid_schedule(93)
    return ref.check_record(
        ref.hybrid_schedule(5) == ref.TOY_SCHEDULE
        and schedule.count("K") == 69
        and schedule.count("G") == 24
        and schedule[-1] == "G",
        toy=list(ref.TOY_SCHEDULE),
        kda_layers=schedule.count("K"),
        global_layers=schedule.count("G"),
        terminal=schedule[-1],
    )


def _full_attnres() -> dict[str, object]:
    weights, output = ref.full_attnres_example()
    return ref.check_record(
        weights == (ref.f(1, 6), ref.f(1, 3), ref.f(1, 2))
        and sum(weights, ref.f(0)) == 1
        and output == (ref.f(7, 6), ref.f(5, 3)),
        weights=[str(value) for value in weights],
        output=[str(value) for value in output],
    )


def _block_sources() -> dict[str, object]:
    completed, tail = ref.block_index_partition(10, 4)
    target9 = ref.block_sources_before(9, 4)
    target10 = ref.block_sources_before(10, 4)
    blocks = ref.trace_block_values()
    covered = tuple(index for group in completed for index in group) + tail
    return ref.check_record(
        completed == ((1, 2, 3, 4), (5, 6, 7, 8))
        and tail == (9, 10)
        and target9[1] == ()
        and target10[1] == (9,)
        and covered == tuple(range(1, 11))
        and len(set(covered)) == 10
        and blocks
        == ((ref.f(1), ref.f(3)), (ref.f(2), ref.f(2)), (ref.f(3), ref.f(-1))),
        completed=[list(group) for group in completed],
        tail=list(tail),
        block_values=[[str(item) for item in value] for value in blocks],
    )


def _stable_merge() -> dict[str, object]:
    items = (
        (ref.f(1), (ref.f(1),)),
        (ref.f(2), (ref.f(4),)),
        (ref.f(3), (ref.f(-2),)),
    )
    singletons = tuple(ref.stable_summary((item,)) for item in items)
    left = ref.merge_summaries(
        ref.merge_summaries(singletons[0], singletons[1]), singletons[2]
    )
    right = ref.merge_summaries(
        singletons[0], ref.merge_summaries(singletons[1], singletons[2])
    )
    grouped = ref.merge_summaries(
        ref.stable_summary(items[:2]), ref.stable_summary(items[2:])
    )
    return ref.check_record(
        left == right == grouped
        and ref.summary_output(grouped) == (ref.f(1, 2),)
        and ref.merge_summaries(None, grouped) == grouped,
        represented_output="1/2",
        associative=left == right,
        identity=True,
    )


def _two_phase_block() -> dict[str, object]:
    fixed = (
        (ref.f(1), (ref.f(1), ref.f(0))),
        (ref.f(3), (ref.f(0), ref.f(2))),
        (ref.f(2), (ref.f(2), ref.f(-1))),
    )
    partial_states = (
        None,
        (ref.f(4), (ref.f(-1), ref.f(3))),
        (ref.f(5), (ref.f(4), ref.f(4))),
    )
    matches = tuple(
        ref.block_two_phase(fixed, partial) == ref.block_direct(fixed, partial)
        for partial in partial_states
    )
    return ref.check_record(
        all(matches),
        partial_lengths=[0, 1, 2],
        exact_matches=list(matches),
    )


def _sequence_shards() -> dict[str, object]:
    unsharded = tuple(
        ref.weighted_mixture(ref.token_row_fixture(ref.PACK_ORIGINS)[origin])
        for origin in ref.PACK_ORIGINS
    )
    sharded = ref.sequence_sharded_attnres(
        ref.PACK_ORIGINS, (ref.PACK_ORIGINS[:4], ref.PACK_ORIGINS[4:])
    )
    replicated, per_rank = ref.depth_storage_bytes(4, 8, 8, 2, 2)
    return ref.check_record(
        sharded == unsharded and replicated == 512 and per_rank == 256,
        origins=list(ref.PACK_ORIGINS),
        replicated_bytes_per_rank=replicated,
        sharded_bytes_per_rank=per_rank,
    )


def _accepted_continuation() -> dict[str, object]:
    a = ref.make_bundle("A", 5)
    b = ref.make_bundle("B", 3)
    candidate = ref.stage_continuation(a, token=6)
    rejected = ref.commit_candidate(a, candidate, accepted=False)
    accepted = ref.commit_candidate(a, candidate, accepted=True)
    prefill_history = ref.consume_depth_history(
        ref.DepthHistory("prefill:A+B", ref.TRACE_OUTPUTS)
    )
    continuation_history = ref.DepthHistory("decode:A6", ref.TRACE_OUTPUTS[:2])
    fault_records = faults.fault_witnesses()
    faults_pass = all(record["status"] == "pass" for record in fault_records.values())
    return ref.check_record(
        rejected == a
        and accepted.endpoint == 6
        and accepted.component_endpoints == (6, 6, 6, 6, 6)
        and b.endpoint == 3
        and prefill_history.consumed
        and continuation_history.forward_id != prefill_history.forward_id
        and not continuation_history.consumed
        and faults_pass,
        accepted_a_endpoint=accepted.endpoint,
        unchanged_b_endpoint=b.endpoint,
        component_endpoints=list(accepted.component_endpoints),
        released_forward=prefill_history.forward_id,
        fresh_forward=continuation_history.forward_id,
        fault_injections=fault_records,
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Return the ten evidence outcomes in stable protocol order."""

    return {
        "owner_rejection": _owner_rejection(),
        "layout_round_trip": _layout_round_trip(),
        "packed_isolation": _packed_isolation(),
        "hybrid_dispatch": _hybrid_dispatch(),
        "full_attnres": _full_attnres(),
        "block_sources": _block_sources(),
        "stable_merge": _stable_merge(),
        "two_phase_block": _two_phase_block(),
        "sequence_shards": _sequence_shards(),
        "accepted_continuation": _accepted_continuation(),
    }
