"""Checks for the Chapter 34 dependency-free semantic reference."""

from __future__ import annotations

from collections.abc import Callable

from . import faults, reference as ref


def test_backward_decomposition() -> None:
    from .backward_checks import check_all

    check_all()


def _assert_value_error(fragment: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError as error:
        assert fragment in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_owner_tag_rejects_same_shape_foreign_state() -> None:
    expected = ref.OwnerTag("A", "prefix:5", 4, None, "r1", "token", "bf16", "accepted")
    foreign = ref.OwnerTag("B", "prefix:3", 4, None, "r1", "token", "bf16", "accepted")
    _assert_value_error("request", lambda: ref.validate_owner(expected, foreign))


def test_compact_record_layout_round_trip_is_value_preserving() -> None:
    logical = (
        (ref.f(1), ref.f(2), ref.f(3), ref.f(4)),
        (ref.f(5), ref.f(6), ref.f(7), ref.f(8)),
        (ref.f(9), ref.f(10), ref.f(11), ref.f(12)),
    )
    physical = ref.logical_to_feature_major(logical)
    assert physical[1][2] == logical[2][1]
    assert ref.feature_major_to_logical(physical) == logical


def test_packed_gated_attention_is_request_local_and_causal() -> None:
    requests = {
        "A": tuple((ref.f(i), ref.f(i + 1)) for i in range(1, 6)),
        "B": tuple((ref.f(-i), ref.f(2 * i)) for i in range(1, 4)),
    }
    gate = (ref.f(1, 2), ref.f(3, 4))
    baseline = ref.packed_gated_causal(requests, gate)
    changed = dict(requests)
    changed["A"] = requests["A"][:-1] + ((ref.f(10_000), ref.f(-10_000)),)
    perturbed = ref.packed_gated_causal(changed, gate)
    assert baseline["B"] == perturbed["B"]
    assert baseline["A"] != perturbed["A"]
    assert baseline["A"][0] == (
        gate[0] * requests["A"][0][0],
        gate[1] * requests["A"][0][1],
    )


def test_hybrid_dispatch_preserves_terminal_global_exception() -> None:
    assert ref.hybrid_schedule(5) == ("K", "K", "K", "G", "G")
    schedule = ref.hybrid_schedule(93)
    assert (schedule.count("K"), schedule.count("G"), schedule[-1]) == (69, 24, "G")
    _assert_value_error("four times", lambda: ref.hybrid_schedule(92))


def test_full_attnres_exact_weights_and_output() -> None:
    weights, output = ref.full_attnres_example()
    assert weights == (ref.f(1, 6), ref.f(1, 3), ref.f(1, 2))
    assert all(weight > 0 for weight in weights)
    assert sum(weights, ref.f(0)) == 1
    assert output == (ref.f(7, 6), ref.f(5, 3))


def test_block_sources_are_disjoint_exhaustive_and_keep_tail() -> None:
    completed, tail = ref.block_index_partition(10, 4)
    assert completed == ((1, 2, 3, 4), (5, 6, 7, 8))
    assert tail == (9, 10)
    assert ref.block_sources_before(9, 4)[1] == ()
    assert ref.block_sources_before(10, 4)[1] == (9,)
    covered = tuple(index for group in completed for index in group) + tail
    assert covered == tuple(range(1, 11))
    assert ref.trace_block_values() == (
        (ref.f(1), ref.f(3)),
        (ref.f(2), ref.f(2)),
        (ref.f(3), ref.f(-1)),
    )


def test_stable_summary_merge_is_associative_with_identity() -> None:
    items = (
        (ref.f(1), (ref.f(1),)),
        (ref.f(2), (ref.f(4),)),
        (ref.f(3), (ref.f(-2),)),
    )
    summaries = tuple(ref.stable_summary((item,)) for item in items)
    left = ref.merge_summaries(
        ref.merge_summaries(summaries[0], summaries[1]), summaries[2]
    )
    right = ref.merge_summaries(
        summaries[0], ref.merge_summaries(summaries[1], summaries[2])
    )
    assert left == right
    assert ref.summary_output(left) == (ref.f(1, 2),)
    assert ref.merge_summaries(None, left) == left


def test_two_phase_block_matches_same_variant_direct_oracle() -> None:
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
    for partial in partial_states:
        assert ref.block_two_phase(fixed, partial) == ref.block_direct(fixed, partial)


def test_sequence_shards_reconstruct_unsharded_rows_and_byte_ledger() -> None:
    fixture = ref.token_row_fixture(ref.PACK_ORIGINS)
    expected = tuple(
        ref.weighted_mixture(fixture[origin]) for origin in ref.PACK_ORIGINS
    )
    actual = ref.sequence_sharded_attnres(
        ref.PACK_ORIGINS, (ref.PACK_ORIGINS[:4], ref.PACK_ORIGINS[4:])
    )
    assert actual == expected
    assert ref.depth_storage_bytes(4, 8, 8, 2, 2) == (512, 256)
    _assert_value_error(
        "disjoint and exhaustive",
        lambda: ref.sequence_sharded_attnres(
            ref.PACK_ORIGINS, (ref.PACK_ORIGINS[:4], ref.PACK_ORIGINS[3:])
        ),
    )


def test_accepted_continuation_is_atomic_and_depth_history_is_fresh() -> None:
    a = ref.make_bundle("A", 5)
    b = ref.make_bundle("B", 3)
    candidate = ref.stage_continuation(a, token=6)
    _assert_value_error(
        "extend the accepted endpoint", lambda: ref.stage_continuation(a, token=7)
    )
    assert ref.commit_candidate(a, candidate, False) == a
    committed = ref.commit_candidate(a, candidate, True)
    assert committed.endpoint == 6
    assert committed.component_endpoints == (6, 6, 6, 6, 6)
    assert b.endpoint == 3
    old = ref.consume_depth_history(ref.DepthHistory("prefill:A+B", ref.TRACE_OUTPUTS))
    new = ref.DepthHistory("decode:A6", ref.TRACE_OUTPUTS[:2])
    assert old.consumed and not new.consumed and old.forward_id != new.forward_id
    witnesses = faults.fault_witnesses()
    assert tuple(witnesses) == faults.FAULT_IDS
    assert len(witnesses) == 6
    assert all(record["status"] == "pass" for record in witnesses.values())
    assert all(
        record["expected_rejection"] in record["observed"]
        for record in witnesses.values()
    )
