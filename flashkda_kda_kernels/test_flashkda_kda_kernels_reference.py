"""Checks for the Chapter 33 semantic reference."""

from __future__ import annotations

from fractions import Fraction

from . import reference as ref


def test_backward_pipeline() -> None:
    from .backward_checks import check_all

    check_all()


def test_pack_contract_and_partial_tails() -> None:
    assert ref.cumulative_offsets(ref.PACK_LENGTHS) == ref.PACK_OFFSETS
    assert ref.packed_chunks(ref.PACK_LENGTHS, 16) == (
        ((0, 16), (16, 19)),
        ((0, 16), (16, 32), (32, 33)),
        ((0, 7),),
    )


def test_exact_composition_and_direct_sum_counterexample() -> None:
    summary, final, direct_sum = ref.m32_composition_example()
    assert summary == (
        ((Fraction(1, 2), Fraction(1, 2)), (Fraction(0), Fraction(1, 2))),
        ((Fraction(1), Fraction(3, 2)), (Fraction(1), Fraction(1, 2))),
    )
    assert final == (
        (Fraction(3, 2), Fraction(3, 2)),
        (Fraction(1), Fraction(1, 2)),
    )
    assert direct_sum != summary[1]


def test_composition_is_associative_but_not_commutative() -> None:
    tokens = ref.make_tokens(3, 0)
    first, second, third = (ref.token_summary(token) for token in tokens)
    assert ref.compose(third, ref.compose(second, first)) == ref.compose(
        ref.compose(third, second), first
    )
    assert ref.compose(second, first) != ref.compose(first, second)


def test_segment_summary_matches_recurrence_for_nonzero_state() -> None:
    tokens = ref.make_tokens(11, 0)
    initial = ref.running_initial_states()[0][0]
    recurrent = ref.run_recurrent(tokens, initial)[1]
    assert ref.apply_summary(ref.segment_summary(tokens), initial) == recurrent


def test_exclusive_scan_recovers_b_rank_boundaries() -> None:
    tokens = ref.make_tokens(33, 0)
    fragments = tuple(
        ref.segment_summary(tokens[start : start + 11]) for start in (0, 11, 22)
    )
    initial = ref.zero(3, 2)
    incoming = ref.exclusive_scan(fragments, initial)
    assert incoming[0] == initial
    assert incoming[1] == ref.run_recurrent(tokens[:11], initial)[1]
    assert incoming[2] == ref.run_recurrent(tokens[:22], initial)[1]


def test_pack_states_are_isolated_and_have_declared_layout() -> None:
    finals, outputs = ref.run_pack32()
    assert tuple(len(head[0]) for head in finals) == (3, 3, 3)
    assert tuple(len(request[0]) for request in outputs) == ref.PACK_LENGTHS
    physical = ref.logical_to_v_first(finals)
    assert len(physical) == 3 and len(physical[0]) == 2
    assert len(physical[0][0]) == 2 and len(physical[0][0][0]) == 3
    assert ref.v_first_to_logical(physical) == finals


def test_layout_witness_detects_missing_transpose() -> None:
    finals, _ = ref.run_pack32()
    physical = ref.logical_to_v_first(finals)
    assert physical[0][0][0][1] == finals[0][0][1][0]
    assert physical[0][0] != finals[0][0]


def test_dispatch_preserves_public_backend_boundary() -> None:
    admitted = ref.Workload("prefill", False, "bf16", 128, 128, True, True, True, 90)
    assert ref.choose_schedule(admitted) == "public_flashkda_forward"
    variants = (
        ref.Workload("training", True, "bf16", 128, 128, True, True, True, 90),
        ref.Workload("prefill", False, "fp16", 128, 128, True, True, True, 90),
        ref.Workload("prefill", False, "bf16", 64, 128, True, True, True, 90),
        ref.Workload("prefill", False, "bf16", 128, 128, False, True, True, 90),
        ref.Workload("prefill", False, "bf16", 128, 128, True, True, True, 89),
    )
    assert {ref.choose_schedule(item) for item in variants} == {"generic_chunkwise_kda"}
    cp = ref.Workload("training", True, "bf16", 128, 128, False, True, True, 90, True)
    assert ref.choose_schedule(cp) == "generic_kda_context_parallel"


def test_state_and_fragment_byte_ledgers() -> None:
    assert ref.state_bytes(3, 16, 128, 128, 2) == 1_572_864
    assert ref.state_bytes(3, 16, 128, 128, 4) == 3_145_728
    ledger = ref.fragment_bytes(3, 16, 128, 128, 4)
    assert ledger == {
        "local_fragment": 2_097_152,
        "received_per_rank": 4_194_304,
        "materialized_per_rank": 6_291_456,
        "aggregate_received": 12_582_912,
    }


def test_projected_replay_matches_snapshot_at_every_acceptance() -> None:
    initial = ref.running_initial_states()[0][0]
    candidates = ref.make_tokens(4, 0)
    snapshots = [initial]
    state = initial
    for token in candidates:
        state = ref.kda_step(state, token).state
        snapshots.append(state)
    for accepted, snapshot in enumerate(snapshots):
        assert ref.recover_accepted_state(initial, candidates, accepted) == snapshot
