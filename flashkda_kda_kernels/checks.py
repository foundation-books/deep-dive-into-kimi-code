"""Deterministic evidence checks built from the Chapter 33 semantic oracle."""

from __future__ import annotations

from . import reference as ref


def _recurrence() -> dict[str, object]:
    _, outputs = ref.run_pack32()
    return ref.check_record(
        len(outputs) == 3 and sum(ref.PACK_LENGTHS) == 59,
        requests=len(outputs),
        tokens=sum(ref.PACK_LENGTHS),
        heads=ref.PACK_HEADS,
    )


def _layout() -> dict[str, object]:
    finals, _ = ref.run_pack32()
    physical = ref.logical_to_v_first(finals)
    return ref.check_record(
        ref.v_first_to_logical(physical) == finals,
        logical_shape=[3, 2, 3, 2],
        v_first_shape=[3, 2, 2, 3],
    )


def _packing() -> dict[str, object]:
    chunks = ref.packed_chunks(ref.PACK_LENGTHS, ref.PACK_CHUNK_SIZE)
    tails = [
        end - start
        for request in chunks
        for start, end in request
        if end - start < ref.PACK_CHUNK_SIZE
    ]
    return ref.check_record(
        ref.cumulative_offsets(ref.PACK_LENGTHS) == ref.PACK_OFFSETS,
        offsets=list(ref.PACK_OFFSETS),
        tail_lengths=tails,
    )


def _composition() -> dict[str, object]:
    summary, composed_final, direct_sum = ref.m32_composition_example()
    expected = ((ref.f(3, 2), ref.f(3, 2)), (ref.f(1), ref.f(1, 2)))
    return ref.check_record(
        composed_final == expected,
        composed_write_00=str(summary[1][0][0]),
        direct_sum_differs=direct_sum != summary[1],
    )


def _monoid() -> dict[str, object]:
    summaries = tuple(ref.token_summary(token) for token in ref.make_tokens(3, 1))
    left = ref.compose(summaries[2], ref.compose(summaries[1], summaries[0]))
    right = ref.compose(ref.compose(summaries[2], summaries[1]), summaries[0])
    noncommutative = ref.compose(summaries[1], summaries[0]) != ref.compose(
        summaries[0], summaries[1]
    )
    return ref.check_record(
        left == right and noncommutative,
        associative=left == right,
        noncommutative_witness=noncommutative,
    )


def _scan() -> dict[str, object]:
    tokens = ref.make_tokens(33, 0)
    fragments = tuple(
        ref.segment_summary(tokens[start : start + 11]) for start in (0, 11, 22)
    )
    initial = ref.zero(3, 2)
    scan = ref.exclusive_scan(fragments, initial)
    serial = tuple(ref.run_recurrent(tokens[:stop], initial)[1] for stop in (0, 11, 22))
    return ref.check_record(
        scan == serial,
        segments=3,
        segment_lengths=[11, 11, 11],
    )


def _handoff() -> dict[str, object]:
    finals, _ = ref.run_pack32()
    independent = finals[0][0] != finals[1][0] and finals[1][0] != finals[2][0]
    return ref.check_record(
        independent,
        independent_final_states=3,
        a_incoming_nonzero=any(ref.flatten_numbers(ref.running_initial_states()[0])),
    )


def _dispatch() -> dict[str, object]:
    admitted = ref.Workload("prefill", False, "bf16", 128, 128, True, True, True, 90)
    fallback = ref.Workload("training", True, "bf16", 128, 128, True, True, True, 90)
    admitted_path = ref.choose_schedule(admitted)
    fallback_path = ref.choose_schedule(fallback)
    return ref.check_record(
        admitted_path == "public_flashkda_forward"
        and fallback_path == "generic_chunkwise_kda",
        admitted=admitted_path,
        gradient_fallback=fallback_path,
    )


def _bytes() -> dict[str, object]:
    bf16 = ref.state_bytes(3, 16, 128, 128, 2)
    return ref.check_record(
        bf16 == 1_572_864,
        bf16_state_bytes=bf16,
        fp32_state_bytes=ref.state_bytes(3, 16, 128, 128, 4),
        three_rank_fp32_fragments=ref.fragment_bytes(3, 16, 128, 128, 4),
    )


def _recovery() -> dict[str, object]:
    initial = ref.running_initial_states()[0][0]
    draft = ref.make_tokens(4, 0)
    replay_ok = all(
        ref.recover_accepted_state(initial, draft, accepted)
        == ref.run_recurrent(draft[:accepted], initial)[1]
        for accepted in range(5)
    )
    return ref.check_record(
        replay_ok,
        draft_positions=4,
        checked_acceptance_boundaries=5,
    )


def all_checks() -> dict[str, dict[str, object]]:
    """Return the ten evidence outcomes in stable protocol order."""

    return {
        "recurrence": _recurrence(),
        "layout": _layout(),
        "packing": _packing(),
        "composition": _composition(),
        "monoid": _monoid(),
        "scan": _scan(),
        "handoff": _handoff(),
        "dispatch": _dispatch(),
        "bytes": _bytes(),
        "recovery": _recovery(),
    }
