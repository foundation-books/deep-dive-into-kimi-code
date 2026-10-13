"""Tests for the exact Chapter 4 KDA reference semantics."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from . import reference


def assert_value_error(action: Callable[[], object]) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_one_step_example_is_exact() -> None:
    state, _, result = reference.one_step_example()
    assert state == ((Fraction(1), Fraction(2)), (Fraction(3), Fraction(4)))
    assert result.retained == (
        (Fraction(1, 2), Fraction(1)),
        (Fraction(3, 4), Fraction(1)),
    )
    assert result.prediction == (Fraction(1, 2), Fraction(1))
    assert result.error == (Fraction(3, 2), Fraction(-2))
    assert result.correction == (
        (Fraction(3, 4), Fraction(-1)),
        (Fraction(0), Fraction(0)),
    )
    assert result.state == (
        (Fraction(5, 4), Fraction(0)),
        (Fraction(3, 4), Fraction(1)),
    )
    assert result.output == (Fraction(27, 20), Fraction(4, 5))


def test_repeated_key_example_matches_outline() -> None:
    trace = reference.repeated_key_example()
    states = tuple(position[0].state for position in trace.positions)
    assert states == (
        ((Fraction(1, 2), Fraction(0)), (Fraction(0), Fraction(0))),
        ((Fraction(3, 8), Fraction(0)), (Fraction(0), Fraction(1, 2))),
        ((Fraction(3, 32), Fraction(1, 2)), (Fraction(0), Fraction(3, 8))),
    )


def test_running_inputs_obey_domains() -> None:
    sequence = reference.running_inputs()
    zero = ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0)))
    for position in sequence:
        for item in position:
            reference.validate_input(item, zero)


def test_chunk_partitions_match_recurrent_trace() -> None:
    sequence = reference.running_inputs()
    recurrent = reference.run_recurrent(sequence)
    for chunk_size in (1, 2, 3, 4, 6, 10, 12):
        chunked = reference.run_chunked(sequence, chunk_size)
        assert chunked.positions == recurrent.positions
        assert chunked.final_states == recurrent.final_states


def test_chunked_rejects_invalid_head_and_state_shapes() -> None:
    sequence = reference.running_inputs()
    initial_states = reference.run_recurrent(sequence[:1]).final_states
    actions = (
        lambda: reference.run_chunked(((),), 1),
        lambda: reference.run_chunked((sequence[0], sequence[1][:1]), 1),
        lambda: reference.run_chunked(
            sequence[:1], 1, initial_states=initial_states[:1]
        ),
        lambda: reference.run_chunked(
            sequence[:1], 1, initial_states=initial_states + (initial_states[0],)
        ),
    )
    for action in actions:
        assert_value_error(action)


def test_prefill_state_is_decode_initial_state() -> None:
    prefill = reference.run_recurrent(reference.running_inputs())
    decoded = reference.run_recurrent(
        (reference.decode_inputs(),), initial_states=prefill.final_states
    )
    assert decoded.positions[0][0].state != prefill.final_states[0]
    assert decoded.positions[0][1].state != prefill.final_states[1]


def test_future_mutation_leaves_prefix_unchanged() -> None:
    sequence = reference.as_float_sequence(reference.running_inputs())
    baseline = reference.run_recurrent(sequence)
    mutated = reference.run_recurrent(reference.mutate_suffix_values(sequence, 6, 9.0))
    assert baseline.positions[:6] == mutated.positions[:6]


def test_sequence_isolation_matches_independent_evaluation() -> None:
    first = reference.running_inputs()[:6]
    second = reference.mutate_suffix_values(reference.as_float_sequence(first), 0, 3.0)
    first_alone = reference.run_recurrent(first)
    first_in_logical_batch = reference.run_recurrent(first)
    second_in_logical_batch = reference.run_recurrent(second)
    assert first_in_logical_batch == first_alone
    assert second_in_logical_batch != first_alone


def test_affine_composition_is_associative_on_running_head() -> None:
    items = tuple(position[0] for position in reference.running_inputs()[:6])
    first = reference.segment_summary(items[:2])
    second = reference.segment_summary(items[2:4])
    third = reference.segment_summary(items[4:])
    left = reference.compose_affine(third, reference.compose_affine(second, first))
    right = reference.compose_affine(reference.compose_affine(third, second), first)
    assert left == right == reference.segment_summary(items)


def test_cost_ledger_values() -> None:
    assert reference.logical_state_elements(1, 2, 2, 2) == 8
    assert reference.logical_state_elements(8, 16, 128, 128) == 2_097_152
    assert reference.formal_crossover(64, 128) == 496
    assert reference.kda_chunk_flops(496, 64, 128) == reference.full_attention_flops(
        496, 128
    )


def main() -> None:
    """Run the dependency-free assertions when invoked as a script."""

    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for _, test in tests:
        test()


if __name__ == "__main__":
    main()
