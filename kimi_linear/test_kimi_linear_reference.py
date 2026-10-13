"""Tests for the dependency-free Chapter 5 hybrid reference."""

from __future__ import annotations

import unittest

from .reference import (
    RUNNING_SCHEDULE,
    cached_decode_trace,
    causal_access_matrix,
    empty_state,
    f,
    flatten_outputs,
    full_running_trace,
    hybrid_prefill_attention_flops,
    k3_schedule,
    logical_footprint,
    mixed_family_state,
    process_token,
    ratio_inventory,
    relative_global_cache_slope,
    replace_suffix,
    repeating_schedule,
    run_sequence,
    running_inputs,
    validate_causal_access_matrix,
    validate_complete_state,
    validate_repeating_inventory,
)


class ScheduleTests(unittest.TestCase):
    def test_running_inventory(self) -> None:
        observed = validate_repeating_inventory(
            RUNNING_SCHEDULE,
            ratio=3,
            complete_blocks=2,
            terminal_global=False,
        )
        self.assertEqual(
            (observed.layers, observed.kda_layers, observed.global_layers), (8, 6, 2)
        )

    def test_k3_inventory_and_terminal_global(self) -> None:
        observed = validate_repeating_inventory(
            k3_schedule(),
            ratio=3,
            complete_blocks=23,
            terminal_global=True,
        )
        self.assertEqual(
            (
                observed.layers,
                observed.kda_layers,
                observed.global_layers,
                observed.complete_blocks,
                observed.terminal_global,
            ),
            (93, 69, 24, 23, True),
        )

    def test_schedule_fault_is_detected(self) -> None:
        faulty = repeating_schedule(3, 2)[:-1]
        with self.assertRaises(ValueError):
            validate_repeating_inventory(
                faulty,
                ratio=3,
                complete_blocks=2,
                terminal_global=False,
            )

    def test_source_tested_ratio_inventories(self) -> None:
        expected = {0: 16, 1: 8, 3: 4, 7: 2, 15: 1}
        for ratio, global_layers in expected.items():
            self.assertEqual(ratio_inventory(16, ratio).global_layers, global_layers)


class StateTests(unittest.TestCase):
    def test_prefix_trace_owns_every_layer_memory(self) -> None:
        trace = full_running_trace()
        state = trace.final_state
        self.assertEqual(len(trace.positions), 10)
        self.assertEqual(state.prefix_length, 10)
        self.assertEqual(len(state.kda_states), 6)
        self.assertEqual(len(state.global_caches), 2)
        self.assertTrue(all(len(cache.records) == 10 for cache in state.global_caches))

    def test_cached_decode_equals_full_evaluation(self) -> None:
        _, continuation = cached_decode_trace()
        full = full_running_trace(include_decode=True)
        self.assertEqual(continuation.positions[0].output, full.positions[-1].output)
        self.assertEqual(continuation.final_state, full.final_state)

    def test_future_perturbation_is_causal(self) -> None:
        baseline = run_sequence(running_inputs())
        changed = run_sequence(replace_suffix(running_inputs(), 6, f(9)))
        self.assertEqual(
            flatten_outputs(run_sequence(running_inputs()[:6])),
            flatten_outputs(
                run_sequence(replace_suffix(running_inputs(), 6, f(9))[:6])
            ),
        )
        self.assertEqual(
            tuple(position.output for position in baseline.positions[:6]),
            tuple(position.output for position in changed.positions[:6]),
        )
        self.assertNotEqual(baseline.positions[-1].output, changed.positions[-1].output)

    def test_partial_restore_faults_are_detected(self) -> None:
        prefix9 = run_sequence(running_inputs()[:9]).final_state
        prefix10 = run_sequence(running_inputs()).final_state
        for mixed in (
            mixed_family_state(prefix9, prefix10, declared_prefix=10),
            mixed_family_state(prefix10, prefix9, declared_prefix=10),
        ):
            with self.assertRaises(ValueError):
                validate_complete_state(mixed)

    def test_global_overwrite_fault_is_detected(self) -> None:
        prefix1 = run_sequence(running_inputs()[:1]).final_state
        with self.assertRaises(ValueError):
            process_token(running_inputs()[1], prefix1, overwrite_global_cache=True)

    def test_causal_mask_faults_are_detected(self) -> None:
        validate_causal_access_matrix(causal_access_matrix(5))
        with self.assertRaises(ValueError):
            validate_causal_access_matrix(
                causal_access_matrix(5, include_diagonal=False)
            )
        with self.assertRaises(ValueError):
            validate_causal_access_matrix(causal_access_matrix(5, allow_future=True))

    def test_empty_state_is_complete(self) -> None:
        validate_complete_state(empty_state())


class AccountingTests(unittest.TestCase):
    def test_running_footprint(self) -> None:
        self.assertEqual(
            logical_footprint(0), logical_footprint(0).__class__(48, 0, 48)
        )
        self.assertEqual(
            logical_footprint(10), logical_footprint(10).__class__(48, 40, 88)
        )
        self.assertEqual(
            logical_footprint(11), logical_footprint(11).__class__(48, 44, 92)
        )

    def test_quarter_global_cache_slope(self) -> None:
        self.assertEqual(relative_global_cache_slope(RUNNING_SCHEDULE), f(1, 4))
        self.assertEqual(relative_global_cache_slope(k3_schedule()), f(24, 93))

    def test_prefill_ledger_is_positive(self) -> None:
        self.assertGreater(hybrid_prefill_attention_flops(128, 64, 128), 0)


if __name__ == "__main__":
    unittest.main()
