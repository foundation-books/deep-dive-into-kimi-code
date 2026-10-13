"""Focused tests for the printed Chapter 39 deterministic companion."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction

from memory_efficient_training.checks import all_checks
from memory_efficient_training.objects import ObjectKey, running_object
from memory_efficient_training.policies import (
    Lease,
    dropout_fixture,
    replay,
    validate_leases,
)
from memory_efficient_training.specialized import moe_fixture, muon_memory_counts


class MemoryEfficientTrainingReferenceTests(unittest.TestCase):
    def test_old_allocation_can_end_before_logical_replay_demand(self) -> None:
        old = ObjectKey("activation-a", 0, 0, "dropout", "activation", 0)
        new = ObjectKey("unrelated-b", 0, 0, "other", "workspace", 0)
        self.assertEqual(running_object(0, "activation", 1, (5,)).lifetime[1], 5)
        validate_leases((Lease("pool:A", old, 0, 2), Lease("pool:A", new, 2, 4)))
        closure = ((2, 3), 2, 0, "exact-int")
        self.assertEqual(replay(dropout_fixture, closure, closure), (4, 0))
        with self.assertRaises(ValueError):
            validate_leases((Lease("pool:A", old, 0, 3), Lease("pool:A", new, 2, 4)))
        with self.assertRaisesRegex(ValueError, "closure version mismatch"):
            replay(dropout_fixture, ((2, 3), 2, 1, "exact-int"), closure)

    def test_backward_consumer_timeline(self) -> None:
        from .backward_lifetimes import check_lifetimes

        check_lifetimes()

    def test_operator_is_part_of_object_identity(self) -> None:
        obj = running_object(1, "routed-input", 4, (12, 18))
        changed = replace(obj.key, operator=obj.key.operator + "-other")
        self.assertNotEqual(changed, obj.key)
        self.assertEqual(obj.full_units, 4)

    def test_all_named_checks_pass(self) -> None:
        results = all_checks()
        self.assertEqual(
            tuple(results), tuple(f"C37-DET{index:02d}" for index in range(1, 13))
        )
        self.assertTrue(all(item["status"] == "pass" for item in results.values()))

    def test_running_peak_witnesses(self) -> None:
        results = all_checks()
        self.assertEqual(results["C37-DET02"]["naive_peaks"], (57, 31))
        self.assertEqual(results["C37-DET09"]["balanced"], (32, 28))

    def test_moe_exact_values(self) -> None:
        fixture = moe_fixture()
        self.assertEqual(fixture["output"], (Fraction(1, 2), Fraction(2)))
        self.assertEqual(
            fixture["route_grads"], (Fraction(0), Fraction(-10), Fraction(8))
        )

    def test_muon_memory_counts(self) -> None:
        self.assertEqual(muon_memory_counts()["owner_peak_elements"], 16)
        self.assertEqual(muon_memory_counts()["gather_peak_elements"], 32)

    def test_compressed_checksum_is_tolerance_qualified(self) -> None:
        record = all_checks()["C37-DET12"]
        self.assertEqual(record["exact_checksum"], Fraction(28))
        self.assertEqual(record["compressed_checksum"], Fraction(113, 4))
        self.assertLessEqual(abs(record["compressed_checksum"] - 28), Fraction(1, 4))

    def test_physical_manifest_and_conservation(self) -> None:
        from .finite_fixture import complete_trace, finite_manifest, sweep_manifest

        exact, compressed = complete_trace(), complete_trace(True)
        self.assertEqual(exact.peaks, (32, 32, 14, 16))
        self.assertEqual(compressed.peaks, (32, 30, 10, 12))
        self.assertEqual((len(exact.events), len(compressed.events)), (255, 243))
        self.assertEqual(
            (exact.activation_traffic, compressed.activation_traffic), (56, 32)
        )
        self.assertEqual(exact.totals[-1], (18, 18, 5, 5))
        self.assertEqual(compressed.totals[-1], (18, 18, 5, 5))
        missing = tuple(e for e in finite_manifest() if e.object != "D.x")
        with self.assertRaises(ValueError):
            sweep_manifest(missing)

    def test_restore_requires_independent_block_release(self) -> None:
        from .finite_fixture import finite_manifest, sweep_manifest

        events = finite_manifest()
        # Keeping C's block alive makes its 4-unit restore hit 34.
        mutant = tuple(e for e in events if not (e.object == "C.b" and e.units == -2))
        with self.assertRaisesRegex(ValueError, "capacity"):
            sweep_manifest(mutant)

    def test_compact_families_preserve_every_physical_event(self) -> None:
        from .finite_fixture import finite_manifest
        from .schedule_spec import resource_sweep, schedule_blocks

        for compressed, peaks in ((False, (32, 32, 14, 16)), (True, (32, 30, 10, 12))):
            expanded = tuple(
                (o.object, o.domain, o.units)
                for block in schedule_blocks(compressed)
                for o in block.operations
            )
            original = tuple(
                (e.object, e.domain, e.units) for e in finite_manifest(compressed)
            )
            self.assertEqual(expanded, original)
            self.assertEqual(resource_sweep(compressed), peaks)

    def test_every_row_reaches_both_next_forward_replicas(self) -> None:
        from .numerical_fixture import row_update_witness

        for j in (0, 1):
            path = row_update_witness(j)
            self.assertEqual(path["update_count"], (1, 1, 1, 1))
            self.assertEqual(
                path["published_replicas"], (path["reference_parameters"],) * 2
            )
            self.assertEqual(
                path["traffic_elements_per_rank"],
                {
                    "candidate_ingress": 8,
                    "direction_delivery": 8,
                    "parameter_publication": 16,
                },
            )

    def test_optimizer_readiness_requires_actual_producer_events(self) -> None:
        from .finite_fixture import finite_manifest, sweep_manifest

        events = finite_manifest()
        for obj, action in (
            ("G0.0", "8-coordinate RS input"),
            ("M0", "candidate 8 local"),
            ("M1", "publish updated rows"),
        ):
            removed = next(
                e
                for e in reversed(events)
                if e.object == obj and e.action.startswith(action)
            )
            with self.assertRaisesRegex(ValueError, "unready prerequisite"):
                sweep_manifest(tuple(e for e in events if e != removed))

    def test_printed_compression_and_full_reduction_inputs(self) -> None:
        from .numerical_fixture import compression_witness, gradient_epochs

        witness = compression_witness()
        self.assertEqual(witness["codes"], (3, -3, 9, 0))
        self.assertEqual(witness["activation_error"], Fraction(1, 16))
        self.assertEqual(witness["gradient_error"], Fraction(3, 32))
        self.assertEqual(sum(witness["components"]), witness["checksum"])
        epochs = gradient_epochs()
        indices = epochs["epoch_coordinates"]
        self.assertEqual(sorted(indices[0] + indices[1]), list(range(16)))
        self.assertEqual(
            epochs["owner_shards"][0] + epochs["owner_shards"][1],
            epochs["reference_gradient"],
        )

    def test_negative_mutants_are_exercised(self) -> None:
        results = all_checks()
        required_flags = {
            "C37-DET01": ("stale_microbatch_rejected",),
            "C37-DET03": ("discard_rejected", "missing_metadata_rejected"),
            "C37-DET04": ("closure_mutants_rejected",),
            "C37-DET05": ("stale_scale_rejected",),
            "C37-DET06": ("alias_overlap_rejected",),
            "C37-DET07": ("route_order_mutant_rejected",),
            "C37-DET08": ("early_release_rejected",),
            "C37-DET10": ("duplicate_epoch_rejected", "missing_replica_rejected"),
            "C37-DET11": ("shard_order_mutant_rejected",),
            "C37-DET12": ("out_of_tolerance_mutant_rejected",),
        }
        for check_id, flags in required_flags.items():
            for flag in flags:
                value = results[check_id][flag]
                if isinstance(value, dict):
                    self.assertTrue(all(value.values()), f"{check_id}:{flag}")
                else:
                    self.assertTrue(value, f"{check_id}:{flag}")


if __name__ == "__main__":
    unittest.main()
