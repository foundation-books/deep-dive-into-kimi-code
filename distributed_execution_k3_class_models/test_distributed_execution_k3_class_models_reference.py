"""Focused tests for the Chapter 37 deterministic companion."""

from __future__ import annotations

import unittest

from distributed_execution_k3_class_models.checks import all_checks
from distributed_execution_k3_class_models.faults import wrong_shape
from distributed_execution_k3_class_models.schedule import (
    canonical_collective_trace,
    first_collective_mismatch,
    pipeline_events,
    pipeline_resource_schedule,
    resource_fixture,
    validate_resource_schedule,
)
from distributed_execution_k3_class_models.semantics import (
    balanced_destinations,
    canonical_routes,
    execution_loads,
    expert_gradients,
    fixed_destinations,
    kda_fixture,
    tp_dense_and_sharded,
)
from distributed_execution_k3_class_models.topology import (
    P,
    V,
    Coordinate,
    decode,
    groups_for_rank,
    placement,
)


class DistributedExecutionReferenceTests(unittest.TestCase):
    def test_backward_microbatch_and_replica_scaling(self) -> None:
        from fractions import Fraction
        from .reader_interfaces import gradient_from_local_means, scalar_owner_witness

        moved, home = scalar_owner_witness(), scalar_owner_witness(False)
        self.assertEqual(moved["replica_numerators"], (160, 224))
        self.assertEqual(moved["normalizer"], 64)
        self.assertEqual(moved["gradient"], 6)
        self.assertEqual(moved["updated"], Fraction(47, 5))
        for field in ("gradient", "updated", "normalizer", "replica_numerators"):
            self.assertEqual(moved[field], home[field])
        self.assertEqual(moved["update_count"], 1)
        self.assertEqual(moved["permanent_revisions"], {1: 8, 5: 8, 17: 8, 21: 8})
        self.assertEqual(moved["context_partials"][(1, 2, 1)], (16, 16))
        means = ((Fraction(2), 8), (Fraction(4), 8), (Fraction(6), 8), (Fraction(8), 8))
        self.assertEqual(gradient_from_local_means(means), 5)
        # Unequal target masses forbid blindly averaging the two holders.
        unequal = ((Fraction(2), 3), (Fraction(8), 5), (Fraction(999), 0))
        self.assertEqual(gradient_from_local_means(unequal), Fraction(23, 4))
        self.assertNotEqual(gradient_from_local_means(unequal), Fraction(5))
        # A target unused by this parameter still contributes objective mass.
        self.assertEqual(
            gradient_from_local_means(((Fraction(8), 1), (Fraction(0), 7))), 1
        )

    def test_all_named_checks_pass(self) -> None:
        results = all_checks()
        self.assertEqual(
            tuple(results), tuple(f"C36-DET{index:02d}" for index in range(1, 13))
        )
        self.assertTrue(all(result["status"] == "pass" for result in results.values()))

    def test_focus_rank_and_groups(self) -> None:
        self.assertEqual(decode(23), Coordinate(1, 0, 1, 1, 1))
        self.assertEqual(placement(23), (2, 7))
        self.assertEqual(groups_for_rank(23)["c"].members, (19, 23))

    def test_illustrated_pipeline_schedule_is_complete(self) -> None:
        events = pipeline_events()
        schedule = pipeline_resource_schedule()
        validate_resource_schedule(schedule, events)
        self.assertEqual(tuple(len(row) for row in schedule), (18, 18))
        for event in events.values():
            self.assertEqual(event.virtual_chunk, event.event_id.logical_chunk // P)
            self.assertLess(event.virtual_chunk, V)

    def test_resource_ledger_preserves_work_across_placements(self) -> None:
        main = resource_fixture()
        alternative = resource_fixture(True)
        for ledger in (main, alternative):
            self.assertEqual(ledger.useful_work, 128)
            self.assertEqual(ledger.resident_storage, 96)
            self.assertEqual(ledger.launches, 14)
            self.assertEqual(ledger.synchronizations, 9)
            self.assertEqual(ledger.duration_input, 17)
            self.assertIsNone(ledger.critical_path_bound)

    def test_tensor_and_context_witnesses(self) -> None:
        tensor = tp_dense_and_sharded()
        self.assertEqual(tensor["dense"], tensor["reduced"])
        self.assertNotEqual(kda_fixture()["ordered"], kda_fixture()["reversed"])

    def test_route_placement_preserves_semantics(self) -> None:
        routes = canonical_routes()
        fixed = fixed_destinations(routes)
        balanced = balanced_destinations(routes)
        self.assertEqual(execution_loads(fixed), (5, 3))
        self.assertEqual(execution_loads(balanced), (4, 4))
        self.assertEqual(
            expert_gradients(routes, fixed), expert_gradients(routes, balanced)
        )

    def test_collective_shape_mutant_is_localized(self) -> None:
        trace = canonical_collective_trace()
        self.assertEqual(
            first_collective_mismatch({19: trace, 23: wrong_shape(trace)}),
            (23, 0, "shape"),
        )

    def test_collective_version_fields_have_distinct_roles(self) -> None:
        from dataclasses import replace

        trace = canonical_collective_trace()
        for field, value, diagnosis in (
            ("epoch", 6, "epoch"),
            ("membership_epoch", 1, "membership_epoch"),
            ("parameter_revision", 6, "revision"),
            ("dtype", "fp32", "type"),
        ):
            changed = (replace(trace[0], **{field: value}), trace[1])
            self.assertEqual(
                first_collective_mismatch({19: trace, 23: changed}), (23, 0, diagnosis)
            )

    def test_workload_layers_depth_sources_and_cp_destinations(self) -> None:
        from .reader_interfaces import (
            cp_reconciliation_groups,
            depth_boundary_sources,
            layer_assignment,
            token_origin,
        )

        self.assertEqual(
            cp_reconciliation_groups(), ((1, 5), (3, 7), (17, 21), (19, 23))
        )
        self.assertEqual(cp_reconciliation_groups(True), ((1, 5), (17, 21)))
        self.assertEqual(
            [token_origin(u, i) for u in (0, 1) for i in range(1, 5)],
            [17, 17, 21, 21, 19, 19, 23, 23],
        )
        self.assertEqual(
            [layer_assignment(i) for i in range(1, 9)],
            [(0, 0, 0)] * 2 + [(1, 1, 0)] * 2 + [(2, 0, 1)] * 2 + [(3, 1, 1)] * 2,
        )
        for chunk in range(3):
            sources = depth_boundary_sources(chunk)
            indices = [i for values in sources.values() for i in values]
            self.assertEqual(sorted(indices), list(range(1, 4 * (chunk + 1) + 1)))
            self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(depth_boundary_sources(1)), {"X0", "b1"})

    def test_expanded_head_exchange_and_inverse(self) -> None:
        from .reader_interfaces import AttentionTag, mla_fixture, redistribute_attention

        for document, ranks in ((0, (25, 29)), (1, (27, 31))):
            result = mla_fixture(document)
            self.assertEqual(result["before"], result["restored"])
            before, after = result["before"], result["after"]
            for c, rank in enumerate(ranks):
                self.assertEqual(len(before[rank]), 8)
                self.assertEqual(
                    {tag.position for tag in before[rank]}, {2 * c + 1, 2 * c + 2}
                )
                self.assertEqual({tag.head for tag in after[rank]}, {c})
                self.assertEqual({tag.position for tag in after[rank]}, {1, 2, 3, 4})
            self.assertEqual(
                (result["remote_qkv_scalars"], result["remote_output_scalars"]), (24, 8)
            )
            for (u, query, head), keys in result["causal_keys"].items():
                self.assertEqual(keys, tuple((u, i, head) for i in range(1, query + 1)))
            malformed = dict(before)
            malformed[ranks[0]] = before[ranks[0]][1:]
            with self.assertRaises(ValueError):
                redistribute_attention(malformed, document, by_head=True)
            malformed[ranks[0]] = (AttentionTag(1 - document, 1, 0, 0),) + before[
                ranks[0]
            ][1:]
            with self.assertRaises(ValueError):
                redistribute_attention(malformed, document, by_head=True)

    def test_fixed_stage_order_full_finish_ledger(self) -> None:
        from .schedule import EventId, earliest_finishes, fixed_stage_order_events
        from .semantics import f

        events = fixed_stage_order_events()
        equal = earliest_finishes(events, {eid: f(1) for eid in events})
        unequal = earliest_finishes(
            events, {eid: f(2 if eid.logical_chunk == 3 else 1) for eid in events}
        )
        # Independent expected rows transcribed from the complete printed DAG.
        order = [("F", k) for k in range(4)] + [("B", k) for k in reversed(range(4))]
        expected_equal = (
            (1, 2, 3, 4, 5, 6, 8, 9),
            (2, 3, 4, 6, 7, 8, 9, 10),
            (5, 10, 11, 12, 13, 14, 16, 17),
            (7, 11, 12, 14, 15, 16, 17, 18),
        )
        expected_unequal = (
            (1, 2, 3, 5, 7, 8, 12, 13),
            (2, 3, 4, 9, 11, 12, 13, 14),
            (5, 14, 15, 17, 19, 20, 24, 25),
            (9, 15, 16, 21, 23, 24, 25, 26),
        )
        for m in range(4):
            keys = [EventId(direction, chunk, m) for direction, chunk in order]
            self.assertEqual(tuple(equal[eid] for eid in keys), expected_equal[m])
            self.assertEqual(tuple(unequal[eid] for eid in keys), expected_unequal[m])
        self.assertEqual(
            sum(cell is None for row in pipeline_resource_schedule() for cell in row), 4
        )

    def test_isolated_placement_inventory_is_reproducible(self) -> None:
        from .reader_interfaces import placement_witness

        self.assertEqual(placement_witness(), (18, {(0, 1): 4, (1, 0): 0}))
        self.assertEqual(placement_witness(True), (18, {(0, 1): 18, (1, 0): 0}))


if __name__ == "__main__":
    unittest.main()
