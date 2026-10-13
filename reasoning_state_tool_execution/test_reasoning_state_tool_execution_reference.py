"""Tests for the dependency-free Chapter 28 teaching oracle."""

from __future__ import annotations

from dataclasses import asdict, replace
import unittest

from .reference import (
    CALL_A,
    EFFECT_CLASSES,
    OBSERVATION_A,
    STUDIO_A_FIRST_ORDER,
    STUDIO_B_FIRST_ORDER,
    STUDIO_CHECKPOINTS,
    STUDIO_CRASH_PREFIX,
    STUDIO_EVENTS,
    Access,
    Event,
    ExecutionContractError,
    all_checks,
    all_topological_orders,
    conflict_edges,
    conflicts,
    canonical_json,
    digest,
    fold_events,
    idempotency_witness,
    preserved_history_errors,
    project_result,
    provider_order,
    recovery_action,
    schedule_batches,
    topological_order,
)


class Chapter27ReferenceTests(unittest.TestCase):
    def test_d01_event_graph_identity_parents_and_cycle(self) -> None:
        ordered = topological_order(reversed(STUDIO_EVENTS))
        self.assertEqual((ordered[0].event_id, ordered[-1].event_id), ("e01", "e14"))
        with self.assertRaisesRegex(ExecutionContractError, "duplicate event ID"):
            fold_events(STUDIO_EVENTS + (STUDIO_EVENTS[0],))
        with self.assertRaisesRegex(ExecutionContractError, "unknown parents"):
            fold_events((Event("x", "history.append", "x", {"message": {}}, ("y",)),))
        cyclic = (
            Event("x", "history.append", "x", {"message": {}}, ("y",)),
            Event("y", "history.append", "y", {"message": {}}, ("x",)),
        )
        with self.assertRaisesRegex(ExecutionContractError, "cycle"):
            fold_events(cyclic)

    def test_d02_fold_is_equal_across_two_linear_extensions(self) -> None:
        orders = all_topological_orders(STUDIO_EVENTS)
        self.assertEqual(len(orders), 924)
        self.assertEqual(
            len({fold_events(order, ordered=True).state_digest() for order in orders}),
            1,
        )
        left = fold_events(STUDIO_A_FIRST_ORDER, ordered=True)
        right = fold_events(STUDIO_B_FIRST_ORDER, ordered=True)
        self.assertEqual(left.canonical_value(), right.canonical_value())
        self.assertEqual(left.state_digest(), right.state_digest())
        with self.assertRaisesRegex(ExecutionContractError, "precedes parents"):
            fold_events(tuple(reversed(STUDIO_EVENTS)), ordered=True)

    def test_d03_preserved_reasoning_and_causal_closure(self) -> None:
        state = fold_events(STUDIO_EVENTS)
        self.assertEqual(preserved_history_errors(state), ())
        self.assertIn("reasoning_content", state.history[0])
        self.assertEqual(
            [row["tool_call_id"] for row in state.history[-1]["tool_results"]],
            ["call-a", "call-b"],
        )

    def test_d04_binding_environment_and_admission(self) -> None:
        state = fold_events(STUDIO_EVENTS[:8])
        self.assertEqual(set(state.registry), {"fetch_evidence", "run_release_check"})
        self.assertEqual({a.status for a in state.attempts.values()}, {"dispatched"})
        wrong = replace(CALL_A, environment_digest=digest({"wrong": True}))
        corrupt = replace(STUDIO_EVENTS[2], payload={"call": asdict(wrong)})
        with self.assertRaisesRegex(
            ExecutionContractError, "environment digest mismatch"
        ):
            fold_events(STUDIO_EVENTS[:2] + (corrupt,))

    def test_d05_call_attempt_and_monotone_lifecycle(self) -> None:
        state = fold_events(STUDIO_EVENTS)
        self.assertEqual(len(state.calls), 2)
        self.assertEqual(len(state.attempts), 2)
        self.assertEqual(state.attempts["attempt-a-1"].status, "succeeded")
        self.assertEqual(state.attempts["attempt-b-1"].status, "unknown")
        changed = {**CALL_A.arguments, "source_version": "studio@v8"}
        self.assertNotEqual(digest(changed), CALL_A.argument_digest)

    def test_d06_effect_classes_and_crash_recovery(self) -> None:
        self.assertEqual(
            EFFECT_CLASSES,
            ("pure", "idempotent", "deduplicated", "compensatable", "nonrepeatable"),
        )
        self.assertTrue(idempotency_witness("keyed-insert")["idempotent"])
        self.assertFalse(idempotency_witness("unkeyed-increment")["idempotent"])
        self.assertNotIn("retry-same-key", recovery_action("unknown", "nonrepeatable"))
        crash = fold_events(STUDIO_CRASH_PREFIX)
        self.assertEqual(crash.attempts["attempt-b-1"].status, "dispatched")
        self.assertEqual(crash.attempts["attempt-b-1"].receipt, "receipt-b-1")
        final = fold_events(STUDIO_EVENTS)
        self.assertTrue(final.observations["observation-b-1"].recovered)

    def test_d07_projection_preserves_status_and_provenance(self) -> None:
        raw = {
            "executor_status": "completed",
            "domain_status": "incomplete",
            "blocking_ids": ["D3"],
            "diagnostics": ["x" * 1000, "y" * 1000],
        }
        result = project_result(
            raw,
            budget=700,
            raw_reference="artifact://report",
            policy_version="projection-v1",
        )
        self.assertTrue(result["truncated"])
        self.assertEqual(result["domain_status"], "incomplete")
        self.assertEqual(result["raw_digest"], digest(raw))
        self.assertLessEqual(len(canonical_json(result).encode("utf-8")), 700)
        tampered = replace(OBSERVATION_A, raw={**OBSERVATION_A.raw, "present": True})
        event = replace(STUDIO_EVENTS[10], payload={"observation": asdict(tampered)})
        with self.assertRaisesRegex(
            ExecutionContractError, "raw payload digest mismatch"
        ):
            fold_events(STUDIO_EVENTS[:10] + (event,))

    def test_d08_conflict_graph_schedule_and_provider_order(self) -> None:
        accesses = (
            Access("A", frozenset({"x"}), frozenset(), 1),
            Access("B", frozenset({"x"}), frozenset(), 2),
            Access("C", frozenset(), frozenset({"x"}), 3),
            Access("D", frozenset(), frozenset({"y"}), 4),
            Access("E", frozenset({"x", "y"}), frozenset(), 5),
        )
        self.assertEqual(
            conflict_edges(accesses),
            (("A", "C"), ("B", "C"), ("C", "E"), ("D", "E")),
        )
        self.assertEqual(len(schedule_batches(accesses)), 2)
        self.assertFalse(conflicts(accesses[0], accesses[1]))
        self.assertTrue(conflicts(accesses[0], accesses[2]))
        self.assertTrue(
            conflicts(
                Access("F", frozenset(), frozenset({"z"}), 6),
                Access("G", frozenset(), frozenset({"z"}), 7),
            )
        )
        self.assertFalse(conflicts(accesses[2], accesses[3]))
        self.assertEqual(
            provider_order(("E", "B", "A", "D", "C"), accesses),
            ("A", "B", "C", "D", "E"),
        )

    def test_d09_invalid_transitions_are_rejected(self) -> None:
        missing = STUDIO_EVENTS[:3] + (
            Event(
                "bad",
                "attempt.dispatch",
                "absent",
                {"attempt_id": "absent"},
                ("e03",),
            ),
        )
        with self.assertRaisesRegex(ExecutionContractError, "unknown attempt"):
            fold_events(missing)

    def test_d10_handoff_and_all_checks(self) -> None:
        self.assertEqual(
            tuple(STUDIO_CHECKPOINTS), ("K0", "K1", "K2", "K3", "K4", "K5")
        )
        state = fold_events(STUDIO_EVENTS)
        self.assertEqual(state.pending, ["D3@new-source-version"])
        self.assertEqual(state.history[-1]["content"]["verdict"], "not_ready")
        checks = all_checks()
        self.assertEqual(len(checks), 10)
        self.assertEqual(
            [row["check_id"] for row in checks.values()],
            [f"CH27-D{index:02d}" for index in range(1, 11)],
        )
        self.assertTrue(all(row["status"] == "pass" for row in checks.values()))
        handoff = checks["handoff"]
        self.assertEqual(handoff["checkpoint_count"], 6)
        self.assertEqual(len(handoff["invariant_coverage"]), 20)


if __name__ == "__main__":
    unittest.main()
