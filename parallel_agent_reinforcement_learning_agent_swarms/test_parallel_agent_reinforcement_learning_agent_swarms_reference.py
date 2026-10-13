"""Tests for the Chapter 26 deterministic reference fixtures."""

from __future__ import annotations

import unittest

from . import reference


class ParallelAgentReinforcementLearningReferenceTests(unittest.TestCase):
    def test_all_checks_pass_and_have_stable_ids(self) -> None:
        checks = reference.all_checks()
        self.assertEqual(len(checks), 10)
        self.assertEqual(
            [result["check_id"] for result in checks.values()],
            [f"CH25-D{index:02d}" for index in range(1, 11)],
        )
        self.assertTrue(all(result["status"] == "pass" for result in checks.values()))

    def test_ready_frontier_waits_for_d3(self) -> None:
        result = reference.graph_readiness_check()
        self.assertEqual(result["post_partial_frontier"], ["D3_REPAIR", "RC", "RV"])
        self.assertNotIn("J", result["post_partial_frontier"])
        self.assertTrue(result["reverse_repair_edge_rejected_as_cycle"])

    def test_work_and_critical_ledgers(self) -> None:
        result = reference.work_critical_check()
        self.assertEqual(
            (result["canonical_work"], result["canonical_critical_steps"]), (32, 16)
        )
        self.assertEqual(
            (result["long_repair_work"], result["long_repair_critical_steps"]), (37, 20)
        )
        self.assertEqual(result["matched_long_repair_speedup"], "37/20")

    def test_critical_tie_is_a_set(self) -> None:
        result = reference.critical_tie_check()
        self.assertEqual(result["critical_branch_sets"][1], ["RC", "RV"])

    def test_reward_endpoint_is_performance_only(self) -> None:
        result = reference.reward_annealing_check()
        self.assertEqual(result["early_totals"]["valid_swarm"], "34/25")
        self.assertEqual(result["exploration_totals"]["spurious_fanout"], "9/2")
        self.assertEqual(result["final_totals"]["valid_swarm"], "9/10")

    def test_process_exit_is_not_valid_completion(self) -> None:
        result = reference.false_parallelism_check()
        self.assertEqual(result["paraphrase_worker_completion_rate"], "0")
        self.assertEqual(result["process_exit_rate"], "1")

    def test_context_accounting_keeps_local_work(self) -> None:
        result = reference.context_ledger_check()
        self.assertEqual(result["sharded_orchestrator_context"], 2_800)
        self.assertEqual(result["minimum_counted_processing"], 8_800)
        self.assertFalse(result["local_work_is_free"])

    def test_version_conflict_requests_repair(self) -> None:
        result = reference.aggregation_check()
        self.assertEqual(result["pre_repair_disposition"], "version_conflict")
        self.assertEqual(result["post_repair_unique_satisfied_coverage"], 1)

    def test_frontier_preserves_incomparability(self) -> None:
        result = reference.frontier_check()
        self.assertTrue(result["valid_dominates_sequential"])
        self.assertTrue(result["valid_and_over_sharded_incomparable"])

    def test_handoff_inventory(self) -> None:
        result = reference.handoff_inventory_check()
        self.assertEqual(result["inventory"]["algorithms"], 5)
        self.assertEqual(result["inventory"]["exercises"], 10)
        self.assertEqual(result["inventory"]["pending_protocols"], 10)


if __name__ == "__main__":
    unittest.main()
