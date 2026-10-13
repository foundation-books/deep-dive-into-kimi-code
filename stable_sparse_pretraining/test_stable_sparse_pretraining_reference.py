"""Dependency-free tests for the Chapter 14 semantic companion."""

from __future__ import annotations

from fractions import Fraction
import unittest

from . import reference


class StableSparsePretrainingReferenceTests(unittest.TestCase):
    def test_per_head_transform_is_block_local_and_not_global(self) -> None:
        fixture = reference.per_head_fixture()
        self.assertEqual(fixture["head_0_blockwise_perturbation"], 0.0)
        self.assertAlmostEqual(
            fixture["head_0_global_perturbation"], 0.05598584050649515
        )
        self.assertAlmostEqual(
            fixture["blockwise_global_difference"], 0.1805467646988024
        )
        self.assertEqual(fixture["rank_deficient_shape"], (2, 2))
        self.assertTrue(fixture["rank_deficient_finite"])

    def test_qb_selects_with_bias_but_weights_with_raw_scores(self) -> None:
        fixture = reference.qb_fixture()
        self.assertEqual(
            fixture["raw_scores"],
            (Fraction(4, 5), Fraction(17, 25), Fraction(16, 25), Fraction(2, 25)),
        )
        self.assertEqual(fixture["selected"], (2, 1))
        self.assertEqual(fixture["weights"], (Fraction(16, 33), Fraction(17, 33)))
        self.assertEqual(
            fixture["current_bias_after_observation"], fixture["current_bias"]
        )
        self.assertNotEqual(fixture["candidate_next_bias"], fixture["current_bias"])
        with self.assertRaisesRegex(ValueError, "strictly between"):
            reference.qb_select(
                (Fraction(1), Fraction(1, 2)),
                (Fraction(0), Fraction(0)),
            )

    def test_moonep_plan_changes_placement_not_semantics(self) -> None:
        home = reference.running_routes()
        balanced = reference.balance_execution(home)
        self.assertEqual(reference.expert_counts(home), (8, 4, 2, 2))
        self.assertEqual(reference.rank_loads(home), (12, 4))
        self.assertEqual(reference.rank_loads(balanced), (8, 8))
        self.assertEqual(
            reference.route_identity(home), reference.route_identity(balanced)
        )
        self.assertEqual(
            reference.combined_outputs(home), reference.combined_outputs(balanced)
        )
        self.assertEqual({record.slot for record in home}, {1, 2})
        self.assertTrue(
            all(
                record.origin == (record.dp_replica, record.source_rank, record.token)
                for record in home
            )
        )
        other_replica = reference.running_routes(dp_replica=1)
        self.assertTrue(
            set(reference.route_identity(home)).isdisjoint(
                reference.route_identity(other_replica)
            )
        )

    def test_route_faults_and_extreme_loads_are_covered(self) -> None:
        fixture = reference.placement_fault_fixture()
        self.assertEqual(fixture["skew_free_loads"], (8, 8))
        self.assertEqual(fixture["maximum_skew_home_loads"], (16, 0))
        self.assertEqual(fixture["maximum_skew_balanced_loads"], (8, 8))
        self.assertTrue(all(fixture["detected_failures"].values()))

    def test_repatriation_then_dp_consensus_conserves_gradient(self) -> None:
        replica_records = tuple(
            reference.balance_execution(reference.running_routes(dp_replica=replica))
            for replica in range(2)
        )
        replicas = tuple(
            reference.repatriate_to_home(reference.execution_partials(records))
            for records in replica_records
        )
        observed = reference.dp_consensus(replicas)
        expected = reference.direct_semantic_gradients(
            tuple(record for records in replica_records for record in records)
        )
        self.assertEqual(observed, expected)
        partials = reference.execution_partials(replica_records[0])
        self.assertIn((1, 0), partials)

    def test_sharded_owner_update_equals_unsharded_update(self) -> None:
        fixture = reference.owner_equivalence_fixture()
        self.assertEqual(fixture["sharded"], fixture["unsharded"])
        self.assertEqual(fixture["unsharded"].step, 1)
        self.assertTrue(all(reference.owner_fault_fixture().values()))

    def test_restart_replays_and_rejects_a_torn_manifest(self) -> None:
        fixture = reference.restart_replay_fixture()
        self.assertEqual(fixture["durable_step"], 1)
        self.assertEqual(fixture["volatile_step"], 2)
        self.assertEqual(fixture["loaded_step"], 1)
        self.assertTrue(fixture["replay_equal"])
        self.assertTrue(fixture["torn_manifest_rejected"])
        self.assertEqual(
            fixture["crash_window_loaded_steps"],
            {
                "before_new_shards": 1,
                "after_partial_shards": 1,
                "after_all_shards_before_manifest": 1,
                "after_valid_manifest": 2,
            },
        )
        self.assertTrue(fixture["compatible_reshard_equal"])


if __name__ == "__main__":
    unittest.main()
