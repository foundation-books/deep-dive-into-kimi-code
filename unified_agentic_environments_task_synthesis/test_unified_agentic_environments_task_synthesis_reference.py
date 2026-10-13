"""Dependency-free tests for the Chapter 24 semantic companion."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import reference
from . import studio_world
from . import task_factory
from . import task_oracle
from . import test_harness_expansion


class UnifiedAgenticEnvironmentReferenceTests(unittest.TestCase):
    def test_expansion_harness_equivalence_and_isolation(self) -> None:
        test_harness_expansion.test_equivalent_operations_and_full_task()
        test_harness_expansion.test_no_hidden_repair_permissions_or_shared_mutation()
        test_harness_expansion.test_equal_marginals_do_not_fill_joint_cells()

    def test_environment_trace(self) -> None:
        result = reference.environment_trace_check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["two_action_probability"], "27/125")
        self.assertEqual(result["vertical_centers"], [40, 100])

    def test_semantic_pairing(self) -> None:
        result = reference.semantic_pairing_check()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["transition_square_commutes"])
        self.assertTrue(result["context_loss_rejected"])
        self.assertTrue(result["paired_observations_equal"])
        self.assertTrue(result["paired_outcomes_pass"])

    def test_batch_inspection_preserves_both_shapes(self) -> None:
        expected = ("InspectBoxes", ("red", "blue"))
        flat = reference.adapt_inspection("inspect_boxes", {"ids": ["red", "blue"]})
        namespaced = reference.adapt_inspection(
            "assets.inspect_bboxes", {"shape_ids": ["red", "blue"]}
        )
        self.assertEqual(flat, expected)
        self.assertEqual(namespaced, expected)
        observation = reference.inspect_boxes(flat[1])
        self.assertEqual(
            observation, {"red": (20, 20, 40, 40), "blue": (20, 80, 40, 40)}
        )
        self.assertEqual(reference.inspect_boxes(namespaced[1]), observation)
        observation["red"] = (0, 0, 1, 1)
        self.assertEqual(reference.inspect_boxes(("red",))["red"], (20, 20, 40, 40))
        self.assertEqual(len(reference.CANONICAL_ACTIONS), 6)
        self.assertEqual(reference.CANONICAL_ACTIONS.count("InspectBoxes"), 1)
        self.assertEqual(reference.CANONICAL_ACTIONS.count("Submit"), 1)

    def test_inspection_rejects_missing_required_argument(self) -> None:
        for tool, arguments in (
            ("inspect_boxes", {}),
            ("assets.inspect_bboxes", {}),
            ("assets.inspect_bboxes", {"ids": ["red", "blue"]}),
            ("inspect_boxes", {"ids": "red"}),
        ):
            with self.subTest(tool=tool, arguments=arguments):
                with self.assertRaises(ValueError):
                    reference.adapt_inspection(tool, arguments)

    def test_harness_population(self) -> None:
        result = reference.harness_population_check()
        self.assertEqual(
            result["counts"], {"flat": 30, "namespaced": 20, "compact": 10}
        )
        self.assertEqual(
            result["masses"], {"flat": "1/2", "namespaced": "1/3", "compact": "1/6"}
        )
        self.assertTrue(result["support_complete"])
        self.assertTrue(result["joint_marginals_equal"])
        self.assertEqual(result["paired_only_empty_cells"], 2)
        self.assertEqual(result["crossed_empty_cells"], 0)

    def test_two_legal_routes_preserve_one_goal(self) -> None:
        base = studio_world.run_studio_attempt("studio_flat")
        alternate = studio_world.run_studio_attempt(
            "studio_namespaced", studio_world.ALTERNATE_TRACE
        )
        self.assertTrue(base.world.succeeded())
        self.assertTrue(alternate.world.succeeded())
        self.assertNotEqual(base.world.events, alternate.world.events)
        self.assertEqual(len(base.world.events), len(alternate.world.events))
        self.assertEqual(base.world.evidence, alternate.world.evidence)
        self.assertEqual(base.world.source_image, studio_world.STUDIO_SOURCE_BYTES)
        self.assertTrue(alternate.policy_observations[0]["asset_id"])

    def test_state_persists_within_attempt_but_not_across_attempts(self) -> None:
        after_edit = studio_world.run_studio_attempt(
            "studio_flat", studio_world.BASE_TRACE[:4]
        ).world
        after_close = studio_world.run_studio_attempt(
            "studio_flat", studio_world.BASE_TRACE[:5]
        ).world
        self.assertEqual(after_edit.report_relation, "blue below red")
        self.assertFalse(after_edit.ticket_closed)
        self.assertEqual(after_close.report_relation, "blue below red")
        self.assertTrue(after_close.ticket_closed)
        self.assertEqual(studio_world.StudioWorld().report_relation, "blue above red")

    def test_fluent_submission_and_fabricated_evidence_fail(self) -> None:
        words_only = studio_world.StudioWorld()
        words_only.execute("Submit", {"final_sentence": "blue below red"})
        self.assertFalse(words_only.succeeded())

        fabricated = studio_world.run_studio_attempt(
            "studio_flat", studio_world.BASE_TRACE[:4]
        ).world
        fabricated.execute(
            "AttachEvidenceAndClose",
            {
                "centers": {"red": (40, 40), "blue": (40, 40)},
                "asset_id": studio_world.STUDIO_ASSET_ID,
                "coordinate_convention": studio_world.COORDINATE_CONVENTION,
            },
        )
        fabricated.execute("Submit")
        self.assertFalse(fabricated.verdict()["centers_grounded"])
        self.assertFalse(fabricated.succeeded())

    def test_missing_action_and_lost_convention_break_feasibility(self) -> None:
        without_edit = studio_world.StudioTask(
            allowed_actions=frozenset(
                set(studio_world.CANONICAL_ACTIONS) - {"EditReport"}
            )
        )
        with self.assertRaisesRegex(ValueError, "unavailable action: EditReport"):
            studio_world.run_studio_attempt("studio_flat", task=without_edit)
        with self.assertRaisesRegex(ValueError, "coordinate convention"):
            studio_world.run_studio_attempt("studio_namespaced", drop_convention=True)

    def test_event_arrival_cannot_affect_earlier_choice(self) -> None:
        self.assertEqual(
            studio_world.event_policy_history(True, 4),
            studio_world.event_policy_history(False, 4),
        )
        self.assertNotEqual(
            studio_world.event_policy_history(True, 6),
            studio_world.event_policy_history(False, 6),
        )
        self.assertIn(
            studio_world.CLARIFICATION, studio_world.event_policy_history(True, 7)
        )

    def test_concept_graph(self) -> None:
        result = reference.concept_graph_check()
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["cycle_edge_rejected"])
        self.assertEqual(
            result["proposal_dispositions"], {"accepted": 3, "rejected": 1, "reused": 1}
        )

    def test_admission(self) -> None:
        result = reference.admission_check()
        self.assertEqual(result["admitted_count"], 5)
        self.assertEqual(result["yield"], "5/12")
        self.assertEqual(result["gate_count"], 8)
        self.assertTrue(result["insufficient_budget_rejected"])
        self.assertTrue(result["fabricated_evidence_rejected"])

    def test_split_contamination(self) -> None:
        result = reference.contamination_check()
        self.assertEqual(result["crossing_component_count"], 3)
        self.assertFalse(result["split_valid"])

    def test_three_task_prohibited_relation_chain(self) -> None:
        self.assertEqual(
            reference.connected_components(
                ("t1", "t2", "t3"), (("t1", "t2"), ("t2", "t3"))
            ),
            (("t1", "t2", "t3"),),
        )

    def test_persistent_aet(self) -> None:
        result = reference.persistent_aet_check()
        self.assertEqual(result["least_cost"], 6)
        self.assertEqual(
            result["event_times"], {"occurred": 3, "exposed": 5, "consumed": 6}
        )
        self.assertTrue(result["hidden_dependencies_isolated"])

    def test_handoff(self) -> None:
        result = reference.handoff_check()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["invariant_count"], 20)
        self.assertEqual(result["pending_protocol_count"], 8)

    def test_probability_rejects_invalid_factor(self) -> None:
        with self.assertRaises(ValueError):
            reference.trace_probability((Fraction(3, 2),), ())


class TaskFactoryTests(unittest.TestCase):
    def test_g1_g2_alias_distinction_and_bad_edges(self) -> None:
        graph = task_factory.ConceptGraph()
        self.assertEqual(graph.resolve("class recall"), "class-wise recall")
        self.assertNotEqual(graph.resolve("precision"), graph.resolve("class recall"))
        self.assertIn(
            "classification evaluation", graph.ancestor_query("balanced accuracy")
        )
        self.assertEqual(
            task_factory.selected_materials(
                query=graph.material_query("balanced accuracy", ("ordinary accuracy",))
            ),
            ("metric-definitions", "methodology", "counts"),
        )
        with self.assertRaisesRegex(ValueError, "local index has no support"):
            task_factory.selected_materials(query="recall")
        with self.assertRaisesRegex(ValueError, "cycle"):
            graph.add_edge("balanced accuracy", "machine learning")
        with self.assertRaisesRegex(ValueError, "specificity"):
            graph.add_edge("ordinary accuracy", "class imbalance")

    def test_m1_changed_or_missing_material_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(task_factory.ROOT / "fixtures", root / "fixtures")
            shutil.copy2(task_factory.ROOT / "sources.json", root / "sources.json")
            (root / "fixtures" / "counts.csv").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "changed source identity"):
                task_factory.construct_task("question", root)
            (root / "fixtures" / "counts.csv").unlink()
            with self.assertRaisesRegex(ValueError, "missing source"):
                task_factory.construct_task("question", root)

    def test_m2_keyword_match_cannot_substitute_for_support(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(task_factory.ROOT / "fixtures", root / "fixtures")
            manifest = json.loads((task_factory.ROOT / "sources.json").read_text())
            manifest["packet_sources"] = [
                record
                for record in manifest["packet_sources"]
                if record["id"] != "metric-definitions"
            ]
            decoy = root / "fixtures" / "decoy.md"
            decoy.write_text("balanced accuracy is mentioned, without a definition\n")
            from hashlib import sha256

            manifest["packet_sources"].append(
                {
                    "id": "decoy",
                    "file": "fixtures/decoy.md",
                    "sha256": sha256(decoy.read_bytes()).hexdigest(),
                    "supports": [],
                }
            )
            (root / "sources.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsupported material dependency"):
                task_factory.construct_task("question", root)

    def test_t1_three_families_change_outcomes_not_only_words(self) -> None:
        tasks = {
            name: task_factory.construct_task(name)
            for name in ("question", "repair", "report")
        }
        self.assertEqual(
            {task.packet_id for task in tasks.values()}, {task_factory.PACKET_ID}
        )
        self.assertEqual(len({task.request for task in tasks.values()}), 3)
        self.assertIn("buggy-program", tasks["repair"].source_ids)
        self.assertNotIn("buggy-program", tasks["question"].source_ids)
        self.assertEqual(len({task.editable for task in tasks.values()}), 3)
        self.assertEqual(len({task.deliverables for task in tasks.values()}), 3)
        self.assertTrue(
            all(task_oracle.run_reference(name)["status"] == "pass" for name in tasks)
        )

    def test_t2_fresh_world_does_not_inherit_program_or_report(self) -> None:
        repair = task_factory.construct_task("repair")
        report = task_factory.construct_task("report")
        with task_factory.fresh_workspace(repair) as first:
            (first / "analysis.py").write_text("def score(rows): return 1\n")
            (first / "output" / "scores.csv").write_text("old output\n")
        with task_factory.fresh_workspace(repair) as second:
            self.assertIn("return sum(correct", (second / "analysis.py").read_text())
            self.assertFalse((second / "output" / "scores.csv").exists())
        with task_factory.fresh_workspace(report) as third:
            self.assertFalse((third / "analysis.py").exists())
            self.assertFalse((third / "output" / "scores.csv").exists())

    def test_v1_v4_valid_repairs_and_selected_shortcuts(self) -> None:
        for name in ("reference", "alternative"):
            with self.subTest(name=name):
                self.assertEqual(
                    task_oracle.evaluate_repair_candidate(name)["status"], "pass"
                )
        for name in ("buggy", "hardcoded", "swapped", "tamper-counts"):
            with self.subTest(name=name):
                self.assertEqual(
                    task_oracle.evaluate_repair_candidate(name)["status"], "fail"
                )
        self.assertEqual(
            task_oracle.evaluate_repair_candidate("tamper-counts")["reason"],
            "protected source changed",
        )
        with self.assertRaisesRegex(ValueError, "only repository-defined"):
            task_oracle.evaluate_repair_candidate("arbitrary.py")

    def test_repair_runner_rejects_nonlist_output(self) -> None:
        with patch.object(task_oracle, "_RUNNER", "print('{}')"):
            result = task_oracle.evaluate_repair_candidate("reference")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "candidate output is not a list")

    def test_v5_report_requires_files_even_after_correct_chat_answer(self) -> None:
        task = task_factory.construct_task("report")
        with task_factory.fresh_workspace(task) as world:
            self.assertEqual(
                task_oracle.evaluate_report(world)["reason"],
                "required report files missing",
            )
            task_oracle.reference_report(world)
            self.assertEqual(task_oracle.evaluate_report(world)["status"], "pass")
            scores = world / "output" / "scores.csv"
            original = scores.read_text(encoding="utf-8")
            scores.write_text(original.replace("0.82", "invalid"), encoding="utf-8")
            self.assertEqual(task_oracle.evaluate_report(world)["status"], "fail")
            scores.write_text(original, encoding="utf-8")
            (world / "output" / "scores.csv").unlink()
            self.assertEqual(task_oracle.evaluate_report(world)["status"], "fail")

    def test_i1_i2_private_reference_and_metadata_leaks_are_rejected(self) -> None:
        task = task_factory.construct_task("question")
        task_factory.validate_reference_reads(task, {"counts.csv", "methodology.md"})
        with self.assertRaisesRegex(ValueError, "evaluator-only"):
            task_factory.validate_reference_reads(task, {"private_data.json"})
        bundle = task_factory.learner_bundle(task)
        self.assertNotIn(task_factory.EVALUATOR_SENTINEL, json.dumps(bundle))
        for field in ("tool_error", "metadata"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "leaked"),
            ):
                task_factory.assert_learner_boundary(
                    {**bundle, field: task_factory.EVALUATOR_SENTINEL}
                )

    def test_s1_s2_packet_holdout_only_for_protected_relation(self) -> None:
        self.assertFalse(
            task_factory.packet_split_valid(
                {
                    "A": ("train", "packet-1", "sklearn-balanced-1.9.1"),
                    "B": ("test", "packet-1", "sklearn-balanced-1.9.1"),
                }
            )
        )
        self.assertTrue(
            task_factory.packet_split_valid(
                {
                    "A": ("train", "packet-1", "sklearn-balanced-1.9.1"),
                    "B": ("test", "packet-2", "sklearn-balanced-1.9.1"),
                }
            )
        )

    def test_q1_q2_surface_length_and_unsupported_trend(self) -> None:
        self.assertEqual(
            task_factory.required_dependencies(),
            task_factory.required_dependencies(
                "Please write a much longer introduction."
            ),
        )
        with self.assertRaisesRegex(ValueError, "single snapshot"):
            task_factory.required_dependencies("Infer a longitudinal trend.")

    def test_arithmetic_domain_and_rank_reversal(self) -> None:
        displayed = task_oracle.displayed_scores()
        self.assertEqual(displayed["Atlas"]["balanced_accuracy"], "0.5")
        self.assertEqual(displayed["Atlas"]["ordinary_accuracy"], "0.82")
        self.assertEqual(displayed["Boreal"]["balanced_accuracy"], "0.8")
        self.assertEqual(displayed["Boreal"]["ordinary_accuracy"], "0.8")
        changed = [("zero", 90, 80), ("one", 10, 9)]
        self.assertGreater(task_oracle.independent_scores(changed)[1], Fraction(4, 5))
        for rows in (
            [("zero", 0, 0), ("one", 10, 1)],
            [("zero", 10, -1), ("one", 10, 1)],
            [("zero", 10, 11), ("one", 10, 1)],
            [("zero", 10, 1)],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                task_oracle.independent_scores(rows)


if __name__ == "__main__":
    unittest.main()
