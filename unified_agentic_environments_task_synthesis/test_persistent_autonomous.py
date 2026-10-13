"""Positive and negative checks for the book-created Chapter 24 cases."""

from __future__ import annotations

from copy import deepcopy
from math import isclose
from typing import cast
import unittest

from .learning_fixture import (
    annotated_record,
    finite_update,
    probability,
    sample_gradient,
)
from .onboarding import (
    DATE_AVAILABLE,
    DAY,
    END,
    GRANT_AVAILABLE,
    MANAGER_AVAILABLE,
    OnboardingWorld,
    scripted_attempt,
)
from .repair_oracle import (
    OracleClient,
    OracleSession,
    TicketService,
    evaluate_candidate,
    investigation_record,
)
from .render_book_assets import assets, clock_time


class PersistentWorkflowTests(unittest.TestCase):
    def test_two_valid_routes_and_counterexamples(self) -> None:
        for name in ("incremental", "collect"):
            with self.subTest(name=name):
                result = scripted_attempt(name).evaluate()
                self.assertEqual(result["status"], "pass")
                self.assertEqual(
                    result["deadlines"], {"date": True, "manager": True, "grant": True}
                )
        for name, failed in (
            ("static", "orientation"),
            ("calendar_only", "task"),
            ("restart", "orientation"),
            ("late", "timely"),
            ("collateral", "protected"),
            ("wrong_notice", "notice"),
        ):
            with self.subTest(name=name):
                result = scripted_attempt(name).evaluate()
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["checks"][failed])

    def test_occurrence_availability_and_history_are_distinct(self) -> None:
        world = OnboardingWorld()
        world.step("wait", until=DAY + 8 * 60)
        self.assertEqual(world.current_start_day, 6)
        self.assertNotIn("date", world.available)
        self.assertEqual(world.step("read_messages")["messages"], [])
        self.assertNotIn("date", world.consumed)
        world.step("wait", until=DATE_AVAILABLE)
        self.assertIn("date", world.available)
        self.assertNotIn("date", world.consumed)
        self.assertEqual(world.step("read_messages")["messages"][0]["update"], "date")
        self.assertEqual(world.consumed["date"], DATE_AVAILABLE)

    def test_earlier_valid_action_is_not_graded_with_future_update(self) -> None:
        world = OnboardingWorld()
        world.step("create_calendar", day=4, manager="Mira")
        world.step("create_task", day=4, manager="Mira")
        world.step("request_provision", day=4, manager="Mira")
        world.step("wait", until=DAY + 8 * 60)
        self.assertTrue(world.initial_valid)
        self.assertEqual(world.calendar[0]["day"], 4)
        self.assertEqual(world.current_start_day, 6)

    def test_delayed_effect_uses_old_arguments_and_requires_new_repair(self) -> None:
        world = OnboardingWorld()
        world.step("request_provision", day=4, manager="Mira")
        world.step("wait", until=DAY + 11 * 60)
        self.assertEqual(world.grants[0]["day"], 4)
        self.assertEqual(world.step("inspect_task")["requests"], ["pending"])
        world.step("wait", until=GRANT_AVAILABLE)
        self.assertEqual(world.step("inspect_task")["requests"], ["complete"])
        self.assertEqual(world.step("inspect_access")["grants"][0]["day"], 4)

    def test_end_state_alone_does_not_erase_missed_deadline(self) -> None:
        timely = scripted_attempt("collect").evaluate()
        late = scripted_attempt("late").evaluate()
        for key in ("orientation", "task", "grant", "notice"):
            self.assertEqual(timely["checks"][key], late["checks"][key])
        self.assertTrue(timely["checks"]["timely"])
        self.assertFalse(late["checks"]["timely"])

    def test_early_finish_and_restore_preserve_obligations_and_budget(self) -> None:
        world = OnboardingWorld()
        world.step("request_provision", day=4, manager="Mira")
        self.assertEqual(
            world.step("finalize")["status"], "early_finalization_rejected"
        )
        self.assertEqual(world.evaluate(), {"status": "unfinished"})
        world.step("wait", until=GRANT_AVAILABLE)
        saved = world.snapshot()
        world.step("inspect_access")
        spent = world.actions
        world.restore(saved)
        self.assertEqual(len(world.grants), 1)
        self.assertEqual(world.actions, spent)
        world.step("wait", until=END)
        self.assertEqual(len(world.grants), 1)

    def test_updates_have_feasible_response_windows(self) -> None:
        world = scripted_attempt("incremental")
        self.assertLess(world.consumed["date"], DATE_AVAILABLE + 4 * 60)
        self.assertLess(world.consumed["manager"], MANAGER_AVAILABLE + 4 * 60)
        self.assertTrue(all(world.deadlines.values()))

    def test_printed_rows_follow_checked_states(self) -> None:
        rendered = assets()
        self.assertIn(
            "D2 11:05 (old result available)",
            rendered["onboarding_obligations_rows.tex"],
        )
        self.assertIn(
            "request D4 (complete); grant D4",
            rendered["onboarding_obligations_rows.tex"],
        )
        self.assertIn("a15 11:15", rendered["onboarding_timeline.tex"])
        self.assertEqual(clock_time(END), "D3 16:00")


class AutonomousInvestigationTests(unittest.TestCase):
    def test_probes_and_budget_counts(self) -> None:
        record = investigation_record()
        self.assertEqual(record["oracle_operations"], 7)
        self.assertEqual(record["public_submissions"], 2)
        self.assertEqual(record["final_submissions"], 1)
        self.assertTrue(record["duplicate_handles_equal"])
        self.assertEqual(record["unassigned_status"], "assignment_required")
        self.assertEqual(record["after_duplicate_diagnostic"]["status"], "public_fail")
        self.assertEqual(record["final_public"]["status"], "public_pass")
        self.assertTrue(record["final_result"]["passed"])
        self.assertEqual(record["trace"][-1]["cumulative"], 7)

    def test_error_paths_do_not_mutate_tickets(self) -> None:
        service = TicketService()
        handle = service.create("R", "camera")["ticket"]
        before = deepcopy(service.tickets)
        self.assertEqual(service.create("R", "lens"), {"status": "reference_conflict"})
        self.assertEqual(service.close(str(handle)), {"status": "assignment_required"})
        self.assertEqual(service.close("missing"), {"status": "unknown_ticket"})
        self.assertEqual(
            service.assign(str(handle), cast(str, 7)), {"status": "invalid_argument"}
        )
        self.assertEqual(
            service.assign(cast(str, []), "repair"), {"status": "unknown_ticket"}
        )
        self.assertEqual(service.close(cast(str, [])), {"status": "unknown_ticket"})
        self.assertEqual(service.tickets, before)
        self.assertFalse(service.tickets[str(handle)].closed)

    def test_valid_alternative_handles_and_wrong_candidate(self) -> None:
        self.assertTrue(evaluate_candidate(lambda: TicketService(prefix="C"))["passed"])
        wrong = evaluate_candidate(
            lambda: TicketService(reuse_identical=False, require_assignment=False)
        )
        self.assertFalse(wrong["passed"])
        self.assertFalse(wrong["cases"]["duplicate"])
        self.assertFalse(wrong["cases"]["unassigned"])

    def test_public_final_and_oracle_are_separate_limited_channels(self) -> None:
        session = OracleSession()
        client = OracleClient(session)
        self.assertFalse(hasattr(client, "author_result"))
        for index in range(8):
            self.assertEqual(
                client.query("create", f"R{index}", "camera")["status"], "created"
            )
        self.assertEqual(
            client.query("create", "extra", "camera")["status"],
            "oracle_budget_exhausted",
        )
        self.assertEqual(session.oracle_operations, 8)
        for _ in range(2):
            client.public_diagnostic(TicketService)
        self.assertEqual(
            client.public_diagnostic(TicketService)["status"],
            "public_budget_exhausted",
        )
        self.assertEqual(session.author_result(), {"status": "unfinished"})
        self.assertEqual(client.final_submit(TicketService), {"status": "submitted"})
        self.assertEqual(
            client.final_submit(TicketService)["status"], "final_budget_exhausted"
        )
        self.assertEqual(client.query("close", "T1")["status"], "attempt_finalized")

    def test_printed_probe_outputs_follow_checked_oracle(self) -> None:
        rendered = assets()
        self.assertIn("7/8", rendered["repair_investigation_rows.tex"])
        self.assertIn("\\texttt{existing} observed", rendered["repair_query_tree.tex"])
        self.assertIn(
            "\\texttt{assignment\\_required} observed",
            rendered["repair_query_tree.tex"],
        )


class FiniteLearningTests(unittest.TestCase):
    def test_sample_and_expected_gradients_and_next_probability(self) -> None:
        result = finite_update()
        self.assertTrue(isclose(result["successful_sample_gradient"], 0.8))
        self.assertEqual(result["failure_sample_gradient"], 0.0)
        self.assertTrue(isclose(result["expected_gradient"], 0.16))
        self.assertTrue(isclose(result["after_logit"] - result["before_logit"], 0.2))
        self.assertTrue(isclose(result["after_probability"], 0.233922341319356))
        epsilon = 1e-6
        logit = result["before_logit"]
        numerical = (probability(logit + epsilon) - probability(logit - epsilon)) / (
            2 * epsilon
        )
        self.assertTrue(isclose(numerical, result["expected_gradient"], rel_tol=1e-8))
        self.assertTrue(isclose(sample_gradient(1, 1, logit), 0.8))

    def test_observation_is_context_without_action_loss(self) -> None:
        record = annotated_record()
        self.assertEqual(
            [row["assistant_action_mask"] for row in record], [0, 1, 0, 0, 1, 0]
        )
        self.assertEqual(record[3]["kind"], "tool_observation")


if __name__ == "__main__":
    unittest.main()
