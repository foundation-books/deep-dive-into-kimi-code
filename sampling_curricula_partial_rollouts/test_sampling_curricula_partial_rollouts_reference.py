"""Tests for the Chapter 20 dependency-free semantic companion."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import unittest

from . import reference


class SamplingCurriculaPartialRolloutsReferenceTests(unittest.TestCase):
    def test_sampling_change_of_measure_and_support(self) -> None:
        q = reference.normalize_priority(
            (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10))
        )
        self.assertEqual(q, (Fraction(1, 15), Fraction(1, 3), Fraction(3, 5)))
        p = (Fraction(1, 3),) * 3
        self.assertEqual(
            reference.importance_weights(p, q),
            (Fraction(5), Fraction(1), Fraction(5, 9)),
        )
        self.assertEqual(reference.expectation(q, (1, 0, -1)), Fraction(-8, 15))
        self.assertEqual(reference.corrected_expectation(p, q, (1, 0, -1)), 0)
        with self.assertRaises(ValueError):
            reference.importance_weights((1, 1), (0, 1))
        with self.assertRaises(ValueError):
            reference.normalize_priority((0, 0))
        self.assertEqual(
            reference.normalize_priority((0, 0), fallback=(1, 1)),
            (Fraction(1, 2), Fraction(1, 2)),
        )

    def test_difficulty_counts_and_smoothing(self) -> None:
        raw = tuple(
            reference.raw_difficulty(s, f)
            for s, f in zip(
                reference.STUDIO_SUCCESSES,
                reference.STUDIO_FAILURES,
                strict=True,
            )
        )
        self.assertEqual(raw, (Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)))
        smooth = tuple(
            reference.beta_smoothed_difficulty(s, f, 1, 1)
            for s, f in zip(
                reference.STUDIO_SUCCESSES,
                reference.STUDIO_FAILURES,
                strict=True,
            )
        )
        self.assertEqual(smooth, (Fraction(1, 6), Fraction(1, 2), Fraction(5, 6)))
        self.assertEqual(reference.beta_smoothed_difficulty(0, 0, 2, 3), Fraction(3, 5))
        with self.assertRaises(ValueError):
            reference.raw_difficulty(0, 0)

    def test_lifecycle_and_checkpoint(self) -> None:
        for source, destination in reference.LEGAL_TRANSITIONS:
            self.assertTrue(reference.validate_transition(source, destination))
        with self.assertRaises(ValueError):
            reference.validate_transition("paused", "terminal")
        checkpoint = reference.PauseCheckpoint(
            "task",
            "group",
            4,
            1,
            120,
            "sampler",
            ("a",),
            "environment",
            "fresh-suffix",
            "tokenizer",
            "schema",
            "content",
        )
        self.assertTrue(reference.validate_checkpoint(checkpoint))
        with self.assertRaises(ValueError):
            reference.validate_checkpoint(replace(checkpoint, environment_state=""))

    def test_reuse_readiness_fairness_and_segments(self) -> None:
        self.assertTrue(reference.ReuseVector(1, 0, 0, 0).validate())
        with self.assertRaises(ValueError):
            reference.ReuseVector(2, 0, 0, 0).validate()
        ready = reference.group_status(("terminal",) * 4, (1, 1, 1, 1))
        self.assertEqual(ready, {"settled": True, "ready": True})
        retired = reference.group_status(
            ("terminal", "terminal", "terminal", "retired"), (1, 1, 1, 0)
        )
        self.assertEqual(retired, {"settled": True, "ready": False})
        with self.assertRaises(ValueError):
            reference.group_status(("terminal",), (2,))
        self.assertTrue(reference.validate_weak_fair_schedule(("b", "a"), {"a", "b"}))
        with self.assertRaises(ValueError):
            reference.validate_weak_fair_schedule(("a",), {"a", "b"})
        self.assertEqual(
            reference.validate_segments(
                20,
                (reference.Segment(1, 10, 120), reference.Segment(11, 20, 121)),
                123,
            ),
            {"lags": (3, 2), "maximum_lag": 3},
        )
        with self.assertRaises(ValueError):
            reference.validate_segments(
                20,
                (reference.Segment(1, 9, 120), reference.Segment(11, 20, 121)),
                123,
            )

    def test_complete_trace_and_six_protocols(self) -> None:
        trace = reference.studio_trace()
        self.assertEqual(trace["acquisition"], ["1/15", "1/3", "3/5"])
        self.assertEqual(trace["weights"], ["5", "1", "5/9"])
        self.assertEqual(trace["total_tokens"], 48)
        self.assertEqual(trace["slot_4_lags"], [3, 2])
        self.assertTrue(trace["ready"])
        self.assertEqual(len(trace["content_sha256"]), 64)
        checks = reference.all_checks()
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(record["status"] == "pass" for record in checks.values()))


if __name__ == "__main__":
    unittest.main()
