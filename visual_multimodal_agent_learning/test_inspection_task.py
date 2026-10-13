"""No-download tests of the inspection task's information boundary."""

from __future__ import annotations

from dataclasses import replace
import unittest

from .inspection_task import source_task, visual_positions


class InspectionTaskTests(unittest.TestCase):
    def test_independent_answer_and_useless_crop(self) -> None:
        task = source_task(44)
        self.assertEqual(task.panels, ("B", "A", "D", "C"))
        self.assertEqual(task.answer, 7)
        self.assertEqual(task.values[0], 7)
        self.assertEqual(task.crop(0), (0, 0, 336, 224))
        with self.assertRaises(ValueError):
            task.crop(4)

    def test_answer_does_not_enter_model_facing_text(self) -> None:
        task = source_task(44)
        values = list(task.values)
        values[task.panels.index(task.query)] = 2
        other = replace(task, values=tuple(values))
        self.assertNotEqual(task.answer, other.answer)
        self.assertEqual(task.allowed_text(), other.allowed_text())
        self.assertNotIn("7", task.allowed_text())
        self.assertNotIn("target", task.allowed_text())

    def test_split_families_and_media_order_accounting(self) -> None:
        training = {p for i in range(44, 68) for p in source_task(i).panels}
        development = {p for i in range(68, 72) for p in source_task(i).panels}
        final = {p for i in range(72, 88) for p in source_task(i).panels}
        self.assertTrue(training.isdisjoint(development | final))
        self.assertTrue(development.isdisjoint(final))
        self.assertEqual(visual_positions(((1, 16, 24),)), 96)
        self.assertEqual(visual_positions(((1, 16, 24), (1, 16, 24))), 192)
        with self.assertRaises(ValueError):
            visual_positions(((1, 15, 24),))
