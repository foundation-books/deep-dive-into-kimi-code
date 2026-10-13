"""Architecture regressions for split companion reference modules."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SPLIT_CHAPTERS = (
    "capability_consolidation_deployment_aware_post_training",
    "long_cot_rl_policy_optimization",
    "multi_domain_reasoning_effort_rl",
    "sampling_curricula_partial_rollouts",
)


class ReferenceArchitectureTest(unittest.TestCase):
    def test_check_modules_depend_on_core_not_facade(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        for chapter in SPLIT_CHAPTERS:
            chapter_root = code_root / chapter
            checks = ast.parse(
                (chapter_root / "reference_checks.py").read_text(encoding="utf-8")
            )
            imported_modules = {
                node.module
                for node in ast.walk(checks)
                if isinstance(node, ast.ImportFrom)
            }
            self.assertIn("reference_core", imported_modules, chapter)
            self.assertNotIn("reference", imported_modules, chapter)

            core = (chapter_root / "reference_core.py").read_text(encoding="utf-8")
            self.assertNotIn("reference_checks", core, chapter)

            for support_path in chapter_root.glob("reference_*.py"):
                if support_path.name == "reference_core.py":
                    continue
                support = ast.parse(support_path.read_text(encoding="utf-8"))
                imported_modules = {
                    node.module
                    for node in ast.walk(support)
                    if isinstance(node, ast.ImportFrom)
                }
                self.assertNotIn("reference", imported_modules, support_path.name)

    def test_split_facades_have_explicit_public_exports(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        for chapter in SPLIT_CHAPTERS:
            facade_path = code_root / chapter / "reference.py"
            facade = ast.parse(facade_path.read_text(encoding="utf-8"))
            self.assertFalse(
                any(
                    isinstance(node, ast.ImportFrom)
                    and any(alias.name == "*" for alias in node.names)
                    for node in ast.walk(facade)
                ),
                chapter,
            )
            exports = [
                node
                for node in facade.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                )
            ]
            self.assertEqual(len(exports), 1, chapter)
            assignment = exports[0]
            value = assignment.value
            self.assertIsNotNone(value, chapter)
            assert value is not None
            exported_names = ast.literal_eval(value)
            imported_names = {
                alias.asname or alias.name
                for node in facade.body
                if isinstance(node, ast.ImportFrom) and node.module != "__future__"
                for alias in node.names
                if not (alias.asname or alias.name).startswith("_")
            }
            self.assertTrue(
                all(isinstance(name, str) for name in exported_names), chapter
            )
            self.assertEqual(len(exported_names), len(imported_names), chapter)
            self.assertEqual(set(exported_names), imported_names, chapter)


if __name__ == "__main__":
    unittest.main()
