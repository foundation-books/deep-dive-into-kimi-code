"""Regression tests for the standalone companion coverage contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from .chapter_coverage import (
    configured_test_timeout_seconds,
    coverage_failures,
    dependency_import_failures,
    load_manifest,
    run_chapter_tests,
)


class ChapterCoverageTests(unittest.TestCase):
    def make_fixture(self, test_source: str = "# fixture\n") -> Path:
        root = Path(tempfile.mkdtemp(prefix="kimi-companion-coverage-test."))
        chapter = root / "demo"
        chapter.mkdir()
        for filename in ("__init__.py", "README.md", "reference.py"):
            (chapter / filename).write_text("# fixture\n", encoding="utf-8")
        (chapter / "test_demo_reference.py").write_text(test_source, encoding="utf-8")
        runner = root / "_maintenance" / "chapter_test_runner.py"
        runner.parent.mkdir()
        runner.write_text(
            (Path(__file__).with_name("chapter_test_runner.py")).read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "chapters": [
                {
                    "printed_chapter": 1,
                    "slug": "demo",
                    "title": "Demo",
                    "directory": "demo",
                    "reference": "reference.py",
                    "test": "test_demo_reference.py",
                }
            ],
        }
        (root / "chapter_coverage.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (root / "README.md").write_text(
            "# Fixture\n\n"
            "| Chapter | Reference |\n"
            "| --- | --- |\n"
            "| 1. Demo | `demo/` |\n",
            encoding="utf-8",
        )
        return root

    def test_complete_fixture_passes(self) -> None:
        root = self.make_fixture()
        _, failures = coverage_failures(root)
        self.assertEqual(failures, [])

    def add_interlude(self, root: Path, number: int | None = None) -> None:
        directory = root / "bridge"
        directory.mkdir()
        for filename in ("__init__.py", "README.md", "reference.py", "test_bridge.py"):
            (directory / filename).write_text("# fixture\n", encoding="utf-8")
        manifest_path = root / "chapter_coverage.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["interludes"] = [
            {
                "printed_chapter": number,
                "slug": "bridge",
                "title": "Bridge",
                "directory": "bridge",
                "reference": "reference.py",
                "test": "test_bridge.py",
            }
        ]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with (root / "README.md").open("a", encoding="utf-8") as readme:
            readme.write("| Interlude. Bridge | `bridge/` |\n")

    def test_interlude_is_covered_without_consuming_a_number(self) -> None:
        root = self.make_fixture()
        self.add_interlude(root)
        entries, failures = coverage_failures(root)
        self.assertEqual(failures, [])
        self.assertEqual([entry.printed_chapter for entry in entries], [1, None])

    def test_interlude_can_follow_historical_chapter_companions(self) -> None:
        root = self.make_fixture()
        manifest_path = root / "chapter_coverage.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["chapters"][0]["status"] = "historical"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        readme = root / "README.md"
        readme.write_text(
            readme.read_text().replace("| 1. Demo", "| Historical 1. Demo")
        )
        self.add_interlude(root)
        entries, failures = coverage_failures(root)
        self.assertEqual(failures, [])
        self.assertEqual([entry.printed_chapter for entry in entries], [1, None])
        self.assertEqual(
            [entry.status for entry in entries], ["historical", "implemented"]
        )

    def test_interlude_cannot_claim_a_chapter_number(self) -> None:
        root = self.make_fixture()
        self.add_interlude(root, number=2)
        _, failures = coverage_failures(root)
        self.assertIn(
            "interludes[0].printed_chapter: an interlude must use null", failures
        )

    def test_interlude_requires_its_test_and_matching_readme(self) -> None:
        root = self.make_fixture()
        self.add_interlude(root)
        (root / "bridge" / "test_bridge.py").unlink()
        (root / "README.md").write_text("| 1. Demo | `demo/` |\n", encoding="utf-8")
        _, failures = coverage_failures(root)
        self.assertIn("bridge/test_bridge.py: required file is missing", failures)
        self.assertTrue(
            any("coverage table is stale" in failure for failure in failures)
        )

    def test_undeclared_reference_is_rejected(self) -> None:
        root = self.make_fixture()
        extra = root / "extra"
        extra.mkdir()
        (extra / "reference.py").write_text("# fixture\n", encoding="utf-8")
        _, failures = coverage_failures(root)
        self.assertIn("extra: reference implementation is undeclared", failures)

    def test_planned_reference_can_be_declared(self) -> None:
        root = self.make_fixture()
        planned = root / "future"
        planned.mkdir()
        for filename in (
            "__init__.py",
            "README.md",
            "reference.py",
            "test_future_reference.py",
        ):
            (planned / filename).write_text("# fixture\n", encoding="utf-8")
        manifest = json.loads((root / "chapter_coverage.json").read_text())
        manifest["chapters"].append(
            {
                "printed_chapter": 2,
                "slug": "future",
                "title": "Future",
                "directory": "future",
                "reference": "reference.py",
                "test": "test_future_reference.py",
                "status": "planned",
            }
        )
        (root / "chapter_coverage.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (root / "README.md").open("a", encoding="utf-8") as readme:
            readme.write("| 2. Future | `future/` |\n")
        _, failures = coverage_failures(root)
        self.assertEqual(failures, [])

    def test_historical_snapshot_may_retain_an_active_number(self) -> None:
        root = self.make_fixture()
        historical = root / "historical-demo"
        historical.mkdir()
        for filename in (
            "__init__.py",
            "README.md",
            "reference.py",
            "test_historical_demo_reference.py",
        ):
            (historical / filename).write_text("# fixture\n", encoding="utf-8")
        manifest_path = root / "chapter_coverage.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["chapters"].append(
            {
                "printed_chapter": 1,
                "slug": "historical-demo",
                "title": "Historical Demo",
                "directory": "historical-demo",
                "reference": "reference.py",
                "test": "test_historical_demo_reference.py",
                "status": "historical",
            }
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with (root / "README.md").open("a", encoding="utf-8") as readme:
            readme.write("| Historical 1. Historical Demo | `historical-demo/` |\n")

        _, failures = coverage_failures(root)

        self.assertEqual(failures, [])

    def test_stale_readme_coverage_is_rejected(self) -> None:
        root = self.make_fixture()
        (root / "README.md").write_text(
            "| Chapter | Reference |\n| --- | --- |\n",
            encoding="utf-8",
        )

        _, failures = coverage_failures(root)

        self.assertIn(
            "README.md coverage table is stale; its chapter number, title, and "
            "directory rows must exactly match chapter_coverage.json",
            failures,
        )

    def test_stale_readme_reports_changed_title(self) -> None:
        root = self.make_fixture()
        (root / "README.md").write_text(
            "| 1. Demo (merged into 2) | `demo/` |\n", encoding="utf-8"
        )

        _, failures = coverage_failures(root)

        self.assertIn(
            "README.md coverage row 1: expected (1, 'Demo', 'demo'), "
            "found (1, 'Demo (merged into 2)', 'demo')",
            failures,
        )

    def test_stale_readme_reports_missing_row(self) -> None:
        root = self.make_fixture()
        (root / "README.md").write_text("# No coverage rows\n", encoding="utf-8")

        _, failures = coverage_failures(root)

        self.assertIn(
            "README.md coverage row 1: expected (1, 'Demo', 'demo'), found None",
            failures,
        )

    def test_stale_readme_reports_extra_row(self) -> None:
        root = self.make_fixture()
        with (root / "README.md").open("a", encoding="utf-8") as readme:
            readme.write("| 2. Extra | `extra/` |\n")

        _, failures = coverage_failures(root)

        self.assertIn(
            "README.md coverage row 2: expected None, found (2, 'Extra', 'extra')",
            failures,
        )

    def test_missing_test_is_rejected(self) -> None:
        root = self.make_fixture()
        (root / "demo" / "test_demo_reference.py").unlink()
        _, failures = coverage_failures(root)
        self.assertIn("demo/test_demo_reference.py: required file is missing", failures)

    def test_third_party_import_is_rejected(self) -> None:
        root = self.make_fixture("import external_test_runner\n")
        self.assertEqual(
            dependency_import_failures(root),
            [
                "demo/test_demo_reference.py:1: third-party import "
                "'external_test_runner' violates the dependency-free companion contract"
            ],
        )

    def test_chapter_runner_rejects_zero_runnable_tests(self) -> None:
        root = self.make_fixture()
        entries, failures = load_manifest(root)
        self.assertEqual(failures, [])
        self.assertEqual(
            run_chapter_tests(root, entries, capture_output=True),
            ["demo/test_demo_reference.py: exited 2"],
        )

    def test_chapter_runner_executes_free_test_functions(self) -> None:
        root = self.make_fixture(
            "def test_failing_contract():\n"
            "    raise AssertionError('runner executed this test')\n"
        )
        entries, failures = load_manifest(root)
        self.assertEqual(failures, [])
        self.assertEqual(
            run_chapter_tests(root, entries, capture_output=True),
            ["demo/test_demo_reference.py: exited 1"],
        )

    def test_chapter_runner_bounds_each_test_process(self) -> None:
        root = self.make_fixture(
            "import time\n\ndef test_never_finishes_in_budget():\n    time.sleep(10)\n"
        )
        entries, failures = load_manifest(root)
        self.assertEqual(failures, [])

        self.assertEqual(
            run_chapter_tests(
                root,
                entries,
                capture_output=True,
                timeout_seconds=0.05,
            ),
            ["demo/test_demo_reference.py: timed out after 0.05 seconds"],
        )

    def test_supplementary_chapter_test_failure_is_reported(self) -> None:
        root = self.make_fixture("def test_primary():\n    assert 1 + 1 == 2\n")
        (root / "demo/test_additional.py").write_text(
            "def test_additional():\n    raise AssertionError('additional test ran')\n",
            encoding="utf-8",
        )
        entries, failures = load_manifest(root)
        self.assertEqual(failures, [])
        self.assertEqual(
            run_chapter_tests(root, entries, capture_output=True),
            ["demo/test_additional.py: exited 1"],
        )

    def test_optimized_python_cannot_silently_disable_test_assertions(self) -> None:
        root = self.make_fixture("def test_assertion():\n    assert False\n")
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                "-B",
                str(root / "_maintenance/chapter_test_runner.py"),
                str(root / "demo/test_demo_reference.py"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("require Python assertions", result.stderr)

    def test_invalid_timeout_environment_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "KIMI_COMPANION_TEST_TIMEOUT_SECONDS: chapter-test timeout",
        ):
            configured_test_timeout_seconds(
                {"KIMI_COMPANION_TEST_TIMEOUT_SECONDS": "infinite"}
            )


if __name__ == "__main__":
    unittest.main()
