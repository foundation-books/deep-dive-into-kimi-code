"""Self-tests for standalone release and readability checks."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from _maintenance.readability import audit_tree as audit_readability
from _maintenance.readability import main as run_readability
from _maintenance.release_hygiene import audit_tree as audit_release


class ReleaseChecksTest(unittest.TestCase):
    def test_clean_tree_passes_and_local_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            readme = root / "README.md"
            readme.write_text("A portable companion tree.\n", encoding="utf-8")

            _files, findings = audit_release(root)
            self.assertEqual(findings, [])

            local_path = "/" + "Users/example/" + "private/result.json"
            readme.write_text(f"A local run lives at {local_path}.\n", encoding="utf-8")
            _files, findings = audit_release(root)
            self.assertTrue(
                any("machine-local absolute path" in item.message for item in findings)
            )

    def test_notebook_outputs_execution_and_shell_commands_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "outputs": [{"output_type": "stream"}],
                        "source": ["!pip install fixture\n"],
                    }
                ]
            }
            (root / "fixture.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

            _files, findings = audit_release(root)

            messages = [item.message for item in findings]
            self.assertIn("notebook code cell has outputs", messages)
            self.assertIn("notebook code cell has an execution count", messages)
            self.assertIn("notebook runs a nonportable shell command", messages)

    def test_browser_and_cuda_sources_are_scanned_for_local_paths(self) -> None:
        local = "/" + "Users/example/" + "private.json"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for suffix in (".js", ".mjs", ".html", ".css", ".cu", ".cuh", ".svg"):
                (root / f"source{suffix}").write_text(local, encoding="utf-8")
            _files, findings = audit_release(root)
            self.assertEqual(len(findings), 7)
            self.assertTrue(
                all("machine-local absolute path" in item.message for item in findings)
            )

    def test_machine_names_fail_in_documentation_data_and_runtime_output(self) -> None:
        machine = "pan" + "ther"
        fixtures = {
            "README.md": f"Historical {machine.title()} wall times.\n",
            "manifest.json": json.dumps({f"experiments/{machine}/run.py": "hash"}),
            "validation.py": f'timing_note = "{machine.upper()} CPU timings"\n',
            "runner.mjs": f'console.log("{machine}_run");\n',
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for name, text in fixtures.items():
                (root / name).write_text(text, encoding="utf-8")

            _files, findings = audit_release(root)

            self.assertEqual(len(findings), len(fixtures))
            self.assertTrue(
                all("known local machine name" in item.message for item in findings)
            )

    def test_machine_names_fail_in_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            directory = root / ("pan" + "ther")
            directory.mkdir()
            (directory / "image.png").write_bytes(b"fixture")

            _files, findings = audit_release(root)

            self.assertEqual(len(findings), 1)
            self.assertIn("file path leaks", findings[0].message)

    def test_installation_and_windows_paths_fail(self) -> None:
        fixtures = (
            "/" + "Applications/Browser.app/Contents/MacOS/Browser",
            "/" + "opt/homebrew/bin/browser",
            "/" + "usr/local/bin/browser",
            "/" + "Volumes/private/browser",
            "C:" + "\\Users\\example\\browser.exe",
            "C:" + "\\Program Files\\Browser\\browser.exe",
            "C:" + "/Program Files (x86)/Browser/browser.exe",
        )
        for local_path in fixtures:
            with self.subTest(local_path=local_path):
                with tempfile.TemporaryDirectory() as raw_root:
                    root = Path(raw_root)
                    for name, text in (
                        ("path.txt", local_path),
                        ("path.json", json.dumps({"executable": local_path})),
                    ):
                        (root / name).write_text(text, encoding="utf-8")

                    _files, findings = audit_release(root)

                    self.assertEqual(len(findings), 2)
                    self.assertTrue(
                        all(
                            "machine-local absolute path" in f.message for f in findings
                        )
                    )

    def test_machine_identity_fails_in_json_and_notebook_metadata(self) -> None:
        metadata = {"host" + "name": "build-node-7"}
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "record.json").write_text(json.dumps(metadata), encoding="utf-8")
            notebook = {"cells": [], "metadata": metadata}
            (root / "record.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

            _files, findings = audit_release(root)

            self.assertEqual(len(findings), 2)
            self.assertTrue(
                all("identity metadata" in item.message for item in findings)
            )

    def test_notebook_source_fragments_are_scanned_after_joining(self) -> None:
        notebook = {
            "cells": [{"cell_type": "markdown", "source": ["pan", "ther timings"]}]
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "fixture.ipynb").write_text(json.dumps(notebook), encoding="utf-8")

            _files, findings = audit_release(root)

            self.assertEqual(len(findings), 1)
            self.assertIn("known local machine name", findings[0].message)

    def test_generic_hardware_loopback_and_configurable_paths_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "README.md").write_text(
                "One H200 GPU; Python CPU timings.\n"
                "Set K3_DASHBOARD_CHROME or add a browser to PATH.\n"
                "DevTools uses http://127.0.0.1 with a disposable profile.\n",
                encoding="utf-8",
            )

            _files, findings = audit_release(root)

            self.assertEqual(findings, [])

    def test_readme_rejects_code_as_a_python_module_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            readme = root / "README.md"
            readme.write_text(
                "Run the checks:\n\n    python3 -B -m code.example.reference_checks\n",
                encoding="utf-8",
            )

            _files, findings = audit_release(root)

            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0].line, 3)
            self.assertIn("unresolvable", findings[0].message)

    def test_readme_rejects_direct_package_test_execution(self) -> None:
        fixtures = (
            "    python3 -B code/example/test_example_reference.py\n",
            "    python3 -B \\\n      example/test_example_reference.py\n",
        )
        for command in fixtures:
            with (
                self.subTest(command=command),
                tempfile.TemporaryDirectory() as raw_root,
            ):
                root = Path(raw_root)
                readme = root / "README.md"
                readme.write_text(f"Run the checks:\n\n{command}", encoding="utf-8")

                _files, findings = audit_release(root)

                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].line, 3)
                self.assertIn("package test as a file", findings[0].message)

    def test_public_companion_definitions_require_docstrings(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            chapter = root / "chapter"
            chapter.mkdir()
            (chapter / "reference.py").write_text(
                "def documented():\n"
                '    """Return the documented fixture."""\n'
                "    return 1\n\n"
                "def undocumented():\n"
                "    return 2\n\n"
                "class UndocumentedClass:\n"
                "    pass\n\n"
                "async def undocumented_async():\n"
                "    return 3\n\n"
                "def _private_helper():\n"
                "    return 4\n",
                encoding="utf-8",
            )

            _files, findings = audit_release(root)

            self.assertEqual(len(findings), 3)
            self.assertEqual(
                {finding.line for finding in findings},
                {5, 8, 11},
            )
            self.assertEqual(
                {finding.message for finding in findings},
                {
                    "public definition undocumented needs a docstring",
                    "public definition UndocumentedClass needs a docstring",
                    "public definition undocumented_async needs a docstring",
                },
            )

    def test_readability_rejects_an_oversized_module(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "large.py").write_text("value = 1\n" * 461, encoding="utf-8")

            failures = audit_readability(root)

            self.assertTrue(any("461 lines; limit is 460" in item for item in failures))

    def test_readability_cli_scopes_to_requested_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            requested = root / "requested"
            requested.mkdir()
            (requested / "small.py").write_text("value = 1\n", encoding="utf-8")
            sibling = root / "sibling"
            sibling.mkdir()
            (sibling / "large.py").write_text(
                "value = 1\n" * 461,
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = run_readability([str(requested)])

            self.assertEqual(result, 0, stderr.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("passed (1 file(s))", stdout.getvalue())

    def test_readability_requires_ten_percent_file_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "near.py").write_text("value = 1\n" * 414, encoding="utf-8")

            failures = audit_readability(root)

            self.assertTrue(
                any(
                    "414 lines and is near its 460-line limit" in item
                    for item in failures
                )
            )

    def test_readability_requires_ten_percent_function_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            body = "\n".join(f"    value_{index} = {index}" for index in range(71))
            (root / "near.py").write_text(
                f"def near_limit():\n{body}\n",
                encoding="utf-8",
            )

            failures = audit_readability(root)

            self.assertTrue(
                any(
                    "72 lines and is near its 80-line limit" in item
                    for item in failures
                )
            )

    def test_readability_counts_comprehension_branches_and_nested_functions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            comprehensions = "\n".join(
                f"    values_{index} = [item for item in range(3) if item]"
                for index in range(16)
            )
            nested = "\n".join(
                f"    def nested_{index}():\n        return {index}"
                for index in range(5)
            )
            (root / "complex.py").write_text(
                f"def branchy():\n{comprehensions}\n\n\ndef nested_owner():\n{nested}\n",
                encoding="utf-8",
            )

            failures = audit_readability(root)

            self.assertTrue(
                any("16 branches; limit is 15" in item for item in failures)
            )
            self.assertTrue(
                any("5 nested functions; limit is 4" in item for item in failures)
            )

    def test_readability_requires_function_and_branch_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            body = "\n".join(f"    if value == {index}: pass" for index in range(15))
            (root / "near.py").write_text(
                f"def branchy(value):\n{body}\n",
                encoding="utf-8",
            )

            failures = audit_readability(root)

            self.assertTrue(
                any("near its 15-branch limit" in item for item in failures)
            )

    def test_readability_rejects_duplicate_production_functions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            body = "\n".join(f"    value_{index} = {index}" for index in range(7))
            (root / "one.py").write_text(f"def one():\n{body}\n", encoding="utf-8")
            (root / "two.py").write_text(f"def two():\n{body}\n", encoding="utf-8")

            failures = audit_readability(root)

            self.assertTrue(
                any("exact duplicate production function" in item for item in failures)
            )


@unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
class BrowserRunnerTest(unittest.TestCase):
    def run_node_checks(self, source: str) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "kimi_code_test_grounded_software_engineering"
            / "web_visual_artifact_learning/runner/browser_checks.mjs"
        )
        completed = subprocess.run(
            ["node", "--input-type=module", "-", str(runner)],
            input=source,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_browser_discovery_override_missing_and_nonexecutable_files(self) -> None:
        self.run_node_checks(r"""
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const {resolveChrome} = await import(pathToFileURL(process.argv[2]).href);
const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'browser-discovery-'));
try {
  const first = path.join(directory, 'first');
  const second = path.join(directory, 'second');
  await fs.mkdir(first); await fs.mkdir(second);
  const chrome = path.join(first, 'google-chrome');
  const chromium = path.join(second, 'chromium.exe');
  const custom = path.join(directory, 'Browser With Spaces');
  for (const file of [chrome, chromium, custom]) {
    await fs.writeFile(file, 'fixture', {mode: 0o755});
  }
  const PATH = [first, second].join(path.delimiter);
  assert.equal(await resolveChrome({PATH}), chrome);
  assert.equal(await resolveChrome({PATH: second}), chromium);
  assert.equal(await resolveChrome({Path: second}), chromium);
  assert.equal(await resolveChrome({PATH, K3_DASHBOARD_CHROME: custom}), custom);
  assert.equal(await resolveChrome({PATH, K3_DASHBOARD_CHROME: 'chromium.exe'}), chromium);
  await assert.rejects(resolveChrome({PATH, K3_DASHBOARD_CHROME: 'missing'}), /K3_DASHBOARD_CHROME/);
  await assert.rejects(resolveChrome({PATH, K3_DASHBOARD_CHROME: first}), /executable file/);
  await assert.rejects(resolveChrome({PATH: directory}), /No Chrome or Chromium/);
  await assert.rejects(resolveChrome({}), /No Chrome or Chromium/);
  if (process.platform !== 'win32') {
    await fs.chmod(chrome, 0o644);
    assert.equal(await resolveChrome({PATH}), chromium);
    await assert.rejects(resolveChrome({PATH: first}), /No Chrome or Chromium/);
  }
} finally {
  await fs.rm(directory, {recursive: true, force: true});
}
""")

    def test_devtools_browser_build_guard(self) -> None:
        self.run_node_checks("""
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {checkChromeVersion} = await import(pathToFileURL(process.argv[2]).href);
for (const brand of ['Chrome', 'HeadlessChrome', 'Chromium']) {
  const product = `${brand}/153.0.8010.53`;
  assert.equal(checkChromeVersion(product), product);
}
for (const product of ['Chrome/152.0.0.0', 'Chrome/153.0.8010.54', 'Firefox/153.0.8010.53', undefined]) {
  assert.throws(() => checkChromeVersion(product), /Expected Chrome or Chromium build/);
}
""")


if __name__ == "__main__":
    unittest.main()
