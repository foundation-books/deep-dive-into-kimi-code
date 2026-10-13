"""Behavioral regressions for the closed, standalone public-release workflow."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import release
from .release_inventory import (
    INVENTORY_NAME,
    ReleaseError,
    load_inventory,
    validate_inventory,
)


class StandaloneReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory(prefix="kimi-release-test.")
        self.addCleanup(self.scratch.cleanup)
        self.parent = Path(self.scratch.name)
        self.root = self.parent / "code"
        (self.root / "pkg").mkdir(parents=True)
        (self.root / "assets").mkdir()
        (self.root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
        (self.root / "README.md").write_text(
            '# Companion\n\n[example](pkg/README.md)\n<img src="assets/cover.png">\n',
            encoding="utf-8",
        )
        (self.root / "pkg/README.md").write_text(
            "[reference](reference.py)\n", encoding="utf-8"
        )
        (self.root / "pkg/reference.py").write_text(
            '"""Finite teaching fixture."""\nVALUE = 3\n', encoding="utf-8"
        )
        (self.root / "assets/cover.png").write_bytes(b"binary teaching image\x00")
        (self.root / "pkg/notes.md").write_text(
            "Author investigation.\n", encoding="utf-8"
        )
        self.included = [
            INVENTORY_NAME,
            "LICENSE",
            "README.md",
            "assets/cover.png",
            "pkg/README.md",
            "pkg/reference.py",
        ]
        self.policy = {
            "schema_version": 1,
            "included": self.included,
            "excluded": {"pkg/notes.md": "Extended author investigation."},
        }
        self.write_policy()

    def write_policy(self) -> None:
        (self.root / INVENTORY_NAME).write_text(
            json.dumps(self.policy) + "\n", encoding="utf-8"
        )

    def test_export_closes_inventory_copies_bytes_modes_and_verifies(self) -> None:
        reference = self.root / "pkg/reference.py"
        reference.chmod(0o755)
        outside = self.parent / "book"
        outside.mkdir()
        (outside / "private.tex").write_text("Private manuscript.\n", encoding="utf-8")
        output = self.parent / "candidate"
        with patch.object(release, "_source_commit", return_value=None):
            manifest = release.build_candidate(self.root, output)

        self.assertEqual(set(manifest["files"]), set(self.included))
        self.assertIsNone(manifest["source_commit"])
        self.assertFalse((output / "pkg/notes.md").exists())
        self.assertFalse((output / "book").exists())
        self.assertEqual((output / "pkg/reference.py").stat().st_mode & 0o777, 0o755)
        self.assertEqual(
            (output / "assets/cover.png").read_bytes(), b"binary teaching image\x00"
        )
        for name in self.included:
            self.assertEqual(
                manifest["files"][name],
                hashlib.sha256((output / name).read_bytes()).hexdigest(),
            )
        self.assertEqual(release.verify_release(output), manifest)
        self.assertEqual(validate_inventory(output).included, frozenset(self.included))

    def test_unknown_or_missing_file_blocks_export_before_destination_creation(
        self,
    ) -> None:
        output = self.parent / "candidate"
        unknown = self.root / "pkg/private.json"
        unknown.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseError, "unclassified file"):
            release.build_candidate(self.root, output)
        self.assertFalse(output.exists())
        unknown.unlink()
        (self.root / "assets/cover.png").unlink()
        with self.assertRaisesRegex(ReleaseError, "missing included file"):
            release.build_candidate(self.root, output)
        self.assertFalse(output.exists())

    def test_local_environment_is_ignored_and_never_copied(self) -> None:
        cache = self.root / ".venv"
        cache.mkdir()
        (cache / "private.txt").write_text("Local environment.\n", encoding="utf-8")
        output = self.parent / "candidate"
        release.build_candidate(self.root, output)
        self.assertFalse((output / ".venv").exists())

    def test_inventory_rejects_unsafe_duplicate_and_overlapping_paths(self) -> None:
        unsafe = (
            "../outside.py",
            "/absolute.py",
            "a/../b.py",
            "a\\b.py",
            ".git/config",
        )
        for name in unsafe:
            with self.subTest(name=name):
                self.policy["included"] = [*self.included, name]
                self.write_policy()
                with self.assertRaisesRegex(ReleaseError, "portable relative"):
                    load_inventory(self.root)
        self.policy["included"] = [*self.included, self.included[0]]
        self.write_policy()
        with self.assertRaisesRegex(ReleaseError, "duplicate included"):
            load_inventory(self.root)
        self.policy["included"] = [*self.included, "pkg/notes.md"]
        self.write_policy()
        with self.assertRaisesRegex(ReleaseError, "overlap"):
            load_inventory(self.root)

    def test_markdown_links_and_html_covers_must_be_exported(self) -> None:
        for text, message in (
            ("[author](notes.md)\n", "not exported"),
            ("[outside](../../book/private.tex)\n", "escapes code tree"),
            ('<img src="../assets/missing.png">\n', "not exported"),
        ):
            with self.subTest(text=text):
                (self.root / "pkg/README.md").write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ReleaseError, message):
                    validate_inventory(self.root)

    def test_symlink_file_or_directory_cannot_escape_inventory(self) -> None:
        reference = self.root / "pkg/reference.py"
        reference.unlink()
        reference.symlink_to(self.parent / "outside.py")
        with self.assertRaisesRegex(ReleaseError, "symlink is not portable"):
            validate_inventory(self.root)
        reference.unlink()
        reference.write_text('"""Fixture."""\n', encoding="utf-8")
        (self.root / "alias").symlink_to(self.root / "pkg", target_is_directory=True)
        with self.assertRaisesRegex(ReleaseError, "symlink is not portable"):
            validate_inventory(self.root)

    def test_existing_and_nested_destinations_are_refused(self) -> None:
        output = self.parent / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("keep\n", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseError, "already exists"):
            release.build_candidate(self.root, output)
        self.assertEqual((output / "keep.txt").read_text(), "keep\n")
        with self.assertRaisesRegex(ReleaseError, "outside the companion"):
            release.build_candidate(self.root, self.root / "candidate")
        self.assertFalse((self.root / "candidate").exists())

    def test_manifest_detects_changed_bytes_and_private_file_additions(self) -> None:
        output = self.parent / "candidate"
        release.build_candidate(self.root, output)
        reference = output / "pkg/reference.py"
        original = reference.read_bytes()
        reference.write_bytes(original + b"VALUE = 9\n")
        with self.assertRaisesRegex(ReleaseError, "release hash mismatch"):
            release.verify_release(output)
        reference.write_bytes(original)
        (output / "pkg/notes.md").write_text("Private notes.\n", encoding="utf-8")
        with self.assertRaisesRegex(ReleaseError, "excluded file present"):
            release.verify_release(output)

    def test_manifest_rejects_an_incomplete_file_set(self) -> None:
        output = self.parent / "candidate"
        release.build_candidate(self.root, output)
        packet = output / "release-manifest.json"
        manifest = json.loads(packet.read_text())
        del manifest["files"]["LICENSE"]
        packet.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ReleaseError, "file set differs"):
            release.verify_release(output)

    def test_source_drift_leaves_no_candidate(self) -> None:
        original_copy = shutil.copy2

        def drifting_copy(source: Path, target: Path) -> str:
            result = original_copy(source, target)
            if source.name == "reference.py":
                source.write_text('"""Changed fixture."""\n', encoding="utf-8")
            return str(result)

        output = self.parent / "candidate"
        with (
            patch.object(release.shutil, "copy2", side_effect=drifting_copy),
            self.assertRaisesRegex(ReleaseError, "source changed"),
        ):
            release.build_candidate(self.root, output)
        self.assertFalse(output.exists())

    def test_export_runs_hygiene_on_selected_files(self) -> None:
        local = "/" + "Users/example/" + "private.json"
        (self.root / "README.md").write_text(local + "\n", encoding="utf-8")
        output = self.parent / "candidate"
        with self.assertRaisesRegex(ReleaseError, "machine-local absolute path"):
            release.build_candidate(self.root, output)
        self.assertFalse(output.exists())

    def test_cli_check_export_verify_and_missing_manifest(self) -> None:
        output = self.parent / "candidate"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(release.main(["--root", str(self.root), "--check"]), 0)
            self.assertEqual(release.main(["--root", str(self.root), "--verify"]), 1)
            self.assertEqual(
                release.main(["--root", str(self.root), "--output", str(output)]), 0
            )
            self.assertEqual(release.main(["--root", str(output), "--verify"]), 0)
            self.assertEqual(release.main(["--root", str(output), "--check"]), 0)
        self.assertIn("release hashes verified", stdout.getvalue())
        self.assertIn("[FAIL] companion release", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
