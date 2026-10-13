"""Validate and optionally execute the companion chapter coverage manifest."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .chapter_coverage_imports import dependency_import_failures
from .chapter_coverage_readme import readme_coverage_failures


MANIFEST_NAME = "chapter_coverage.json"
REQUIRED_FIELDS = {
    "printed_chapter",
    "slug",
    "title",
    "directory",
    "reference",
    "test",
}
OPTIONAL_FIELDS = {"status"}
VALID_STATUSES = {"implemented", "planned", "historical"}
DEFAULT_TEST_TIMEOUT_SECONDS = 120.0
TEST_TIMEOUT_ENV = "KIMI_COMPANION_TEST_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class ChapterEntry:
    """One directly supported printed chapter."""

    printed_chapter: int | None
    slug: str
    title: str
    directory: str
    reference: str
    test: str
    status: str


def parse_test_timeout_seconds(value: str | float) -> float:
    """Return one finite, positive chapter-test timeout."""

    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("chapter-test timeout must be a positive number") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("chapter-test timeout must be a finite positive number")
    return seconds


def configured_test_timeout_seconds(
    environ: Mapping[str, str] | None = None,
) -> float:
    """Read the optional standalone test timeout from the environment."""

    values = os.environ if environ is None else environ
    raw_value = values.get(TEST_TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_TEST_TIMEOUT_SECONDS
    try:
        return parse_test_timeout_seconds(raw_value)
    except ValueError as error:
        raise ValueError(f"{TEST_TIMEOUT_ENV}: {error}") from error


def _single_component(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and Path(value).name == value
        and value not in {".", ".."}
    )


def _entry_field_failures(raw: Mapping[str, object], label: str) -> list[str]:
    """Return missing or unexpected manifest-field failures."""

    missing = sorted(REQUIRED_FIELDS - set(raw))
    extra = sorted(set(raw) - REQUIRED_FIELDS - OPTIONAL_FIELDS)
    failures: list[str] = []
    if missing:
        failures.append(f"{label}: missing fields: {', '.join(missing)}")
    if extra:
        failures.append(f"{label}: unexpected fields: {', '.join(extra)}")
    return failures


def _chapter_number_failures(value: Any, label: str, interlude: bool) -> list[str]:
    """Keep interlude identity separate from the positive chapter-number domain."""
    if interlude:
        return (
            []
            if value is None
            else [f"{label}.printed_chapter: an interlude must use null"]
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return [f"{label}.printed_chapter: expected a positive integer"]
    return []


def _parse_entry(
    raw: Any, index: int, *, interlude: bool = False
) -> tuple[ChapterEntry | None, list[str]]:
    label = f"{'interludes' if interlude else 'chapters'}[{index}]"
    if not isinstance(raw, dict):
        return None, [f"{label}: expected an object"]
    failures = _entry_field_failures(raw, label)
    if failures:
        return None, failures

    failures.extend(_chapter_number_failures(raw["printed_chapter"], label, interlude))
    for field in ("slug", "title"):
        if not isinstance(raw[field], str) or not raw[field].strip():
            failures.append(f"{label}.{field}: expected nonempty text")
    for field in ("directory", "reference", "test"):
        if not _single_component(raw[field]):
            failures.append(f"{label}.{field}: expected one relative path component")
    status = raw.get("status", "implemented")
    if status not in VALID_STATUSES:
        failures.append(
            f"{label}.status: expected one of {', '.join(sorted(VALID_STATUSES))}"
        )
    if failures:
        return None, failures
    return ChapterEntry(
        printed_chapter=raw["printed_chapter"],
        slug=raw["slug"],
        title=raw["title"],
        directory=raw["directory"],
        reference=raw["reference"],
        test=raw["test"],
        status=status,
    ), []


def load_manifest(root: Path) -> tuple[list[ChapterEntry], list[str]]:
    """Load the standalone coverage manifest and return entries plus failures."""

    path = root / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [f"{path}: could not load manifest: {exc}"]
    required = {"schema_version", "chapters"}
    if (
        not isinstance(raw, dict)
        or not required <= set(raw)
        or set(raw) - required - {"interludes"}
    ):
        return [], [
            f"{path}: expected schema_version, chapters, and optional interludes"
        ]
    failures: list[str] = []
    if raw["schema_version"] != 1:
        failures.append(f"{path}: schema_version must be 1")
    if not isinstance(raw["chapters"], list) or not raw["chapters"]:
        failures.append(f"{path}: chapters must be a nonempty list")
        return [], failures

    entries: list[ChapterEntry] = []
    for index, item in enumerate(raw["chapters"]):
        entry, entry_failures = _parse_entry(item, index)
        failures.extend(entry_failures)
        if entry is not None:
            entries.append(entry)
    interludes = raw.get("interludes", [])
    if not isinstance(interludes, list):
        failures.append(f"{path}: interludes must be a list")
        return entries, failures
    for index, item in enumerate(interludes):
        entry, entry_failures = _parse_entry(item, index, interlude=True)
        failures.extend(entry_failures)
        if entry is not None:
            entries.append(entry)
    return entries, failures


def _duplicate_failure(label: str, values: Sequence[Any]) -> str | None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if not duplicates:
        return None
    return f"duplicate {label}: {', '.join(map(str, duplicates))}"


def coverage_failures(root: Path) -> tuple[list[ChapterEntry], list[str]]:
    """Return manifest entries and repository coverage failures."""

    entries, failures = load_manifest(root)
    active_numbers = [
        entry.printed_chapter
        for entry in entries
        if entry.status == "implemented" and entry.printed_chapter is not None
    ]
    slugs = [entry.slug for entry in entries]
    directories = [entry.directory for entry in entries]
    tests = [entry.test for entry in entries]
    failures.extend(
        filter(
            None,
            (
                _duplicate_failure(label, values)
                for label, values in (
                    ("printed chapter", active_numbers),
                    ("slug", slugs),
                    ("directory", directories),
                    ("test filename", tests),
                )
            ),
        )
    )
    if active_numbers != sorted(active_numbers):
        failures.append("implemented chapters must be ordered by printed_chapter")
    status_order = [
        entry.status == "historical"
        for entry in entries
        if entry.printed_chapter is not None
    ]
    if status_order != sorted(status_order):
        failures.append("historical entries must follow active and planned entries")

    for entry in entries:
        chapter_dir = root / entry.directory
        if entry.slug != entry.directory:
            failures.append(
                f"chapter {entry.printed_chapter}: slug and directory must agree"
            )
        if not chapter_dir.is_dir():
            failures.append(f"{entry.directory}: chapter directory is missing")
            continue
        for filename in ("__init__.py", "README.md", entry.reference, entry.test):
            if not (chapter_dir / filename).is_file():
                failures.append(
                    f"{entry.directory}/{filename}: required file is missing"
                )

    discovered = sorted(
        path.parent.name
        for path in root.glob("*/reference.py")
        if not path.parent.name.startswith("_")
    )
    undeclared = sorted(set(discovered) - set(directories))
    stale = sorted(set(directories) - set(discovered))
    failures.extend(
        f"{directory}: reference implementation is undeclared"
        for directory in undeclared
    )
    failures.extend(
        f"{directory}: manifest entry has no reference.py" for directory in stale
    )
    failures.extend(dependency_import_failures(root))
    failures.extend(readme_coverage_failures(root, entries))
    return entries, failures


def run_chapter_tests(
    root: Path,
    entries: list[ChapterEntry],
    *,
    capture_output: bool = False,
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
) -> list[str]:
    """Execute primary and supplementary tests in each registered directory."""

    timeout_seconds = parse_test_timeout_seconds(timeout_seconds)
    failures: list[str] = []
    runner = root / "_maintenance" / "chapter_test_runner.py"
    if not runner.is_file():
        return [f"{runner}: dependency-free chapter test runner is missing"]
    for entry in entries:
        chapter_dir = root / entry.directory
        test_paths = [chapter_dir / entry.test]
        test_paths.extend(
            path
            for path in sorted(chapter_dir.glob("test_*.py"))
            if path.name != entry.test
        )
        failures.extend(
            _run_test_paths(root, test_paths, runner, capture_output, timeout_seconds)
        )
    return failures


def _run_test_paths(
    root: Path,
    test_paths: list[Path],
    runner: Path,
    capture_output: bool,
    timeout_seconds: float,
) -> list[str]:
    """Run a bounded process for each test and retain every failure."""

    failures: list[str] = []
    for test_path in test_paths:
        label = test_path.relative_to(root).as_posix()
        command = [sys.executable, "-B", str(runner), str(test_path)]
        print(f"check: {label}")
        try:
            completed = subprocess.run(
                command,
                cwd=test_path.parent,
                check=False,
                capture_output=capture_output,
                text=capture_output,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{label}: timed out after {timeout_seconds:g} seconds")
            continue
        if completed.returncode:
            failures.append(f"{label}: exited {completed.returncode}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="companion-code root (default: directory containing this module)",
    )
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        help=(f"per-test timeout; defaults to 120 seconds or ${TEST_TIMEOUT_ENV}"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    entries, failures = coverage_failures(root)
    if not failures and args.run_tests:
        try:
            timeout_seconds = (
                configured_test_timeout_seconds()
                if args.timeout_seconds is None
                else parse_test_timeout_seconds(args.timeout_seconds)
            )
        except ValueError as error:
            print(f"[FAIL] {error}", file=sys.stderr)
            return 2
        failures.extend(
            run_chapter_tests(root, entries, timeout_seconds=timeout_seconds)
        )
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    suffix = " and tests passed" if args.run_tests else " passed"
    numbered = sum(
        entry.status == "implemented" and entry.printed_chapter is not None
        for entry in entries
    )
    historical = sum(entry.status == "historical" for entry in entries)
    interludes = sum(entry.printed_chapter is None for entry in entries)
    print(
        f"[OK] companion coverage{suffix}: {numbered} active chapter(s), {historical} historical snapshot(s), {interludes} interlude(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
