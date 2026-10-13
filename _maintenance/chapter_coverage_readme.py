"""Validate the companion chapter coverage table in README.md."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chapter_coverage import ChapterEntry


README_COVERAGE_ROW_RE = re.compile(
    r"^\| (?P<chapter>[1-9][0-9]*|Historical [1-9][0-9]*|Interlude)\. (?P<title>.+) "
    r"\| `(?P<directory>[^`/]+)/` \|$"
)


def readme_coverage_failures(root: Path, entries: list[ChapterEntry]) -> list[str]:
    """Keep the human-readable coverage table synchronized with the manifest."""

    path = root / "README.md"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        return [f"{path}: could not read coverage documentation: {error}"]
    observed = [
        (
            None
            if match.group("chapter") == "Interlude"
            else match.group("chapter")
            if match.group("chapter").startswith("Historical ")
            else int(match.group("chapter")),
            match.group("title"),
            match.group("directory"),
        )
        for line in lines
        if (match := README_COVERAGE_ROW_RE.fullmatch(line)) is not None
    ]
    expected = [
        (
            entry.printed_chapter
            if entry.status != "historical"
            else f"Historical {entry.printed_chapter}",
            entry.title,
            entry.directory,
        )
        for entry in entries
    ]
    if observed == expected:
        return []
    failures = [
        "README.md coverage table is stale; its chapter number, title, and "
        "directory rows must exactly match chapter_coverage.json"
    ]
    for row, (wanted, actual) in enumerate(zip_longest(expected, observed), start=1):
        if wanted != actual:
            failures.append(
                f"README.md coverage row {row}: expected {wanted!r}, found {actual!r}"
            )
    return failures
