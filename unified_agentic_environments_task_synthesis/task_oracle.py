"""Independent arithmetic and bounded episode checks for the analysis packet.

Only repository-defined, named repair candidates may run. This is a teaching
execution boundary, not a sandbox for arbitrary Python supplied by strangers.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any, Mapping

from .task_factory import (
    ROOT,
    Task,
    construct_task,
    fresh_workspace,
    validate_reference_reads,
)


_BUG = (
    "return sum(correct for _, support, correct in rows) / sum(\n"
    "        support for _, support, correct in rows\n"
    "    )"
)
_REPAIR_LINES = {
    "buggy": _BUG,
    "reference": "return sum(correct / support for _, support, correct in rows) / len(rows)",
    "alternative": "return (rows[0][2] / rows[0][1] + rows[1][2] / rows[1][1]) / 2",
    "hardcoded": "return 0.50 if rows[0][2] == 81 else 0.80",
    "swapped": "return (rows[0][2] / rows[1][1] + rows[1][2] / rows[0][1]) / 2",
    "tamper-counts": _BUG,
}
_RUNNER = (
    "import json,runpy,sys\n"
    "score=runpy.run_path(sys.argv[1])['score']\n"
    "out=[]\n"
    "for rows in json.load(sys.stdin):\n"
    " try: out.append(score(rows))\n"
    " except ValueError: out.append('ValueError')\n"
    "print(json.dumps(out))\n"
)


def read_counts(file: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Read the visible fictional counts with an explicit two-class domain."""
    groups: dict[str, list[tuple[str, int, int]]] = {}
    with file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["classifier"]
            groups.setdefault(name, []).append(
                (row["true_class"], int(row["support"]), int(row["correct"]))
            )
    for rows in groups.values():
        validate_rows(rows)
    if set(groups) != {"Atlas", "Boreal"}:
        raise ValueError("the displayed packet requires Atlas and Boreal")
    return groups


def validate_rows(rows: list[tuple[str, int, int]]) -> None:
    """Reject malformed count rows before any metric division."""
    if len(rows) != 2 or len({row[0] for row in rows}) != 2:
        raise ValueError("exactly two distinct classes are required")
    for label, support, correct in rows:
        if not label or not isinstance(label, str):
            raise ValueError("class label must be nonempty text")
        if type(support) is not int or support <= 0:
            raise ValueError("support must be a positive integer")
        if type(correct) is not int or not 0 <= correct <= support:
            raise ValueError("correct must lie between zero and support")


def independent_scores(
    rows: list[tuple[str, int, int]],
) -> tuple[tuple[Fraction, Fraction], Fraction, Fraction]:
    """Compute class recalls, mean recall, and pooled accuracy with Fractions."""
    validate_rows(rows)
    recalls = tuple(Fraction(correct, support) for _, support, correct in rows)
    balanced = sum(recalls, Fraction()) / len(recalls)
    ordinary = Fraction(
        sum(correct for _, _, correct in rows),
        sum(support for _, support, _ in rows),
    )
    return (recalls[0], recalls[1]), balanced, ordinary


def displayed_scores() -> dict[str, dict[str, str]]:
    """Return the four chapter aggregates from the pinned local count file."""
    result: dict[str, dict[str, str]] = {}
    for name, rows in read_counts(ROOT / "fixtures" / "counts.csv").items():
        recalls, balanced, ordinary = independent_scores(rows)
        result[name] = {
            "class_0_recall": str(float(recalls[0])),
            "class_1_recall": str(float(recalls[1])),
            "balanced_accuracy": str(float(balanced)),
            "ordinary_accuracy": str(float(ordinary)),
        }
    return result


def _protected_input_cases() -> list[list[tuple[str, int, int]]]:
    base = read_counts(ROOT / "fixtures" / "counts.csv")
    return [
        base["Atlas"],
        base["Boreal"],
        list(reversed(base["Atlas"])),
        [("zero", 50, 45), ("one", 50, 5)],
        [("positive", 10, 5), ("negative", 90, 18)],
        [("zero", 90, 80), ("one", 10, 9)],
        [("zero", 0, 0), ("one", 10, 8)],
        [("zero", 10, -1), ("one", 10, 8)],
        [("zero", 10, 11), ("one", 10, 8)],
        [("zero", 10, 8)],
    ]


def _resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def _candidate_source(name: str, original: str) -> str:
    if name not in _REPAIR_LINES:
        raise ValueError("only repository-defined candidate names may execute")
    if original.count(_BUG) != 1:
        raise ValueError("defective source anchor changed")
    return original.replace(_BUG, _REPAIR_LINES[name])


def _check_sources(world: Path, task: Task) -> bool:
    for name in task.files:
        if name in task.editable:
            continue
        if (world / name).read_bytes() != (ROOT / "fixtures" / name).read_bytes():
            return False
    return True


def _run_candidate_program(
    program: Path, cases: list[list[tuple[str, int, int]]], world: Path
) -> tuple[list[Any], str | None]:
    try:
        process = subprocess.run(
            (
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _RUNNER,
                str(program),
            ),
            input=json.dumps(cases),
            text=True,
            capture_output=True,
            cwd=world,
            env={},
            timeout=3,
            check=False,
            preexec_fn=_resource_limits,
        )
    except subprocess.TimeoutExpired:
        return [], "execution time limit"
    if process.returncode != 0:
        return [], "candidate execution error"
    output = json.loads(process.stdout)
    if not isinstance(output, list):
        return [], "candidate output is not a list"
    return output, None


def evaluate_repair_candidate(name: str) -> dict[str, object]:
    """Run a named fixed program in a fresh process on independent cases."""
    task = construct_task("repair")
    with fresh_workspace(task) as world:
        note = (world / "methodology.md").read_text(encoding="utf-8")
        definitions = (world / "metric_definitions.md").read_text(encoding="utf-8")
        if "balanced accuracy" not in note or "mean recall" not in definitions:
            raise ValueError("visible material does not define the required repair")
        program = world / "analysis.py"
        program.write_text(
            _candidate_source(name, program.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        if name == "tamper-counts":
            (world / "counts.csv").chmod(0o644)
            (world / "counts.csv").write_text(
                (world / "counts.csv")
                .read_text()
                .replace("Atlas,Class 1,10,1", "Atlas,Class 1,10,9"),
                encoding="utf-8",
            )
        if not _check_sources(world, task):
            return {"status": "fail", "reason": "protected source changed"}
        cases = _protected_input_cases()
        observed, failure = _run_candidate_program(program, cases, world)
        if failure is not None:
            return {"status": "fail", "reason": failure}
        expected: list[Fraction | str] = []
        for rows in cases:
            try:
                expected.append(independent_scores(rows)[1])
            except ValueError:
                expected.append("ValueError")
        if len(observed) != len(expected):
            return {"status": "fail", "reason": "wrong output count"}
        matches = [
            value == target
            if isinstance(target, str)
            else abs(float(value) - float(target)) < 1e-12
            for value, target in zip(observed, expected, strict=True)
        ]
        result = {
            "status": "pass" if all(matches) else "fail",
            "reason": "all declared cases"
            if all(matches)
            else "changed or invalid input mismatch",
            "cases": len(expected),
            "matched": sum(matches),
        }
        (world / "output" / "test-results.json").write_text(json.dumps(result) + "\n")
        return result


def reference_question(world: Path) -> dict[str, object]:
    """Use only the visible note, definition, and count file for Task A."""
    values = _reference_metrics_from_visible(world)
    return {
        "winner": "Boreal" if values["Boreal"][1] > values["Atlas"][1] else "Atlas",
        "balanced": {name: str(item[1]) for name, item in values.items()},
        "ordinary": {name: str(item[2]) for name, item in values.items()},
        "definition_source": "sklearn-balanced-1.9.1",
        "explanation": "Balanced accuracy gives equal weight to class-wise recall; "
        "ordinary accuracy pools correct predictions, so the rankings reverse.",
    }


def _reference_metrics_from_visible(
    world: Path,
) -> dict[str, tuple[tuple[float, float], float, float]]:
    """Follow the visible specification without calling the evaluator oracle."""
    note = (world / "methodology.md").read_text(encoding="utf-8")
    definitions = (world / "metric_definitions.md").read_text(encoding="utf-8")
    if "balanced accuracy" not in note or "mean recall" not in definitions:
        raise ValueError("visible material does not define the required metric")
    groups = read_counts(world / "counts.csv")
    values: dict[str, tuple[tuple[float, float], float, float]] = {}
    for name, rows in groups.items():
        recalls = (rows[0][2] / rows[0][1], rows[1][2] / rows[1][1])
        balanced = (recalls[0] + recalls[1]) / 2
        ordinary = (rows[0][2] + rows[1][2]) / (rows[0][1] + rows[1][1])
        values[name] = (recalls, balanced, ordinary)
    return values


def evaluate_question(answer: Mapping[str, object]) -> dict[str, object]:
    """Check deterministic fields and a limited explanatory content rubric."""
    expected = displayed_scores()
    numeric = True
    for kind, key in (
        ("balanced", "balanced_accuracy"),
        ("ordinary", "ordinary_accuracy"),
    ):
        supplied = answer.get(kind)
        if not isinstance(supplied, Mapping):
            numeric = False
            break
        for name in ("Atlas", "Boreal"):
            try:
                numeric &= Fraction(str(supplied.get(name, "-1"))) == Fraction(
                    expected[name][key]
                )
            except (ValueError, ZeroDivisionError):
                numeric = False
    prose = str(answer.get("explanation", "")).lower()
    rubric = all(
        word in prose for word in ("equal", "class-wise recall", "pools", "reverse")
    )
    passed = (
        answer.get("winner") == "Boreal"
        and answer.get("definition_source") == "sklearn-balanced-1.9.1"
        and numeric
        and rubric
    )
    return {
        "status": "pass" if passed else "fail",
        "numeric": numeric,
        "limited_prose_rubric": rubric,
    }


def reference_report(world: Path) -> None:
    """Write Task C's required files from visible packet material."""
    values = _reference_metrics_from_visible(world)
    output = world / "output"
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "classifier",
                "class_0_recall",
                "class_1_recall",
                "balanced_accuracy",
                "ordinary_accuracy",
            )
        )
        for name, (recalls, balanced, ordinary) in values.items():
            writer.writerow(
                (name, *(str(value) for value in (*recalls, balanced, ordinary)))
            )
    (output / "audit.md").write_text(
        "# Analysis audit\n\nBoreal wins under balanced accuracy. Each class recall "
        "has equal weight. Atlas wins under ordinary accuracy because the "
        "pooled correct count is dominated by Class 0. The metric definition "
        "comes from scikit-learn 1.9.1 balanced_accuracy_score; counts are "
        "fictional book data.\n",
        encoding="utf-8",
    )


def evaluate_report(world: Path) -> dict[str, object]:
    """Check required files, independent numbers, source bytes, and key claims."""
    task = construct_task("report")
    if not _check_sources(world, task):
        return {"status": "fail", "reason": "protected source changed"}
    scores = world / "output" / "scores.csv"
    audit = world / "output" / "audit.md"
    if not scores.is_file() or not audit.is_file():
        return {"status": "fail", "reason": "required report files missing"}
    try:
        with scores.open(newline="", encoding="utf-8") as stream:
            parsed = list(csv.DictReader(stream))
        rows = {row["classifier"]: row for row in parsed}
        expected = displayed_scores()
        numeric = (
            len(parsed) == len(rows)
            and set(rows) == set(expected)
            and all(
                Fraction(rows[name][key]) == Fraction(value)
                for name, metrics in expected.items()
                for key, value in metrics.items()
            )
        )
    except (OSError, csv.Error, KeyError, TypeError, ValueError, ZeroDivisionError):
        numeric = False
    prose = audit.read_text(encoding="utf-8").lower()
    rubric = all(
        item in prose
        for item in ("boreal", "atlas", "equal weight", "pooled", "scikit-learn")
    )
    return {
        "status": "pass" if numeric and rubric else "fail",
        "numeric": numeric,
        "limited_prose_rubric": rubric,
    }


def run_reference(family: str) -> dict[str, object]:
    """Run each task from a fresh world without previous-family artifacts."""
    task = construct_task(family)
    validate_reference_reads(task, set(task.files))
    if family == "repair":
        return evaluate_repair_candidate("reference")
    with fresh_workspace(task) as world:
        if family == "question":
            answer = reference_question(world)
            return evaluate_question(answer)
        reference_report(world)
        return evaluate_report(world)
