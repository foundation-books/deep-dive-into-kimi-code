"""Replay the authored validation-ladder study with portable local subprocesses."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any

from ._data import load_data

FIXTURE = Path(__file__).with_name("validation_fixture.py")
POLICIES = ("broad_first", "targeted_first", "coverage_cost_adaptive")
TARGETED = ("syntax", "basic", "trim", "empty", "casefold", "order")
METRICS = (
    "defect_detection",
    "predicate_coverage",
    "false_confidence",
    "commands",
    "wall_ns",
    "early_stop",
)


def digest(file: Path) -> str:
    """Hash the exact authored candidate bytes used by the subprocess tests."""
    return sha256(file.read_bytes()).hexdigest()


def source_text(spec: dict[str, Any], patch: dict[str, Any], *, baseline: bool) -> str:
    """Run the source text step for the authored validation fixture."""
    expression = (
        patch.get(
            "baseline_expression", spec["repository"]["default_baseline_expression"]
        )
        if baseline
        else patch["expression"]
    )
    return spec["repository"]["module_template"].format(expression=expression) + (
        "# baseline\n" if baseline else f"# patch {patch['id']}\n"
    )


def command(test_id: str, source: Path) -> dict[str, Any]:
    """Run the command step for the authored validation fixture."""
    argv = [
        sys.executable,
        "-B",
        str(FIXTURE),
        "--case",
        test_id,
        "--source",
        str(source),
    ]
    start = perf_counter_ns()
    try:
        completed = subprocess.run(
            argv, text=True, capture_output=True, timeout=10, check=False
        )
        return {
            "test_id": test_id,
            "command": f"python -B experiments/validation_fixture.py --case {test_id} --source src/labels.py",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timeout": False,
            "wall_ns": perf_counter_ns() - start,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "test_id": test_id,
            "command": f"python -B experiments/validation_fixture.py --case {test_id} --source src/labels.py",
            "returncode": None,
            "stdout": (error.stdout or b"").decode("utf-8", "replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or ""),
            "stderr": (error.stderr or b"").decode("utf-8", "replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or ""),
            "timeout": True,
            "wall_ns": perf_counter_ns() - start,
        }


def choose_next(
    spec: dict[str, Any], policy: str, executed: list[str], covered: set[str]
) -> str:
    """Run the choose next step for the authored validation fixture."""
    if policy == "broad_first":
        return "broad"
    if policy == "targeted_first":
        return next(test_id for test_id in TARGETED if test_id not in executed)
    tests = spec["tests"]
    available = [
        test
        for test in tests
        if test["id"] not in executed and (test["id"] != "broad" or executed)
    ]
    return max(
        available,
        key=lambda test: (
            len(set(test["covers"]) - covered) / test["nominal_cost_ms"],
            -next(i for i, item in enumerate(tests) if item["id"] == test["id"]),
        ),
    )["id"]


def run_cell(
    spec: dict[str, Any],
    patch: dict[str, Any],
    source: Path,
    oracle_defect: bool,
    policy: str,
    repetition: int,
) -> dict[str, Any]:
    """Run the run cell step for the authored validation fixture."""
    tests = {test["id"]: test for test in spec["tests"]}
    required = set(spec["required_predicates"])
    commands: list[dict[str, Any]] = []
    covered: set[str] = set()
    attempted: set[str] = set()
    while covered != required:
        test_id = choose_next(
            spec, policy, [row["test_id"] for row in commands], covered
        )
        row = command(test_id, source)
        commands.append(row)
        attempted.update(tests[test_id]["covers"])
        if row["returncode"] != 0 or row["timeout"]:
            break
        covered.update(tests[test_id]["covers"])
    failed = any(row["returncode"] != 0 or row["timeout"] for row in commands)
    green = not failed and covered == required
    return {
        "patch_id": patch["id"],
        "policy": policy,
        "repetition": repetition,
        "candidate_sha256": digest(source),
        "oracle_defect": oracle_defect,
        "commands": commands,
        "metrics": {
            "defect_detection": int(oracle_defect and failed),
            "predicate_coverage": len(covered),
            "attempted_predicates": sorted(attempted),
            "uncovered_required_predicates": sorted(required - covered),
            "false_confidence": int(green and oracle_defect),
            "commands": len(commands),
            "wall_ns": sum(row["wall_ns"] for row in commands),
            "early_stop": int(failed),
            "green": green,
        },
    }


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the summarize step for the authored validation fixture."""
    summary: dict[str, Any] = {}
    for policy in POLICIES:
        group = [cell for cell in cells if cell["policy"] == policy]
        summary[policy] = {
            "patch_repetitions": len(group),
            **{name: sum(cell["metrics"][name] for cell in group) for name in METRICS},
            "green": sum(cell["metrics"]["green"] for cell in group),
            "exact_uncovered_predicate_counts": {
                predicate: sum(
                    predicate in cell["metrics"]["uncovered_required_predicates"]
                    for cell in group
                )
                for predicate in (
                    "import",
                    "basic",
                    "trim",
                    "empty",
                    "casefold",
                    "order",
                )
            },
        }
    return summary


def _signature(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_id": cell["patch_id"],
        "policy": cell["policy"],
        "candidate_sha256": cell["candidate_sha256"],
        "oracle_defect": cell["oracle_defect"],
        "commands": [
            {key: row[key] for key in ("test_id", "returncode", "timeout")}
            for row in cell["commands"]
        ],
        "metrics": {
            key: value for key, value in cell["metrics"].items() if key != "wall_ns"
        },
    }


def verify_cells(
    observed: list[dict[str, Any]], archived: list[dict[str, Any]]
) -> None:
    """Compare every recorded repetition with its portable deterministic replay."""
    index = {(row["patch_id"], row["policy"]): _signature(row) for row in observed}
    expected_grid = {
        (patch, policy, repetition)
        for patch, policy in index
        for repetition in range(10)
    }
    actual_grid = {
        (row["patch_id"], row["policy"], row["repetition"]) for row in archived
    }
    if len(index) != 27 or len(archived) != 270 or actual_grid != expected_grid:
        raise ValueError("validation: incomplete patch/policy grid")
    for cell in archived:
        key = (cell["patch_id"], cell["policy"])
        if index[key] != _signature(cell):
            raise ValueError(f"validation: replay differs for {key}")


def run() -> dict[str, Any]:
    """Replay each patch/policy once; keep historical measured times separate."""
    packet = load_data("validation")
    spec = packet["fixture"]
    cells: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="kimi-validation-") as temporary:
        source = Path(temporary) / "labels.py"
        for patch in spec["patches"]:
            source.write_text(
                source_text(spec, patch, baseline=False), encoding="utf-8"
            )
            oracle = [
                command(test_id, source)
                for test_id in spec["independent_oracle"]["tests"]
            ]
            defect = any(row["returncode"] != 0 or row["timeout"] for row in oracle)
            cells.extend(
                run_cell(spec, patch, source, defect, policy, 0) for policy in POLICIES
            )
    verify_cells(cells, packet["recorded_cells"])
    return {
        "experiment": "validation",
        "status": "pass",
        "checks": len(packet["recorded_cells"]),
        "local_replay_cells": len(cells),
        "local_summary": summarize(cells),
        "recorded_summary": packet["recorded_summary"],
        "claim_boundary": packet["provenance"]["claim_boundary"],
        "timing_note": "Local wall times are new observations on your CPU; historical wall times are reported separately.",
    }
