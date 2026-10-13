"""Small checks against exact pinned source files; no upstream package install."""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, List


BENCH_SHA256 = "358519da832192e407c9fa046f636ba758ca0ce34354b22b024136d7fa80b9b6"
DESIGN_SHA256 = "94a8f9227a668a0c5761f902ed4e93e51dfd1153c0558c28c2a230e67db7c955"
CORE_SHA256 = "f863f44be9cb14836b15e1fc200a2c0eb36c2b07aef2441edc4069f00d6933b0"
ROOT = Path(__file__).resolve().parents[1]


def checked_source(source: Path, expected_hash: str) -> None:
    """Require the exact inspected upstream file before executing a probe."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Pinned source hash mismatch: {source} has {digest}")


def source_function(source: Path, name: str, namespace: dict[str, object]) -> Any:
    """Extract and execute one pinned function body without its heavy imports."""
    matches = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Pinned function {name} not found exactly once")
    # The caller verifies this exact pinned file hash before extracting the body.
    source_body: list[ast.stmt] = list(matches)
    exec(  # noqa: S102
        compile(ast.Module(body=source_body, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


def run_bench_aggregate(source: Path) -> dict[str, object]:
    """Execute the pinned Bolt.diy aggregator on four book-created cases."""
    checked_source(source, BENCH_SHA256)
    with tempfile.TemporaryDirectory(prefix="wva-bench-") as temp_dir:
        temp = Path(temp_dir)
        shutil.copy2(source, temp / "compute_acc.py")
        # tqdm is only a progress display in this source path.
        (temp / "tqdm.py").write_text(
            "def tqdm(items, *args, **kwargs):\n    return items\n"
        )
        (temp / "data").mkdir()
        cases = [
            ("Select A in [3,8]", "A2 and A3 appear", "YES"),
            ("Export selected rows", "CSV contains A2 and A3 only", "PARTIAL"),
            ("Inspect chart legend", "Legend is visible", "NO"),
            ("Operate group selector", "Group changes by pointer", None),
        ]
        fixture = {
            "Category": {"primary_category": "User Interaction"},
            "ui_instruct": [
                {
                    "task": task,
                    "expected_result": expected,
                    "task_category": {"primary_category": "Functional Testing"},
                }
                for task, expected, _ in cases
            ],
        }
        (temp / "data" / "test.jsonl").write_text(json.dumps(fixture) + "\n")
        result_dir = temp / "fixture" / "extracted" / "results"
        result_dir.mkdir(parents=True)
        for index, (_, _, judgment) in enumerate(cases):
            if judgment is None:
                continue
            item_dir = result_dir / f"task_1_{index}"
            item_dir.mkdir()
            (item_dir / "interact_messages.json").write_text(
                json.dumps([{"role": "assistant", "content": judgment}]) + "\n"
            )
        run = subprocess.run(
            [sys.executable, str(temp / "compute_acc.py"), str(temp / "fixture")],
            cwd=temp,
            capture_output=True,
            text=True,
            check=True,
        )
        table = (
            (temp / "fixture" / "extracted" / "table.md").read_text().splitlines()[-1]
        )
        cells = [cell.strip() for cell in table.strip("| ").split("|")]
        yes, partial, no, missing, total = (int(value) for value in cells[1:6])
        accuracy = float(cells[10])
        assert (yes, partial, no, missing, total, accuracy) == (1, 1, 1, 1, 4, 37.5)
        assert "start_failed: 1" in run.stdout
        return {
            "sourceSha256": BENCH_SHA256,
            "judgments": [judgment for _, _, judgment in cases],
            "counts": {"yes": yes, "partial": partial, "no": no, "missing": missing},
            "total": total,
            "partialCreditPercent": accuracy,
        }


def run_position_component(source: Path, results: Path) -> dict[str, object]:
    """Execute only the pinned position-distance helper on captured DOM boxes."""
    checked_source(source, DESIGN_SHA256)
    namespace: dict[str, object] = {}
    distance = source_function(source, "calculate_distance_max_1d", namespace)
    control = json.loads(results.read_text())["metricControl"]
    baseline = control["baseline"]["box"]
    values = {}
    for condition in ("baseline", "layout", "capture"):
        box = control[condition]["box"]
        values[condition] = 1 - distance(
            baseline["x"], baseline["y"], box["x"], box["y"]
        )
    assert abs(values["baseline"] - 1) < 1e-12
    assert abs(values["layout"] - 0.95) < 1e-12
    return {
        "sourceSha256": DESIGN_SHA256,
        "input": "Chart-title DOM center normalized by capture viewport; matching held fixed",
        "positionSubformula": values,
        "scope": "Pinned distance helper only; OCR-free matching, CLIP, and full evaluator not run",
    }


class MaskTensor:
    """Only the indexing, cloning, and multiplication used by the source probe."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = [list(row) for row in rows]

    def clone(self) -> MaskTensor:
        return MaskTensor(self.rows)

    def float(self) -> MaskTensor:
        return self

    def __getitem__(self, key: tuple[int, slice]) -> MaskSlice:
        row, span = key
        return MaskSlice(self.rows[row][span])

    def __setitem__(
        self, key: tuple[int, slice], values: Iterable[builtins.float]
    ) -> None:
        row, span = key
        self.rows[row][span] = list(values)


class MaskSlice:
    """Mutable slice with the scalar multiplication used by the estimator."""

    def __init__(self, values: Iterable[float]) -> None:
        self.values = list(values)

    def __mul__(self, coefficient: float) -> list[float]:
        return [value * coefficient for value in self.values]


def run_step_coefficients(source: Path) -> dict[str, object]:
    """Execute the pinned function body with a tiny scalar tensor shim."""
    checked_source(source, CORE_SHA256)

    def tensor(value: list[float] | float, dtype: object = None) -> list[float] | float:
        """Provide only the scalar tensor construction this function uses."""
        return [float(x) for x in value] if isinstance(value, list) else float(value)

    shim = SimpleNamespace(
        Tensor=MaskTensor,
        float64=float,
        no_grad=contextlib.nullcontext,
        tensor=tensor,
        mean=lambda values: sum(values) / len(values),
        std=lambda values: math.sqrt(
            sum((x - sum(values) / len(values)) ** 2 for x in values)
            / (len(values) - 1)
        ),
    )
    namespace = {
        "List": List,
        "torch": shim,
        "np": SimpleNamespace(ndarray=list),
        "copy": copy,
        "defaultdict": defaultdict,
    }
    estimator = source_function(source, "compute_step_level_grpo_advantage", namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        result, _ = estimator(
            [
                [{"reward": 1, "position": 5}, {"reward": 3, "position": -1}],
                [{"reward": 5, "position": -1}],
            ],
            MaskTensor([[1, 1, 0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]),
            [0, 0],
        )
        equal, _ = estimator(
            [[{"reward": 4, "position": 1}, {"reward": 4, "position": -1}]],
            MaskTensor([[1, 1]]),
            [0],
        )
        singleton, _ = estimator(
            [[{"reward": 2, "position": -1}]], MaskTensor([[1]]), [0]
        )
    assert all(abs(result.rows[0][i] + 0.99999950000025) < 1e-9 for i in (0, 1))
    assert result.rows[0][2:] == [0, 0, 0, 0, 0, 0]
    assert all(abs(result.rows[1][i] - 0.99999950000025) < 1e-9 for i in (0, 1, 2))
    assert equal.rows == [[0.0, 0.0]]
    assert abs(singleton.rows[0][0] - 1.999998000002) < 1e-9
    return {
        "sourceSha256": CORE_SHA256,
        "maskedCoefficients": result.rows,
        "equalRewards": equal.rows,
        "singleton": singleton.rows,
        "scope": "Pinned estimator function body with a minimal scalar tensor shim; no Torch, Verl, or trainer run",
    }


def main() -> None:
    """Run the three bounded probes and print their evidence classes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webgen-bench-source", type=Path, required=True)
    parser.add_argument("--design2code-source", type=Path, required=True)
    parser.add_argument("--webgen-core-source", type=Path, required=True)
    parser.add_argument(
        "--dashboard-results", type=Path, default=ROOT / "evidence" / "results.json"
    )
    args = parser.parse_args()
    output = {
        "webGenBench": run_bench_aggregate(args.webgen_bench_source),
        "design2codePosition": run_position_component(
            args.design2code_source, args.dashboard_results
        ),
        "webGenStep": run_step_coefficients(args.webgen_core_source),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
