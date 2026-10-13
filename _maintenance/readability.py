"""Small dependency-free readability budgets for standalone companion Python."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys


SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "results",
}


@dataclass(frozen=True)
class Budget:
    file_lines: int
    function_lines: int
    branches: int
    nested_defs: int


BUDGET_PATH = Path(__file__).resolve().parents[1] / "readability_budget.json"
_CONFIG = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
if _CONFIG.get("schema_version") != 1:
    raise RuntimeError(f"{BUDGET_PATH}: unsupported readability budget schema")


def _budget(name: str) -> Budget:
    values = _CONFIG[name]
    return Budget(
        values["file_lines"],
        values["function_lines"],
        values["branch_points"],
        values["nested_defs"],
    )


PRODUCTION = _budget("production")
TEST = _budget("test")
HEADROOM_RATIO = _CONFIG["headroom_ratio"]
DUPLICATE_FUNCTION_MIN_LINES = _CONFIG["duplicate_function_min_lines"]
BRANCH_TYPES = (
    ast.AsyncFor,
    ast.AsyncWith,
    ast.BoolOp,
    ast.For,
    ast.If,
    ast.IfExp,
    ast.Match,
    ast.Try,
    ast.While,
    ast.With,
)


class BranchCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.count = 0

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, BRANCH_TYPES):
            self.count += 1
        if isinstance(node, (ast.ExceptHandler, ast.match_case)):
            self.count += 1
        if isinstance(node, ast.comprehension):
            self.count += len(node.ifs)
        super().generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node


def branch_count(node: ast.AST) -> int:
    counter = BranchCounter()
    for statement in getattr(node, "body", []):
        counter.visit(statement)
    return counter.count


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _near_limit(used: int, limit: int) -> bool:
    return math.ceil(limit * HEADROOM_RATIO) <= used <= limit


def _is_nested(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        parent = parents.get(parent)
    return False


def failures_for_file(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root)
    budget = TEST if path.name.startswith(("test_", "test-")) else PRODUCTION
    source = path.read_text(encoding="utf-8")
    failures: list[str] = []
    line_count = len(source.splitlines())
    if line_count > budget.file_lines:
        failures.append(f"{relative}: {line_count} lines; limit is {budget.file_lines}")
    if _near_limit(line_count, budget.file_lines):
        failures.append(
            f"{relative}: file has {line_count} lines and is near its "
            f"{budget.file_lines}-line limit"
        )
    tree = ast.parse(source, filename=str(path))
    parents = _parent_map(tree)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in functions:
        function_lines = (node.end_lineno or node.lineno) - node.lineno + 1
        if function_lines > budget.function_lines:
            failures.append(
                f"{relative}:{node.lineno}: {node.name} has {function_lines} lines; "
                f"limit is {budget.function_lines}"
            )
        elif _near_limit(function_lines, budget.function_lines):
            failures.append(
                f"{relative}:{node.lineno}: {node.name} has {function_lines} lines "
                f"and is near its {budget.function_lines}-line limit"
            )
        branches = branch_count(node)
        if branches > budget.branches:
            failures.append(
                f"{relative}:{node.lineno}: {node.name} has {branches} branches; "
                f"limit is {budget.branches}"
            )
        elif _near_limit(branches, budget.branches):
            failures.append(
                f"{relative}:{node.lineno}: {node.name} has {branches} branches "
                f"and is near its {budget.branches}-branch limit"
            )
    nested = sum(_is_nested(node, parents) for node in functions)
    if nested > budget.nested_defs:
        failures.append(
            f"{relative}: has {nested} nested functions; limit is {budget.nested_defs}"
        )
    elif _near_limit(nested, budget.nested_defs):
        failures.append(
            f"{relative}: has {nested} nested functions and is near its "
            f"{budget.nested_defs}-function limit"
        )
    return failures


def python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if SKIP_DIRS.isdisjoint(path.relative_to(root).parts)
    )


def duplicate_function_failures(files: list[Path], root: Path) -> list[str]:
    """Reject exact top-level production functions duplicated across modules."""

    groups: dict[tuple[str, str, str], list[tuple[Path, str, int]]] = {}
    for path in files:
        relative = path.relative_to(root)
        if path.name.startswith(("test_", "test-")) or "tests" in relative.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            lines = (node.end_lineno or node.lineno) - node.lineno + 1
            if lines < DUPLICATE_FUNCTION_MIN_LINES:
                continue
            signature = ast.dump(node.args, include_attributes=False)
            body = ast.dump(
                ast.Module(body=node.body, type_ignores=[]),
                include_attributes=False,
            )
            returns = (
                ast.dump(node.returns, include_attributes=False)
                if node.returns is not None
                else ""
            )
            groups.setdefault((signature, body, returns), []).append(
                (relative, node.name, node.lineno)
            )

    failures: list[str] = []
    for matches in groups.values():
        if len(matches) < 2:
            continue
        locations = ", ".join(
            f"{path}:{line} ({name})" for path, name, line in sorted(matches)
        )
        failures.append(f"exact duplicate production function: {locations}")
    return sorted(failures)


def audit_tree(root: Path) -> list[str]:
    failures: list[str] = []
    files = python_files(root)
    for path in files:
        try:
            failures.extend(failures_for_file(path, root))
        except (OSError, UnicodeError, SyntaxError) as error:
            failures.append(
                f"{path.relative_to(root)}: could not inspect Python: {error}"
            )
    failures.extend(duplicate_function_failures(files, root))
    return failures


def _audit_root(arguments: Sequence[str] | None) -> Path:
    parser = argparse.ArgumentParser(
        description="Enforce companion Python readability budgets."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="directory to audit (default: current directory)",
    )
    root = parser.parse_args(arguments).root.resolve()
    if not root.is_dir():
        parser.error(f"audit root is not a directory: {root}")
    return root


def main(arguments: Sequence[str] | None = None) -> int:
    root = _audit_root(arguments)
    failures = audit_tree(root)
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    print(
        f"[OK] companion Python readability passed ({len(python_files(root))} file(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
