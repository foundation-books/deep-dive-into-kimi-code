"""Validate imports in the dependency-free companion-code tree."""

from __future__ import annotations

import ast
from pathlib import Path
import sys


GENERATED_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


def dependency_import_failures(root: Path) -> list[str]:
    """Reject third-party imports from the dependency-free companion tree."""

    local_roots = {
        path.stem if path.is_file() else path.name
        for path in root.iterdir()
        if path.is_dir() or path.suffix == ".py"
    }
    allowed = set(sys.stdlib_module_names) | local_roots | {"__future__"}
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if not GENERATED_DIRECTORY_NAMES.isdisjoint(path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            failures.append(
                f"{path.relative_to(root)}: could not inspect imports: {error}"
            )
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            import_line = 1
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                import_line = node.lineno
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
                import_line = node.lineno
            for module in modules:
                import_root = module.split(".", 1)[0]
                if import_root not in allowed:
                    failures.append(
                        f"{path.relative_to(root)}:{import_line}: third-party import "
                        f"{import_root!r} violates the dependency-free companion contract"
                    )
    return failures
