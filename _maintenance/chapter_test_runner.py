"""Run one companion test module with only the Python standard library."""

from __future__ import annotations

import argparse
from importlib import import_module
import inspect
from pathlib import Path
import sys
import unittest


def test_suite(namespace: dict[str, object], module_name: str) -> unittest.TestSuite:
    """Discover local TestCase classes and zero-argument test functions."""

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for name, value in sorted(namespace.items()):
        if (
            inspect.isclass(value)
            and issubclass(value, unittest.TestCase)
            and value.__module__ == module_name
        ):
            suite.addTests(loader.loadTestsFromTestCase(value))
            continue
        if not (
            name.startswith("test_")
            and inspect.isfunction(value)
            and value.__module__ == module_name
        ):
            continue
        required = [
            parameter.name
            for parameter in inspect.signature(value).parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if required:
            joined = ", ".join(required)
            raise ValueError(
                f"{name} requires runner-provided arguments ({joined}); "
                "companion tests must be dependency-free"
            )
        suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite


def _import_test_module(path: Path) -> tuple[str, dict[str, object]]:
    root = path.parents[1]
    relative = path.relative_to(root).with_suffix("")
    if not all(part.isidentifier() for part in relative.parts):
        raise ValueError(f"test path is not an importable module: {relative}")
    module_name = ".".join(relative.parts)
    sys.path.insert(0, str(root))
    try:
        module = import_module(module_name)
    finally:
        sys.path.pop(0)
    return module_name, vars(module)


def main(argv: list[str] | None = None) -> int:
    """Execute one test file and reject modules with no runnable tests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file", type=Path)
    args = parser.parse_args(argv)
    if not __debug__:
        print("[FAIL] companion tests require Python assertions", file=sys.stderr)
        return 2
    path = args.test_file.resolve()
    if not path.is_file():
        print(f"[FAIL] companion test file is missing: {path}", file=sys.stderr)
        return 2
    try:
        module_name, namespace = _import_test_module(path)
        suite = test_suite(namespace, module_name)
    except (ImportError, OSError, SyntaxError, ValueError) as error:
        print(f"[FAIL] {path}: {error}", file=sys.stderr)
        return 2
    count = suite.countTestCases()
    if count == 0:
        print(f"[FAIL] {path}: no runnable companion tests found", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        return 1
    print(
        f"[OK] {path.name}: {result.testsRun} companion test(s) run"
        f" ({len(result.skipped)} skipped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
