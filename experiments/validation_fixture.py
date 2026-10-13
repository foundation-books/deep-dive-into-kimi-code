"""Authored Python test cases for the finite validation-ladder experiment."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def check(case: str, source: Path) -> None:
    """Run the check step for the authored validation fixture."""
    spec = importlib.util.spec_from_file_location("labels", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if case == "syntax":
        return
    samples = {
        "basic": [("Alpha,Beta", ["alpha", "beta"])],
        "trim": [(" Alpha , Beta ", ["alpha", "beta"])],
        "empty": [("a,, ,b", ["a", "b"])],
        "casefold": [("Straße", ["strasse"])],
        "order": [("z,a,z", ["z", "a", "z"])],
        "duplicate": [("a,a", ["a", "a"])],
        "broad": [
            ("Alpha,Beta", ["alpha", "beta"]),
            (" Alpha , Beta ", ["alpha", "beta"]),
            ("a,, ,b", ["a", "b"]),
            ("Straße", ["strasse"]),
            ("z,a,z", ["z", "a", "z"]),
        ],
    }
    for input_text, expected in samples[case]:
        actual = module.normalize_labels(input_text)
        if actual != expected:
            raise AssertionError(f"{case}: {actual!r} != {expected!r}")
    if case == "broad":
        # Make the broad regression genuinely more expensive than one target.
        checksum = 0
        for value in range(300_000):
            checksum = (checksum + value) % 104729
        assert checksum >= 0


def main() -> int:
    """Run the main step for the authored validation fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    try:
        check(args.case, args.source)
    except Exception as error:
        print(f"FAIL {args.case}: {type(error).__name__}: {error}")
        return 1
    print(f"PASS {args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
