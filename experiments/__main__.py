"""Run selected book experiments from the companion repository root."""

from __future__ import annotations

import argparse
import json
from typing import Any

from . import gated_mla, kda, moba, recovery, validation
from ._data import check_reference


NAMES = ("kda", "moba", "gated-mla", "validation", "recovery")


def run(name: str) -> dict[str, Any]:
    """Run a finite experiment or analyze its supplied measurement packet."""
    if name == "kda":
        return check_reference(name, kda.run_all_checks())
    if name == "moba":
        return check_reference(name, moba.run_all_checks())
    if name == "gated-mla":
        return gated_mla.run()
    if name == "validation":
        return validation.run()
    if name == "recovery":
        return recovery.run()
    raise ValueError(f"unknown experiment: {name}")


def main() -> int:
    """Print results, or a concise check report; never write evidence files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=NAMES + ("all",))
    parser.add_argument(
        "--check", action="store_true", help="print only verification status"
    )
    args = parser.parse_args()
    names = NAMES if args.experiment == "all" else (args.experiment,)
    for name in names:
        try:
            result = run(name)
        except (ValueError, KeyError, OSError) as error:
            parser.exit(1, f"[FAIL] {name}: {error}\n")
        if args.check:
            print(f"[OK] {name}: {result['checks']} checks")
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
