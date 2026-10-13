"""Child-process entry point for a tiny bounded rational-comparison fixture."""

from __future__ import annotations

from fractions import Fraction
import json
import sys
import time


def main() -> int:
    """Read one rational comparison request and write its typed native result."""

    request = json.load(sys.stdin)
    time.sleep(float(request.get("delay_seconds", 0)))
    try:
        reference = Fraction(request["reference"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        json.dump({"invalid": "reference"}, sys.stdout)
        return 0
    try:
        candidate = Fraction(request["candidate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        json.dump({"invalid": "candidate"}, sys.stdout)
        return 0
    json.dump({"accepted": candidate == reference}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
