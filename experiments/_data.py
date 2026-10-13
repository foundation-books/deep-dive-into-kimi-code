"""Read release fixtures and compare JSON results without manuscript dependencies."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


DATA_ROOT = Path(__file__).with_name("data")


def load_data(name: str) -> dict[str, Any]:
    """Load one named fixture only after checking its release checksum."""
    manifest = json.loads(DATA_ROOT.with_name("manifest.json").read_text())
    filename = f"{name}.json"
    expected = manifest["data_sha256"][filename]
    payload = (DATA_ROOT / filename).read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f"{filename}: release fixture checksum differs")
    return json.loads(payload)


def json_value(value: Any) -> Any:
    """Normalize tuples and remove author-side protocol envelopes for comparison."""
    if isinstance(value, dict):
        return {
            key: json_value(item) for key, item in value.items() if key != "protocol"
        }
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def protocol(**fields: str) -> dict[str, str]:
    """Describe a toy comparison without attaching internal evidence identifiers."""
    return fields


def exact_matrix(matrix: Iterable[Iterable[object]]) -> list[list[str]]:
    """Render every exact-arithmetic matrix coordinate without rounding."""
    return [[str(value) for value in row] for row in matrix]


def exact_vector(vector: Iterable[object]) -> list[str]:
    """Render every exact-arithmetic vector coordinate without rounding."""
    return [str(value) for value in vector]


def check_reference(name: str, observed: dict[str, Any]) -> dict[str, Any]:
    """Compare all computed fields with the selected archived toy results."""
    packet = load_data(name)
    expected = packet["results"]
    actual = json_value(observed)
    if actual != expected:
        differences = [
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        ]
        raise ValueError(f"{name}: result differs: {', '.join(sorted(differences))}")
    if any(row["status"] != "pass" for row in actual.values()):
        raise ValueError(f"{name}: a correctness check failed")
    return {
        "experiment": name,
        "status": "pass",
        "checks": len(actual),
        "results": actual,
        "comparisons": {key: row["protocol"] for key, row in observed.items()},
        "claim_boundary": packet["provenance"]["claim_boundary"],
    }
