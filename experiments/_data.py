"""Read release fixtures and compare JSON results without manuscript dependencies."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import json
import math
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


def _float_field_matches(
    observed: Any, expected: Any, tolerance: float, *, error_metric: bool
) -> bool:
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _float_field_matches(left, right, tolerance, error_metric=error_metric)
                for left, right in zip(observed, expected, strict=True)
            )
        )
    if not isinstance(expected, float) or type(observed) not in (int, float):
        return False
    if not math.isfinite(observed):
        return False
    if error_metric:
        return 0.0 <= observed <= tolerance
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance)


def _result_matches(name: str, key: str, observed: Any, expected: Any) -> bool:
    # Only the two floating MoBA oracles permit reduction-rounding differences.
    fields: dict[str, tuple[float, bool]]
    if name == "moba" and key == "RESULT_MOBA_FORWARD_ORACLE":
        fields = {"max_abs_error": (1e-12, True), "final_query_outputs": (1e-12, False)}
    elif name == "moba" and key == "RESULT_MOBA_BACKWARD_ORACLE":
        fields = {"max_abs_error": (1e-8, True), "q_k_v_max_abs_errors": (1e-8, True)}
    else:
        return observed == expected
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    comparable = dict(observed)
    for field, (tolerance, error_metric) in fields.items():
        if field not in observed or field not in expected:
            return False
        if not _float_field_matches(
            observed[field], expected[field], tolerance, error_metric=error_metric
        ):
            return False
        comparable[field] = expected[field]
    return comparable == expected


def check_reference(name: str, observed: dict[str, Any]) -> dict[str, Any]:
    """Compare archived fields, allowing only the documented MoBA float bounds."""
    packet = load_data(name)
    expected = packet["results"]
    actual = json_value(observed)
    differences = [
        key
        for key in set(actual) | set(expected)
        if key not in actual
        or key not in expected
        or not _result_matches(name, key, actual[key], expected[key])
    ]
    if differences:
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
