"""Scalar float64 reconstruction adapted from the archived Gated MLA reviewer."""

from __future__ import annotations

import math
from typing import Any


def dot(left: list[float], right: list[float]) -> float:
    """Compute dot for the supplied numeric packet."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Compute matvec for the supplied numeric packet."""
    return [dot(row, vector) for row in matrix]


def sigmoid(value: float) -> float:
    """Compute sigmoid for the supplied numeric packet."""
    return (
        1.0 / (1.0 + math.exp(-value))
        if value >= 0
        else math.exp(value) / (1.0 + math.exp(value))
    )


def _head_output(
    source: dict[str, Any],
    head: int,
    row: list[float],
    visible: list[Any],
    scale: float,
    value: int,
) -> list[float]:
    query_content = matvec(source["query_content_weight"][head], row)
    query_shared = matvec(source["query_shared_weight"][head], row)
    keys = [matvec(source["key_up_weight"][head], latent) for latent, _ in visible]
    values = [matvec(source["value_up_weight"][head], latent) for latent, _ in visible]
    scores = [
        (dot(query_content, key) + dot(query_shared, visible[i][1])) * scale
        for i, key in enumerate(keys)
    ]
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    denominator = sum(exponentials)
    probabilities = [item / denominator for item in exponentials]
    return [
        sum(
            probability * row_value[channel]
            for probability, row_value in zip(probabilities, values, strict=True)
        )
        for channel in range(value)
    ]


def materialized_forward(
    source: dict[str, Any], gate: list[Any], case: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct per-head K/V for every visible record, in float64 scalar math."""
    x = source["input_x"]
    heads, value = int(case["heads"]), int(case["value_dim"])
    latent = int(case["latent_dim"])
    scale = 1.0 / math.sqrt(int(case["content_dim"]) + int(case["shared_dim"]))
    projected = [matvec(source["latent_weight"], row) for row in x]
    records = [(row[:latent], row[latent:]) for row in projected]
    attention: list[Any] = []
    output: list[Any] = []
    row_start = incoming_start = 0
    for length, incoming in zip(case["lengths"], case["incoming_lengths"], strict=True):
        row_end, incoming_end = row_start + int(length), incoming_start + int(incoming)
        visible_records = [
            (source["incoming_latent"][i], source["incoming_shared_key"][i])
            for i in range(incoming_start, incoming_end)
        ] + records[row_start:row_end]
        for local, row_index in enumerate(range(row_start, row_end)):
            row = x[row_index]
            visible = visible_records[: int(incoming) + local + 1]
            head_outputs = [
                _head_output(source, head, row, visible, scale, value)
                for head in range(heads)
            ]
            attention.append(head_outputs)
            flat = [
                head_outputs[head][channel] * gate[row_index][head][channel]
                for head in range(heads)
                for channel in range(value)
            ]
            output.append(matvec(source["output_weight"], flat))
        row_start, incoming_start = row_end, incoming_end
    return {
        "attention_output": attention,
        "output": output,
        "record_latent": [row[0] for row in records],
        "record_shared_key": [row[1] for row in records],
    }


def flatten(value: Any) -> list[Any]:
    """Compute flatten for the supplied numeric packet."""
    if isinstance(value, list):
        return [coordinate for item in value for coordinate in flatten(item)]
    return [value]


def quantiles(values: list[float], requested: list[float]) -> dict[str, float]:
    """Compute quantiles for the supplied numeric packet."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty numeric output")
    result: dict[str, float] = {}
    for quantile in requested:
        position = (len(ordered) - 1) * quantile
        low = math.floor(position)
        high = math.ceil(position)
        result[str(quantile)] = ordered[low] + (ordered[high] - ordered[low]) * (
            position - low
        )
    return result


def error_report(
    candidate: Any, reference: Any, policy: dict[str, Any]
) -> dict[str, Any]:
    """Compute error report for the supplied numeric packet."""
    observed = flatten(candidate)
    expected = flatten(reference)
    if len(observed) != len(expected) or not expected:
        return {
            "status": "shape_failure",
            "candidate_coordinates": len(observed),
            "reference_coordinates": len(expected),
        }
    nonfinite = sum(value is None or not math.isfinite(value) for value in observed)
    nonfinite += sum(value is None or not math.isfinite(value) for value in expected)
    if nonfinite:
        return {
            "status": "nonfinite",
            "nonfinite_coordinates": nonfinite,
            "candidate_coordinates": len(observed),
            "reference_coordinates": len(expected),
        }
    absolute = [abs(a - b) for a, b in zip(observed, expected, strict=True)]
    relative = [
        difference / max(abs(value), 1e-6)
        for difference, value in zip(absolute, expected, strict=True)
    ]
    rms_error = math.sqrt(sum(value * value for value in absolute) / len(absolute))
    rms_reference = math.sqrt(sum(value * value for value in expected) / len(expected))
    relative_rms = rms_error / (rms_reference + 1e-8)
    finite_metrics = all(
        math.isfinite(value) for value in absolute + relative + [relative_rms]
    )
    passed = finite_metrics and (
        max(absolute) <= policy["maximum_absolute_tolerance"]
        or relative_rms <= policy["relative_rms_tolerance"]
    )
    return {
        "status": "pass" if passed else "fail",
        "coordinate_count": len(absolute),
        "maximum_absolute_error": max(absolute),
        "maximum_relative_error": max(relative),
        "relative_rms_error": relative_rms,
        "absolute_error_quantiles": quantiles(absolute, policy["quantiles"]),
        "relative_error_quantiles": quantiles(relative, policy["quantiles"]),
        "absolute_errors": absolute,
        "relative_errors": relative,
        "nonfinite_coordinates": 0,
    }


def gate_replay(source: dict[str, Any], gate: Any, case: dict[str, Any]) -> float:
    """Compute gate replay for the supplied numeric packet."""
    observed = flatten(gate)
    heads, value = int(case["heads"]), int(case["value_dim"])
    expected = [
        sigmoid(dot(weight, row))
        for row in source["input_x"]
        for weight in source["gate_weight"]
    ]
    if (
        len(observed) != len(expected)
        or len(expected) != len(source["input_x"]) * heads * value
    ):
        raise ValueError("gate shape mismatch")
    if any(item is None or not math.isfinite(item) for item in observed):
        raise ValueError("nonfinite gate")
    return max(abs(a - b) for a, b in zip(observed, expected, strict=True))


def compare_records(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    """Compute compare records for the supplied numeric packet."""
    differences: list[float] = []
    for key in ("record_latent", "record_shared_key"):
        observed, expected = flatten(candidate[key]), flatten(reference[key])
        if len(observed) != len(expected) or any(item is None for item in observed):
            raise ValueError(f"{key} shape/nonfinite mismatch")
        differences.extend(abs(a - b) for a, b in zip(observed, expected, strict=True))
    return max(differences)
