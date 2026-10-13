"""Recompute numerical errors and allocation summaries from recorded H200 data."""

from __future__ import annotations

import math
import statistics
from typing import Any

from ._data import load_data
from .gated_mla_math import (
    compare_records,
    error_report,
    gate_replay,
    materialized_forward,
)


def analyze_numerics(packet: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct all supplied FP32 cases with independent scalar float64 math."""
    records = packet["records"]
    if len(records) != packet["expected_cells"] or len(records) != 36:
        raise ValueError("Gated MLA numerics: incomplete 36-cell packet")
    policy = packet["numeric_policy"]
    errors = []
    for row in records:
        case, source, candidate = row["case"], row["sources"], row["candidate"]
        if row["status"] != "executed" or row["nonfinite_count"] != 0:
            raise ValueError(f"{case['id']}: producer did not return finite tensors")
        reference = materialized_forward(source, candidate["gate"], case)
        for field in ("attention_output", "output"):
            report = error_report(candidate[field], reference[field], policy)
            if report["status"] != "pass":
                raise ValueError(f"{case['id']}/{field}: numerical gate failed")
            errors.append(report["relative_rms_error"])
        if (
            gate_replay(source, candidate["gate"], case)
            > policy["gate_replay_absolute_tolerance"]
        ):
            raise ValueError(f"{case['id']}: gate replay failed")
        if (
            compare_records(candidate, reference)
            > policy["compact_record_replay_absolute_tolerance"]
        ):
            raise ValueError(f"{case['id']}: compact record replay failed")
    return {"cells": len(records), "maximum_relative_rms_error": max(errors)}


def _allocation_samples(report: dict[str, Any], count: int) -> list[int]:
    samples = report["samples"]
    if report["status"] != "measured" or len(samples) != count:
        raise ValueError("Gated MLA cache: missing measured samples")
    allocated = []
    for sample in samples:
        baseline, peak = sample["baseline"], sample["peak"]
        pairs = {
            "active": "active_bytes",
            "allocated": "allocated_bytes",
            "reserved": "reserved_bytes",
        }
        increments = {
            name: peak[field] - baseline[field] for name, field in pairs.items()
        }
        increments["sampled_device"] = (
            peak["sampled_device_used_bytes"] - baseline["device_used_bytes"]
        )
        if (
            increments != sample["incremental_peak_bytes"]
            or min(increments.values()) < 0
        ):
            raise ValueError("Gated MLA cache: incremental peak differs")
        if sample["released_allocated_bytes"] != baseline["allocated_bytes"]:
            raise ValueError("Gated MLA cache: allocation was not released")
        allocated.append(increments["allocated"])
    if report["median_incremental_allocated_peak_bytes"] != statistics.median(
        allocated
    ):
        raise ValueError("Gated MLA cache: median differs from samples")
    return allocated


def analyze_cache(packet: dict[str, Any]) -> dict[str, Any]:
    """Recalculate paired allocator peaks; this performs no device measurement."""
    records = packet["records"]
    if len(records) != packet["expected_cells"] or len(records) != 48:
        raise ValueError("Gated MLA cache: incomplete 48-cell packet")
    ratios = []
    count = packet["memory_policy"]["measured_repetitions"]
    for row in records:
        if row["status"] != "measured" or not row["parity"]["continuation"]["passed"]:
            raise ValueError(f"{row['case']['id']}: recorded continuation gate failed")
        compact = _allocation_samples(row["arms"]["compact"], count)
        materialized = _allocation_samples(row["arms"]["materialized"], count)
        ratio = statistics.median(materialized) / statistics.median(compact)
        if not math.isfinite(ratio):
            raise ValueError("Gated MLA cache: nonfinite ratio")
        ratios.append(ratio)
    return {
        "cells": len(records),
        "samples": len(records) * count * 2,
        "median_materialized_over_compact_allocated_peak": statistics.median(ratios),
        "ratio_range": [min(ratios), max(ratios)],
    }


def run() -> dict[str, Any]:
    """Analyze the supplied H200 packets on CPU without PyTorch or a GPU."""
    numerics = load_data("gated_mla_numerics")
    cache = load_data("gated_mla_cache")
    return {
        "experiment": "gated-mla",
        "status": "pass",
        "checks": 84,
        "numerics": analyze_numerics(numerics),
        "cache": analyze_cache(cache),
        "claim_boundaries": [
            numerics["provenance"]["claim_boundary"],
            cache["provenance"]["claim_boundary"],
        ],
    }
