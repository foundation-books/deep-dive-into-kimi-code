"""Raw-result projection helpers for the Chapter 28 fixture."""

from __future__ import annotations

from typing import Any

from .model import canonical_json, digest


def project_result(
    raw: dict[str, Any], *, budget: int, raw_reference: str, policy_version: str
) -> dict[str, Any]:
    """Produce a typed bounded projection without changing semantic status."""

    if budget < 128:
        raise ValueError("projection budget must leave room for provenance")
    diagnostics = list(raw.get("diagnostics", []))
    retained: list[str] = []
    projection = {
        "executor_status": raw["executor_status"],
        "domain_status": raw["domain_status"],
        "blocking_ids": list(raw.get("blocking_ids", [])),
        "diagnostics": retained,
        "omitted_diagnostics": len(diagnostics),
        "truncated": bool(diagnostics),
        "raw_reference": raw_reference,
        "raw_digest": digest(raw),
        "projection_policy": policy_version,
    }
    if len(canonical_json(projection).encode("utf-8")) > budget:
        raise ValueError("projection budget is smaller than the provenance envelope")
    for diagnostic in diagnostics:
        candidate = {
            **projection,
            "diagnostics": retained + [diagnostic],
            "omitted_diagnostics": len(diagnostics) - len(retained) - 1,
            "truncated": len(retained) + 1 < len(diagnostics),
        }
        if len(canonical_json(candidate).encode("utf-8")) > budget:
            break
        retained.append(diagnostic)
        projection = candidate
    return projection
