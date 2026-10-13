"""Repository-state identity helpers for the Chapter 29 oracle."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json


def canonical_digest(value: object) -> str:
    """Return a stable digest for JSON-shaped fixture data."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def repository_fingerprint(state: Mapping[str, object]) -> str:
    """Bind identity, base, branch, dirty state, and environment."""

    required = {
        "repository",
        "base",
        "branch",
        "tracked_diff",
        "untracked",
        "environment",
    }
    if set(state) != required:
        raise ValueError(f"repository state must contain exactly {sorted(required)}")
    return canonical_digest(state)


__all__ = ["canonical_digest", "repository_fingerprint"]
