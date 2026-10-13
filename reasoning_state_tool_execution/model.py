"""Typed records for the Chapter 28 execution-state teaching oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Return the chapter's deterministic JSON spelling for finite fixtures."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    """Digest a JSON-compatible fixture value; this is identity, not truth."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Event:
    """Record an immutable execution event with provenance and parent links."""

    event_id: str
    kind: str
    subject: str
    payload: dict[str, Any]
    parents: tuple[str, ...] = ()
    observed_at: int = 0
    source: str = "ch27-fixture"


@dataclass(frozen=True)
class RegistryEntry:
    """Describe a versioned tool interface, resources, and effect class."""

    name: str
    interface_version: str
    executable: str
    schema_digest: str
    adapter_version: str
    capabilities: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    effect_class: str


@dataclass(frozen=True)
class CanonicalCall:
    """Bind normalized call arguments to registry and environment digests."""

    call_id: str
    name: str
    interface_version: str
    arguments: dict[str, Any]
    argument_digest: str
    registry_digest: str
    environment_digest: str
    effect_key: str | None = None


@dataclass
class Attempt:
    """Track one numbered executor attempt and its optional receipt."""

    attempt_id: str
    call_id: str
    number: int
    executor: str
    status: str
    receipt: str | None = None


@dataclass(frozen=True)
class Observation:
    """Bind raw and projected executor results to one call attempt."""

    observation_id: str
    call_id: str
    attempt_id: str
    executable: str
    environment_digest: str
    executor_status: str
    domain_status: str
    raw: dict[str, Any]
    raw_digest: str
    projection: dict[str, Any]
    projection_digest: str
    recovered: bool = False


@dataclass
class ExecutionState:
    """Accumulate replayable calls, attempts, observations, and diagnostics."""

    history: list[dict[str, Any]] = field(default_factory=list)
    registry: dict[str, RegistryEntry] = field(default_factory=dict)
    registry_digest: str | None = None
    calls: dict[str, CanonicalCall] = field(default_factory=dict)
    attempts: dict[str, Attempt] = field(default_factory=dict)
    observations: dict[str, Observation] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def canonical_value(self) -> dict[str, Any]:
        """Return a map-order-independent representation used by replay checks."""

        return {
            "history": self.history,
            "registry": {
                key: asdict(self.registry[key]) for key in sorted(self.registry)
            },
            "registry_digest": self.registry_digest,
            "calls": {key: asdict(self.calls[key]) for key in sorted(self.calls)},
            "attempts": {
                key: asdict(self.attempts[key]) for key in sorted(self.attempts)
            },
            "observations": {
                key: asdict(self.observations[key]) for key in sorted(self.observations)
            },
            "environment": self.environment,
            "diagnostics": self.diagnostics,
            "pending": self.pending,
        }

    def state_digest(self) -> str:
        return digest(self.canonical_value())
