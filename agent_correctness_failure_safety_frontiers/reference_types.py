"""Typed records used by the Chapter 32 deterministic reference model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """Observed tool event; payload is the bound request digest, not result text."""

    kind: str
    call: str
    tool: str
    payload: str
    request: str


@dataclass(frozen=True)
class Grant:
    """Trusted-issuer fixture; epoch lookup and commit are atomic in Service."""

    principal: str = "worker"
    resource: str = "studio-preview"
    revision: str = "r1"
    epoch: int = 4
    expires: int = 10
    actions: frozenset[str] = frozenset({"allocate", "query"})
    delegates: frozenset[str] = frozenset()

    def valid(self, epoch: int, now: int, principal: str, action: str) -> bool:
        return (
            self.epoch == epoch
            and now < self.expires
            and self.principal == principal
            and action in self.actions
            and self.resource == "studio-preview"
            and self.revision == "r1"
        )


@dataclass(frozen=True)
class History:
    """Retain pending-call state and its origin across history compaction."""

    pending: str | None
    origin: str


@dataclass(frozen=True)
class Claim:
    """Finite admission fixture; the argument itself is not a general proof oracle."""

    quantifier: str
    evidence_kind: str
    versions: tuple[str, ...]
    required_versions: tuple[str, ...]
    assumptions: tuple[str, ...]
    unresolved: tuple[str, ...] = ()
    retained_unknowns: tuple[str, ...] = ()
