"""Small, deterministic models for Chapter 32; not production agent defenses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import permutations
from math import isfinite

from .reference_verdicts import (
    classify_run as classify_run,
    fold as fold,
    verdict as verdict,
)
from .reference_types import (
    Claim,
    Event,
    Grant,
    History,
)


def non_amplifying(parent: Grant, child: Grant) -> bool:
    """Trusted lineage fixture; a different principal needs explicit permission."""
    principal_allowed = (
        child.principal == parent.principal or child.principal in parent.delegates
    )
    return (
        principal_allowed
        and child.delegates <= parent.delegates
        and child.actions <= parent.actions
        and child.expires <= parent.expires
        and child.epoch == parent.epoch
        and child.revision == parent.revision
        and child.resource == parent.resource
    )


class Service:
    """Sequential atomic operations model serialization against revocation."""

    def __init__(self) -> None:
        self.epoch = 4
        self.jobs: dict[str, tuple[str, int]] = {}
        self.attempts = 0

    def revoke(self) -> None:
        self.epoch += 1

    def commit(self, key: str, payload: str, grant: Grant, now: int = 1) -> int:
        """Validate current authority and key binding in one modeled atomic step."""
        self.attempts += 1
        if not key or not payload:
            raise ValueError("empty key/payload")
        if not grant.valid(self.epoch, now, "worker", "allocate"):
            raise PermissionError("authority invalid at commitment")
        if key in self.jobs:
            old_payload, job = self.jobs[key]
            if payload != old_payload:
                raise ValueError("same key with a different payload")
            return job
        job = len(self.jobs) + 1
        self.jobs[key] = (payload, job)
        return job


def disclosure(secret: int, expose: bool) -> str:
    """Return the observation exposed by the disclosure action."""
    if secret not in {0, 1}:
        raise ValueError("the finite domain has two secret bits")
    return str(secret) if expose else "preview checked"


def lossy_compaction(_history: History) -> tuple[str, ...]:
    """Return a deliberately insufficient history summary."""
    return ("preview not confirmed",)


def sufficient_compaction(history: History) -> tuple[str | None, str]:
    """Return the sufficient compacted history state."""
    return history.pending, history.origin


def compaction_collision(
    histories: Sequence[History],
    compact: Callable[[History], object],
) -> tuple[int, int] | None:
    """Exhaust the supplied finite domain for the fresh-allocation predicate."""
    for i, first in enumerate(histories):
        for j in range(i + 1, len(histories)):
            second = histories[j]
            same_summary = compact(first) == compact(second)
            different_permission = (first.pending is None) != (second.pending is None)
            if same_summary and different_permission:
                return i, j
    return None


def recovery_decision(status: str, actions: int, recoveries: int) -> str:
    """One invocation; final absence excludes later commitment of the old attempt."""
    if not (0 <= actions <= 12 and 0 <= recoveries <= 2 and recoveries <= actions):
        raise ValueError("invalid global counters")
    if status == "committed":
        return "use_receipt"
    if actions == 12:
        return "stop"
    if status == "absent_final":
        return "consider_fresh_admission"
    if status in {"unknown", "not_found_yet"}:
        return "query" if recoveries < 2 else "stop"
    raise ValueError("unknown receipt disposition")


def race_schedules(atomic: bool = False) -> dict[tuple[str, ...], int]:
    """Enumerate exactly the declared finite schedules, returning final counts."""
    results: dict[tuple[str, ...], int] = {}
    if atomic:
        for order in permutations(("a", "b")):
            count = 0
            for _worker in order:
                if count == 0:
                    count = 1
            results[order] = count
        return results
    for order in permutations(("read_a", "act_a", "read_b", "act_b")):
        if any(order.index("read_" + w) > order.index("act_" + w) for w in ("a", "b")):
            continue
        cached: dict[str, int] = {}
        count = 0
        for event in order:
            operation, worker = event.split("_")
            if operation == "read":
                cached[worker] = count
            elif cached[worker] < 1:
                count += 1
        results[order] = count
    return results


def reliability(steps: int, success: float) -> tuple[float, float]:
    """Return independent-step and union-bound reliability estimates."""
    if steps < 1 or not isfinite(success) or not 0 <= success <= 1:
        raise ValueError("positive horizon and probability required")
    return success**steps, min(1.0, steps * (1.0 - success))


def zero_harm_upper(cases: int, alpha: float = 0.05) -> float:
    """Return the upper zero-event probability bound."""
    if cases < 1 or not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("positive case count and significance in (0,1) required")
    return 1 - alpha ** (1 / cases)


def claim_admissible(claim: Claim) -> bool:
    """Return whether evidence supports the proposed claim."""
    required_kind = {
        "prefix": "trace",
        "finite_model": "exhaustive",
        "paired_runs": "paired",
        "population": "sample",
    }
    return (
        claim.quantifier in required_kind
        and claim.evidence_kind == required_kind[claim.quantifier]
        and bool(claim.required_versions)
        and set(claim.required_versions) <= set(claim.versions)
        and bool(claim.assumptions)
        and set(claim.unresolved) <= set(claim.retained_unknowns)
    )


def check_layers() -> dict[str, object]:
    """Return diagnostics for the layers invariant."""
    events = (
        Event("call", "c", "allocate", "r1", "q"),
        Event("result", "c", "allocate", "r1", "q"),
    )
    vector = classify_run((0, 1, 2), (True, False), True, events, True)
    return {
        "status": "pass"
        if vector
        == dict(
            syntax=True,
            protocol=True,
            authority=False,
            goal=True,
            invariant=False,
            progress=True,
        )
        else "fail",
        "vector": vector,
    }


def check_worlds() -> dict[str, object]:
    """Return diagnostics for the worlds invariant."""
    values = [
        verdict((0, 1), lambda n: n > 0),
        verdict((0, 1), lambda n: n <= 1),
        verdict((), lambda n: n <= 1),
    ]
    return {
        "status": "pass"
        if values == ["unknown", "supported", "inconsistent"]
        else "fail",
        "verdicts": values,
    }


def check_replay() -> dict[str, object]:
    """Return diagnostics for the replay invariant."""
    events = (
        Event("call", "c5", "allocate", "r1", "q"),
        Event("timeout", "c5", "allocate", "r1", "q"),
    )
    expected = (("c5",), ())
    return {
        "status": "pass" if fold(events) == fold(events) == expected else "fail",
        "open_calls": list(fold(events)[0]),
        "external_calls": 0,
    }


def check_authority() -> dict[str, object]:
    """Return diagnostics for the authority invariant."""
    service = Service()
    grant = Grant()
    service.revoke()
    rejected = False
    try:
        service.commit("q", "r1", grant)
    except PermissionError:
        rejected = True
    return {
        "status": "pass" if rejected and not service.jobs else "fail",
        "stale_grant_rejected": rejected,
        "committed_jobs": len(service.jobs),
    }


def check_information() -> dict[str, object]:
    """Return diagnostics for the information invariant."""
    bad = [disclosure(h, True) for h in (0, 1)]
    good = [disclosure(h, False) for h in (0, 1)]
    return {
        "status": "pass" if bad[0] != bad[1] and good[0] == good[1] else "fail",
        "disclosing_pair": bad,
        "constant_pair": good,
    }


def check_compaction() -> dict[str, object]:
    """Return diagnostics for the compaction invariant."""
    histories = (History("q", "service-timeout"), History(None, "initial-state"))
    collision = compaction_collision(histories, lossy_compaction)
    repaired = compaction_collision(histories, sufficient_compaction)
    return {
        "status": "pass" if collision == (0, 1) and repaired is None else "fail",
        "collision": list(collision or ()),
        "repaired_collision": repaired,
    }


def check_reconciliation() -> dict[str, object]:
    """Return diagnostics for the reconciliation invariant."""
    service = Service()
    first = service.commit("q", "r1", Grant())
    second = service.commit("q", "r1", Grant())
    decisions = [
        recovery_decision("unknown", 5, 0),
        recovery_decision("unknown", 6, 1),
        recovery_decision("unknown", 7, 2),
    ]
    ok = (
        first == second
        and len(service.jobs) == 1
        and decisions == ["query", "query", "stop"]
    )
    return {
        "status": "pass" if ok else "fail",
        "jobs": len(service.jobs),
        "physical_attempts": service.attempts,
        "recovery_decisions": decisions,
    }


def check_swarm() -> dict[str, object]:
    """Return diagnostics for the swarm invariant."""
    unsafe = race_schedules()
    atomic = race_schedules(True)
    failures = sum(count > 1 for count in unsafe.values())
    ok = (
        len(unsafe) == 6
        and failures == 4
        and len(atomic) == 2
        and set(atomic.values()) == {1}
    )
    return {
        "status": "pass" if ok else "fail",
        "split_schedules": len(unsafe),
        "violating_schedules": failures,
        "atomic_schedules": len(atomic),
        "atomic_violations": sum(count > 1 for count in atomic.values()),
    }


def check_arithmetic() -> dict[str, object]:
    """Return diagnostics for the arithmetic invariant."""
    chain, bound = reliability(100, 0.99)
    upper = zero_harm_upper(20)
    ok = (
        abs(chain - 0.3660323412732292) < 1e-12
        and bound == 1
        and 0.13910 < upper < 0.13912
    )
    return {
        "status": "pass" if ok else "fail",
        "independent_chain": chain,
        "union_failure_bound": bound,
        "independent_redundant_failure": 0.1**2,
        "identical_redundant_failure": 0.1,
        "zero_of_twenty_upper": upper,
    }


def check_assurance() -> dict[str, object]:
    """Return diagnostics for the assurance invariant."""
    args = dict(
        versions=("q31.1", "allocator-v1"),
        required_versions=("q31.1", "allocator-v1"),
        assumptions=("complete covered-effect model",),
        unresolved=("q",),
        retained_unknowns=("q",),
    )
    narrow = claim_admissible(Claim("prefix", "trace", **args))
    broad = claim_admissible(Claim("all_future_attacks", "trace", **args))
    return {
        "status": "pass" if narrow and not broad else "fail",
        "bounded_claim_admitted": narrow,
        "all_future_claim_admitted": broad,
    }


def all_checks() -> dict[str, object]:
    """Run and return every chapter invariant check."""
    return dict(
        layers=check_layers(),
        worlds=check_worlds(),
        replay=check_replay(),
        authority=check_authority(),
        information=check_information(),
        compaction=check_compaction(),
        reconciliation=check_reconciliation(),
        swarm=check_swarm(),
        arithmetic=check_arithmetic(),
        assurance=check_assurance(),
    )
