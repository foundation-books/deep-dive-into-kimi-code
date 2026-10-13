"""Book-created repair-ticket black box with bounded investigation.

This is a behavioral teaching fixture, not a secure sandbox for untrusted code.
The oracle session exposes only calls and public diagnostics to its caller.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Ticket:
    """One repair request's public state in the fictional service."""

    reference: str
    device: str
    team: str | None = None
    closed: bool = False


class TicketService:
    """Reference semantics, also usable for controlled wrong candidates."""

    def __init__(
        self,
        *,
        reuse_identical: bool = True,
        require_assignment: bool = True,
        prefix: str = "T",
    ) -> None:
        self.reuse_identical = reuse_identical
        self.require_assignment = require_assignment
        self.prefix = prefix
        self.tickets: dict[str, Ticket] = {}
        self.references: dict[str, str] = {}

    def create(self, reference: str, device: str) -> dict[str, Any]:
        if (
            not isinstance(reference, str)
            or not reference
            or not isinstance(device, str)
            or not device
        ):
            return {"status": "invalid_argument"}
        old_handle = self.references.get(reference)
        if old_handle is not None:
            old = self.tickets[old_handle]
            if old.device != device:
                return {"status": "reference_conflict"}
            if self.reuse_identical:
                return {"status": "existing", "ticket": old_handle}
        handle = f"{self.prefix}{len(self.tickets) + 1}"
        self.tickets[handle] = Ticket(reference, device)
        self.references[reference] = handle
        return {"status": "created", "ticket": handle}

    def assign(self, handle: str, team: str) -> dict[str, Any]:
        if not isinstance(handle, str) or not handle:
            return {"status": "unknown_ticket"}
        ticket = self.tickets.get(handle)
        if ticket is None:
            return {"status": "unknown_ticket"}
        if not isinstance(team, str) or not team:
            return {"status": "invalid_argument"}
        if ticket.closed:
            return {"status": "already_closed"}
        ticket.team = team
        return {"status": "assigned"}

    def close(self, handle: str) -> dict[str, Any]:
        if not isinstance(handle, str) or not handle:
            return {"status": "unknown_ticket"}
        ticket = self.tickets.get(handle)
        if ticket is None:
            return {"status": "unknown_ticket"}
        if ticket.closed:
            return {"status": "already_closed"}
        if self.require_assignment and ticket.team is None:
            return {"status": "assignment_required"}
        ticket.closed = True
        return {"status": "closed"}

    def call(self, operation: str, *arguments: str) -> dict[str, Any]:
        if operation == "create" and len(arguments) == 2:
            return self.create(*arguments)
        if operation == "assign" and len(arguments) == 2:
            return self.assign(*arguments)
        if operation == "close" and len(arguments) == 1:
            return self.close(*arguments)
        return {"status": "invalid_call"}


def _run_case(
    factory: Callable[[], TicketService], case: str, namespace: str
) -> tuple[object, ...]:
    """Compare public behavior while allowing unrelated opaque handle formats."""
    service = factory()
    outputs: list[object] = []

    def create(reference: str, device: str) -> str:
        result = service.create(f"{namespace}-{reference}", device)
        outputs.append((result["status"], result.get("ticket") is not None))
        return str(result.get("ticket", "missing"))

    if case == "normal":
        handle = create("normal", "camera")
        outputs.append(service.assign(handle, "repair"))
        outputs.append(service.close(handle))
    elif case == "duplicate":
        first = create("duplicate", "camera")
        second = create("duplicate", "camera")
        outputs.append(first == second)
    elif case == "unassigned":
        handle = create("unassigned", "lens")
        outputs.append(service.close(handle))
        outputs.append(service.assign(handle, "repair"))
        outputs.append(service.close(handle))
    elif case == "conflict":
        handle = create("conflict", "camera")
        outputs.append(service.create(f"{namespace}-conflict", "projector"))
        same_again = create("conflict", "camera")
        outputs.append(handle == same_again)
    elif case == "two_tickets":
        first = create("ticket-a", "camera")
        second = create("ticket-b", "lens")
        outputs.append(service.assign(first, "repair"))
        outputs.append(service.close(first))
        outputs.append(service.assign(second, "repair"))
        outputs.append(service.close(second))
    elif case == "repeat_close":
        handle = create("repeat-close", "camera")
        outputs.append(service.assign(handle, "repair"))
        outputs.append(service.close(handle))
        outputs.append(service.close(handle))
    elif case == "invalid_handle":
        outputs.append(service.assign("missing", "repair"))
        outputs.append(service.close("missing"))
    else:
        raise ValueError(case)
    return tuple(outputs)


PUBLIC_CASES = ("normal", "unassigned")
FINAL_CASES = (
    "normal",
    "duplicate",
    "unassigned",
    "conflict",
    "two_tickets",
    "repeat_close",
    "invalid_handle",
)


def evaluate_candidate(
    factory: Callable[[], TicketService], cases: tuple[str, ...] = FINAL_CASES
) -> dict[str, Any]:
    """Independent author-side evaluator; final cases never enter an attempt."""
    results = {
        case: all(
            _run_case(factory, case, namespace)
            == _run_case(TicketService, case, namespace)
            for namespace in ("fresh", "heldout")
        )
        for case in cases
    }
    return {"passed": all(results.values()), "cases": results}


class OracleSession:
    """Reference calls, bounded diagnostics, and final author-side grading."""

    QUERY_LIMIT = 8
    PUBLIC_LIMIT = 2
    FINAL_LIMIT = 1

    def __init__(self) -> None:
        self.__reference = TicketService()
        self.oracle_operations = 0
        self.public_submissions = 0
        self.final_submissions = 0
        self.finalized = False

    def query(self, operation: str, *arguments: str) -> dict[str, Any]:
        if self.finalized:
            return {"status": "attempt_finalized"}
        if self.oracle_operations >= self.QUERY_LIMIT:
            return {"status": "oracle_budget_exhausted"}
        self.oracle_operations += 1
        return self.__reference.call(operation, *arguments)

    def public_diagnostic(self, factory: Callable[[], TicketService]) -> dict[str, Any]:
        if self.finalized:
            return {"status": "attempt_finalized"}
        if self.public_submissions >= self.PUBLIC_LIMIT:
            return {"status": "public_budget_exhausted"}
        self.public_submissions += 1
        result = evaluate_candidate(factory, PUBLIC_CASES)
        if result["passed"]:
            return {"status": "public_pass", "scope": "two public cases only"}
        return {
            "status": "public_fail",
            "diagnostic": "A public ticket transition differs; inspect close preconditions.",
        }

    def final_submit(self, factory: Callable[[], TicketService]) -> dict[str, Any]:
        if self.final_submissions >= self.FINAL_LIMIT:
            return {"status": "final_budget_exhausted"}
        self.final_submissions += 1
        self.finalized = True
        # The learner receives no protected case-by-case feedback in this attempt.
        self.__final_result = evaluate_candidate(factory)
        return {"status": "submitted"}

    def author_result(self) -> dict[str, Any]:
        if not self.finalized:
            return {"status": "unfinished"}
        return deepcopy(self.__final_result)


class OracleClient:
    """The declared attempt interface; author_result is absent from it."""

    def __init__(self, session: OracleSession) -> None:
        self.__session = session

    def query(self, operation: str, *arguments: str) -> dict[str, Any]:
        return self.__session.query(operation, *arguments)

    def public_diagnostic(self, factory: Callable[[], TicketService]) -> dict[str, Any]:
        return self.__session.public_diagnostic(factory)

    def final_submit(self, factory: Callable[[], TicketService]) -> dict[str, Any]:
        return self.__session.final_submit(factory)


def _record_query(
    client: OracleClient,
    session: OracleSession,
    trace: list[dict[str, Any]],
    operation: str,
    *arguments: str,
) -> dict[str, Any]:
    result = client.query(operation, *arguments)
    trace.append(
        {
            "operation": operation,
            "arguments": arguments,
            "output": result,
            "cumulative": session.oracle_operations,
        }
    )
    return result


def _before_probe_candidate() -> TicketService:
    return TicketService(reuse_identical=False, require_assignment=False)


def _after_duplicate_candidate() -> TicketService:
    return TicketService(reuse_identical=True, require_assignment=False)


def _final_candidate() -> TicketService:
    return TicketService(prefix="C")


def investigation_record() -> dict[str, Any]:
    """Seven checked oracle calls, two candidate revisions, one final check."""
    session = OracleSession()
    client = OracleClient(session)
    trace: list[dict[str, Any]] = []

    ordinary = _record_query(client, session, trace, "create", "A", "camera")
    _record_query(client, session, trace, "assign", str(ordinary["ticket"]), "repair")
    _record_query(client, session, trace, "close", str(ordinary["ticket"]))
    first = _record_query(client, session, trace, "create", "B", "camera")
    duplicate = _record_query(client, session, trace, "create", "B", "camera")

    diagnostic = client.public_diagnostic(_after_duplicate_candidate)
    new_ticket = _record_query(client, session, trace, "create", "C", "lens")
    unassigned = _record_query(
        client, session, trace, "close", str(new_ticket["ticket"])
    )

    public_pass = client.public_diagnostic(_final_candidate)
    submission = client.final_submit(_final_candidate)
    return {
        "trace": trace,
        "ordinary_count": 3,
        "duplicate_handles_equal": first["ticket"] == duplicate["ticket"],
        "unassigned_status": unassigned["status"],
        "initial_candidate_passes": evaluate_candidate(_before_probe_candidate)[
            "passed"
        ],
        "after_duplicate_diagnostic": diagnostic,
        "final_public": public_pass,
        "submission": submission,
        "final_result": session.author_result(),
        "oracle_operations": session.oracle_operations,
        "public_submissions": session.public_submissions,
        "final_submissions": session.final_submissions,
    }
