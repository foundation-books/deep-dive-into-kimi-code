"""Deterministic, book-created changing-obligation teaching world.

The policy only receives values returned by ``initial_observation`` and
``step``.  The evaluator deliberately has a separate, author-side view.
Times are integer simulated minutes from Day 1 midnight.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


DAY = 24 * 60
START = 9 * 60
END = 2 * DAY + 16 * 60
DATE_AVAILABLE = DAY + 8 * 60 + 30
MANAGER_AVAILABLE = DAY + 9 * 60 + 15
GRANT_AVAILABLE = DAY + 11 * 60 + 5
WINDOW = 4 * 60
MAX_ACTIONS = 80


@dataclass(order=True)
class Event:
    """Scheduled transition ordered by simulated time and tie breaker."""

    at: int
    order: int
    name: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)


class OnboardingWorld:
    """Four fictional applications, an ordered clock, and independent checks."""

    def __init__(self) -> None:
        self.now = START
        self.actions = 0
        self.submitted = False
        self.current_start_day = 4
        self.current_manager = "Mira"
        self.available: set[str] = set()
        self.consumed: dict[str, int] = {}
        self.calendar: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.grants: list[dict[str, Any]] = []
        self.notices: list[dict[str, Any]] = []
        self.protected = {"employee": "Noor", "start_day": 5}
        self.initial_valid: bool | None = None
        self.deadlines: dict[str, bool | None] = {
            "date": None,
            "manager": None,
            "grant": None,
        }
        self.events = [
            Event(DAY + 8 * 60, 0, "date_occurs"),
            Event(DATE_AVAILABLE, 0, "date_available"),
            Event(DAY + 9 * 60, 0, "manager_occurs"),
            Event(MANAGER_AVAILABLE, 0, "manager_available"),
            Event(GRANT_AVAILABLE, 0, "grant_available"),
            Event(DATE_AVAILABLE + WINDOW, 1, "date_deadline"),
            Event(MANAGER_AVAILABLE + WINDOW, 1, "manager_deadline"),
            Event(GRANT_AVAILABLE + WINDOW, 1, "grant_deadline"),
        ]
        self.history: list[dict[str, Any]] = []
        self.milestones: list[dict[str, Any]] = []

    def mark(self, label: str) -> None:
        """Save the author-side state used for printed tables and timelines."""
        self.milestones.append(
            {
                "label": label,
                "time": self.now,
                "calendar": deepcopy(self.calendar),
                "tasks": deepcopy(self.tasks),
                "requests": deepcopy(self.requests),
                "grants": deepcopy(self.grants),
                "available": sorted(self.available),
                "consumed": deepcopy(self.consumed),
            }
        )

    def initial_observation(self) -> dict[str, Any]:
        return {
            "time": self.now,
            "brief": "Coordinate Ada's Day 4 onboarding, orientation at 10:00, manager Mira.",
            "window_end": END,
            "response_window_minutes": WINDOW,
            "authority": "Revise Ada's objects and send coordination notices; leave other employees unchanged.",
            "provisioning": "Acceptance means pending; requests cannot be amended, but resulting grants can be reconciled.",
        }

    def _one(self, objects: list[dict[str, Any]]) -> dict[str, Any] | None:
        return objects[0] if len(objects) == 1 else None

    def _date_correct(self) -> bool:
        calendar, task = self._one(self.calendar), self._one(self.tasks)
        return bool(calendar and task and calendar["day"] == 6 and task["day"] == 6)

    def _manager_correct(self) -> bool:
        calendar, task = self._one(self.calendar), self._one(self.tasks)
        return bool(
            calendar
            and task
            and calendar["manager"] == "Tomas"
            and task["manager"] == "Tomas"
        )

    def _grant_correct(self) -> bool:
        grant = self._one(self.grants)
        return bool(grant and grant["day"] == 6 and not grant["activated_early"])

    def _apply_event(self, event: Event) -> None:
        name = event.name
        if name == "date_occurs":
            calendar, task, request = (
                self._one(self.calendar),
                self._one(self.tasks),
                self._one(self.requests),
            )
            self.initial_valid = bool(
                calendar
                and task
                and request
                and calendar["day"] == task["day"] == request["day"] == 4
                and calendar["manager"] == task["manager"] == "Mira"
            )
            self.current_start_day = 6
        elif name == "date_available":
            self.available.add("date")
        elif name == "manager_occurs":
            self.current_manager = "Tomas"
        elif name == "manager_available":
            self.available.add("manager")
        elif name == "request_completes":
            request = self.requests[int(event.payload["index"])]
            if request["status"] == "pending":
                request["status"] = "complete"
                self.grants.append(
                    {
                        "day": request["day"],
                        "manager": request["manager"],
                        "activated_early": False,
                        "request_index": event.payload["index"],
                    }
                )
        elif name == "grant_available":
            self.available.add("grant")
        elif name == "date_deadline":
            self.deadlines["date"] = self._date_correct()
        elif name == "manager_deadline":
            self.deadlines["manager"] = self._manager_correct()
        elif name == "grant_deadline":
            self.deadlines["grant"] = self._grant_correct()
        self.history.append({"time": self.now, "event": name})

    def advance_to(self, target: int) -> None:
        """Apply every due event in (time, order) order before the next action."""
        if not self.now <= target <= END:
            raise ValueError("wait target outside the open coordination window")
        while True:
            due = sorted(event for event in self.events if event.at <= target)
            if not due:
                break
            event = due[0]
            self.events.remove(event)
            self.now = event.at
            self._apply_event(event)
        self.now = target

    def snapshot(self) -> OnboardingWorld:
        return deepcopy(self)

    def restore(self, saved: OnboardingWorld) -> None:
        """Restore world/queue/cursor while retaining the attempt's spent budget."""
        spent = self.actions
        self.__dict__ = deepcopy(saved.__dict__)
        self.actions = max(spent, saved.actions)

    def _read_messages(self, at: int) -> list[dict[str, Any]]:
        messages = []
        updates = (
            ("date", "Ada starts Day 6.", DAY + 8 * 60, DATE_AVAILABLE, 1),
            (
                "manager",
                "Tomas replaces Mira as manager.",
                DAY + 9 * 60,
                MANAGER_AVAILABLE,
                2,
            ),
        )
        for key, body, recorded, available, sequence in updates:
            if key in self.available:
                messages.append(
                    {
                        "update": key,
                        "body": body,
                        "source": "PeopleOps",
                        "sequence": sequence,
                        "recorded_at": recorded,
                        "available_at": available,
                    }
                )
                self.consumed.setdefault(key, at)
        return messages

    def _write(
        self, operation: str, arguments: dict[str, Any], at: int
    ) -> dict[str, Any]:
        writer_name = self.WRITE_OPERATIONS.get(operation)
        if writer_name is None:
            return {"status": "unknown_operation"}
        return getattr(self, writer_name)(arguments, at)

    WRITE_OPERATIONS = {
        "create_calendar": "_create_calendar",
        "create_task": "_create_task",
        "revise_calendar": "_revise_calendar",
        "revise_task": "_revise_task",
        "request_provision": "_request_provision",
        "reconcile_grant": "_reconcile_grant",
        "send_notice": "_send_notice",
        "edit_unrelated": "_edit_unrelated",
    }

    def _create_calendar(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        self.calendar.append(
            {
                "day": int(arguments["day"]),
                "hour": 10,
                "manager": str(arguments["manager"]),
            }
        )
        return {"status": "ok"}

    def _create_task(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        self.tasks.append(
            {"day": int(arguments["day"]), "manager": str(arguments["manager"])}
        )
        return {"status": "ok"}

    def _revise_calendar(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        return self._revise(self.calendar, "missing_calendar", arguments)

    def _revise_task(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        return self._revise(self.tasks, "missing_task", arguments)

    @staticmethod
    def _revise(
        objects: list[dict[str, Any]],
        missing_status: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if not objects:
            return {"status": missing_status}
        objects[0].update(
            {key: arguments[key] for key in ("day", "manager") if key in arguments}
        )
        return {"status": "ok"}

    def _request_provision(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        if self.now >= DAY + 11 * 60:
            return {"status": "request_too_late"}
        index = len(self.requests)
        self.requests.append(
            {
                "day": int(arguments["day"]),
                "manager": str(arguments["manager"]),
                "status": "pending",
            }
        )
        self.events.append(
            Event(DAY + 11 * 60, index, "request_completes", {"index": index})
        )
        return {"status": "accepted_pending", "request": index}

    def _reconcile_grant(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        if "grant" not in self.available or not self.grants:
            return {"status": "grant_unavailable"}
        self.grants[0]["day"] = int(arguments["day"])
        self.grants[0]["manager"] = str(arguments["manager"])
        return {"status": "ok"}

    def _send_notice(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        self.notices.append(
            {
                "day": int(arguments["day"]),
                "manager": str(arguments["manager"]),
                "recipient": str(arguments["recipient"]),
                "time": at,
            }
        )
        return {"status": "ok"}

    def _edit_unrelated(self, arguments: dict[str, Any], at: int) -> dict[str, Any]:
        self.protected["start_day"] = int(arguments["day"])
        return {"status": "ok"}

    def _dispatch_step_operation(
        self, operation: str, arguments: dict[str, Any], at: int
    ) -> dict[str, Any]:
        if operation == "wait":
            self.advance_to(int(arguments["until"]))
            return {"status": "ok"}
        if operation == "read_messages":
            return {"status": "ok", "messages": self._read_messages(at)}
        if operation == "inspect_task":
            visible_status = "pending" if "grant" not in self.available else "complete"
            return {
                "status": "ok",
                "requests": [visible_status for _ in self.requests],
                "tasks": deepcopy(self.tasks),
            }
        if operation == "inspect_access":
            grants = deepcopy(self.grants) if "grant" in self.available else []
            return {"status": "ok", "grants": grants}
        if operation == "finalize":
            if self.now < END:
                return {"status": "early_finalization_rejected"}
            self.submitted = True
            return {"status": "ok"}
        return self._write(operation, arguments, at)

    def step(self, operation: str, **arguments: Any) -> dict[str, Any]:
        if self.submitted or self.actions >= MAX_ACTIONS:
            return {"status": "budget_exhausted" if not self.submitted else "terminal"}
        if self.now == END and operation != "finalize":
            return {"status": "window_closed"}
        at = self.now
        self.actions += 1
        reply = self._dispatch_step_operation(operation, arguments, at)
        self.history.append(
            {
                "time": at,
                "index": self.actions,
                "action": operation,
                "reply": deepcopy(reply),
            }
        )
        if operation not in ("wait", "finalize") and self.now < END:
            self.advance_to(min(self.now + 5, END))
        reply["time"] = at
        return reply

    def evaluate(self) -> dict[str, Any]:
        """Author-side test of state, timing, historical validity, and damage."""
        if self.now < END:
            return {"status": "unfinished"}
        calendar = self._one(self.calendar)
        task = self._one(self.tasks)
        grant = self._one(self.grants)
        notice = self.notices[-1] if self.notices else None
        checks = {
            "orientation": bool(
                calendar and calendar == {"day": 6, "hour": 10, "manager": "Tomas"}
            ),
            "task": bool(task and task == {"day": 6, "manager": "Tomas"}),
            "grant": bool(grant and self._grant_correct() and len(self.requests) == 1),
            "notice": bool(
                notice
                and notice["day"] == 6
                and notice["manager"] == "Tomas"
                and notice["recipient"] == "Tomas"
            ),
            "timely": all(value is True for value in self.deadlines.values()),
            "protected": self.protected == {"employee": "Noor", "start_day": 5},
            "initially_valid": self.initial_valid is True,
            "submitted": self.submitted,
        }
        return {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "deadlines": deepcopy(self.deadlines),
            "actions": self.actions,
        }


def scripted_attempt(strategy: str = "incremental") -> OnboardingWorld:
    """Build a named deterministic scenario for chapter assets and audits."""
    from .onboarding_scenarios import scripted_attempt as build_attempt

    return build_attempt(strategy)
