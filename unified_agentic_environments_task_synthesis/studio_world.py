"""Small executable studio world for the Chapter 24 teaching example.

This book-authored fixture borrows no AppWorld task data or Kimi trainer code.
Its six operations and geometry are the chapter's declared canonical interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Sequence, cast


CANONICAL_ACTIONS = (
    "SearchMessages",
    "OpenAsset",
    "InspectBoxes",
    "EditReport",
    "AttachEvidenceAndClose",
    "Submit",
)
STUDIO_BOXES = {"red": (20, 20, 40, 40), "blue": (20, 80, 40, 40)}
STUDIO_ASSET_ID = "studio-image-v1"
# Symbolic source bytes for this finite geometry fixture, not a released image.
STUDIO_SOURCE_BYTES = b"studio-ops-001|160x160|red:20,20,40,40|blue:20,80,40,40"
STUDIO_SOURCE_DIGEST = sha256(STUDIO_SOURCE_BYTES).hexdigest()
COORDINATE_CONVENTION = "image y increases downward"
CLARIFICATION = "current asset only; future batch awaiting separate audit"
BASE_TRACE = CANONICAL_ACTIONS
ALTERNATE_TRACE = (
    "OpenAsset",
    "SearchMessages",
    *CANONICAL_ACTIONS[2:],
)
HARNESS_TO_INSPECTION = {
    "studio_flat": ("inspect_boxes", "ids"),
    "studio_namespaced": ("assets.inspect_bboxes", "shape_ids"),
}


def adapt_inspection(
    tool: str, arguments: Mapping[str, object]
) -> tuple[str, tuple[str, ...]]:
    """Adapt either declared batch spelling before any environment action."""

    fields = {"inspect_boxes": "ids", "assets.inspect_bboxes": "shape_ids"}
    if tool not in fields:
        raise ValueError("unknown inspection tool")
    identifiers = arguments.get(fields[tool])
    if not isinstance(identifiers, (list, tuple)) or not identifiers:
        raise ValueError("inspection requires a nonempty shape identifier list")
    if not all(isinstance(item, str) for item in identifiers):
        raise ValueError("shape identifiers must be strings")
    return "InspectBoxes", tuple(identifiers)


def inspect_boxes(identifiers: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Return boxes from the unchanged symbolic image in one batch action."""

    return {identifier: STUDIO_BOXES[identifier] for identifier in identifiers}


def box_center(box: Sequence[int]) -> tuple[int, int]:
    """Return the integer center of an ``(x, y, width, height)`` box."""

    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise ValueError("a box needs four fields and positive dimensions")
    x, y, width, height = box
    return x + width // 2, y + height // 2


@dataclass(frozen=True)
class StudioTask:
    """One admitted task package and its learner-visible initial boundary."""

    task_id: str = "studio-ops-001"
    concepts: tuple[str, str] = ("vertical order", "evidence-backed closure")
    materials: tuple[str, ...] = (
        "pinned image",
        "client brief",
        "coordinate convention",
        "audit template",
    )
    asset_id: str = STUDIO_ASSET_ID
    initial_relation: str = "blue above red"
    action_limit: int = 6
    allowed_actions: frozenset[str] = frozenset(CANONICAL_ACTIONS)

    def initial_observation(self) -> dict[str, object]:
        """Expose the request, capabilities and asset id, but no answer key."""

        return {
            "request": "Correct the studio report, attach grounded evidence, close and submit",
            "asset_id": self.asset_id,
            "capabilities": tuple(
                action for action in CANONICAL_ACTIONS if action in self.allowed_actions
            ),
        }


@dataclass
class StudioWorld:
    """Mutable state within one episode; a new instance starts a fresh attempt."""

    task: StudioTask = field(default_factory=StudioTask)
    source_image: bytes = STUDIO_SOURCE_BYTES
    report_relation: str = "blue above red"
    report_scope: str = "batch scope unspecified"
    ticket_closed: bool = False
    evidence: dict[str, tuple[int, int]] = field(default_factory=dict)
    evidence_source: str | None = None
    evidence_convention: str | None = None
    messages_read: bool = False
    asset_open: bool = False
    inspected: dict[str, tuple[int, ...]] = field(default_factory=dict)
    submissions: int = 0
    final_sentence: str = ""
    events: list[str] = field(default_factory=list)

    def _edit_report(self, arguments: Mapping[str, object]) -> dict[str, object]:
        if not self.messages_read or not self.inspected:
            raise ValueError("brief and observed boxes are required before editing")
        relation = arguments.get("relation")
        if not isinstance(relation, str):
            raise ValueError("report relation is required")
        self.report_relation = relation
        scope = arguments.get("scope_note")
        if scope is not None:
            if not isinstance(scope, str):
                raise ValueError("scope note must be text")
            self.report_scope = scope
        return {
            "report_relation": self.report_relation,
            "scope_note": self.report_scope,
        }

    def _attach_evidence(self, arguments: Mapping[str, object]) -> dict[str, object]:
        centers = arguments.get("centers")
        if not isinstance(centers, dict) or set(centers) != {"red", "blue"}:
            raise ValueError("both observed centers are required")
        if not all(
            isinstance(value, tuple) and len(value) == 2 for value in centers.values()
        ):
            raise ValueError("centers must be coordinate pairs")
        self.evidence = dict(centers)
        self.evidence_source = str(arguments.get("asset_id"))
        self.evidence_convention = str(arguments.get("coordinate_convention"))
        self.ticket_closed = True
        return {"evidence_attached": True, "ticket_closed": True}

    def execute(
        self, action: str, payload: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Perform one legal canonical action and return its raw observation."""

        arguments = payload or {}
        if action not in self.task.allowed_actions:
            raise ValueError(f"unavailable action: {action}")
        if self.submissions or len(self.events) >= self.task.action_limit:
            raise ValueError("attempt is terminal or out of actions")
        if action == "SearchMessages":
            self.messages_read = True
            observation: dict[str, object] = {
                "brief": "Check the current image and correct the audit",
                "coordinate_convention": COORDINATE_CONVENTION,
            }
        elif action == "OpenAsset":
            if arguments.get("asset_id") != self.task.asset_id:
                raise ValueError("unknown pinned asset")
            self.asset_open = True
            observation = {
                "asset_id": self.task.asset_id,
                "source_digest": STUDIO_SOURCE_DIGEST,
            }
        elif action == "InspectBoxes":
            if not self.asset_open:
                raise ValueError("asset must be open before inspection")
            identifiers = arguments.get("ids")
            if identifiers != ("red", "blue"):
                raise ValueError("this task requires one red-and-blue batch inspection")
            self.inspected = inspect_boxes(("red", "blue"))
            observation = {
                "asset_id": self.task.asset_id,
                "boxes": dict(self.inspected),
            }
        elif action == "EditReport":
            observation = self._edit_report(arguments)
        elif action == "AttachEvidenceAndClose":
            observation = self._attach_evidence(arguments)
        elif action == "Submit":
            self.submissions += 1
            self.final_sentence = str(arguments.get("final_sentence", ""))
            observation = {"submitted": True}
        else:
            raise ValueError(f"unknown canonical action: {action}")
        self.events.append(action)
        return observation

    def verdict(self) -> dict[str, bool]:
        """Check outcome and retained evidence without requiring one exact path."""

        observed_centers = {
            key: box_center(value) for key, value in self.inspected.items()
        }
        checks = {
            "report_correct": self.report_relation == "blue below red",
            "centers_grounded": self.evidence == observed_centers
            and set(self.evidence) == {"red", "blue"},
            "source_grounded": self.evidence_source == self.task.asset_id
            and self.asset_open
            and "OpenAsset" in self.events,
            "convention_grounded": self.evidence_convention == COORDINATE_CONVENTION
            and self.messages_read,
            "ticket_closed": self.ticket_closed,
            "single_submission": self.submissions == 1,
            "source_preserved": sha256(self.source_image).hexdigest()
            == STUDIO_SOURCE_DIGEST,
            "inspection_retained": "InspectBoxes" in self.events
            and "AttachEvidenceAndClose" in self.events,
        }
        return checks

    def succeeded(self) -> bool:
        return all(self.verdict().values())


@dataclass
class StudioAttempt:
    """A fresh world paired with only the observations shown to the policy."""

    world: StudioWorld
    policy_observations: list[dict[str, object]]


def _edit_report_payload(
    policy_observations: list[dict[str, object]],
    observed_boxes: dict[str, tuple[int, ...]],
) -> dict[str, object]:
    if not any(
        COORDINATE_CONVENTION == observation.get("coordinate_convention")
        for observation in policy_observations
    ):
        raise ValueError("coordinate convention is absent from the policy history")
    if not observed_boxes:
        raise ValueError("the policy has no observed centers")
    red_y = box_center(observed_boxes["red"])[1]
    blue_y = box_center(observed_boxes["blue"])[1]
    relation = "blue below red" if blue_y > red_y else "blue above red"
    return {"relation": relation}


def _inspection_payload(tool: str, argument_key: str) -> dict[str, object]:
    canonical, identifiers = adapt_inspection(tool, {argument_key: ["red", "blue"]})
    if canonical != "InspectBoxes":
        raise ValueError("inspection adapter changed the canonical action")
    return {"ids": identifiers}


def _action_payload(
    action: str,
    policy_observations: list[dict[str, object]],
    observed_boxes: dict[str, tuple[int, ...]],
    tool: str,
    argument_key: str,
) -> dict[str, object]:
    if action == "OpenAsset":
        return {"asset_id": str(policy_observations[0]["asset_id"])}
    if action == "InspectBoxes":
        return _inspection_payload(tool, argument_key)
    if action == "EditReport":
        return _edit_report_payload(policy_observations, observed_boxes)
    if action == "AttachEvidenceAndClose":
        return {
            "centers": {key: box_center(box) for key, box in observed_boxes.items()},
            "asset_id": policy_observations[0]["asset_id"],
            "coordinate_convention": COORDINATE_CONVENTION,
        }
    if action == "Submit":
        return {"final_sentence": "The blue square is below the red circle."}
    return {}


def run_studio_attempt(
    agent_name: str,
    order: Sequence[str] = BASE_TRACE,
    task: StudioTask | None = None,
    *,
    drop_convention: bool = False,
) -> StudioAttempt:
    """Select one teaching loop, start fresh, and execute a learner-visible route."""

    if agent_name not in HARNESS_TO_INSPECTION:
        raise ValueError("unregistered teaching agent loop")
    task = task or StudioTask()
    world = StudioWorld(task=task, report_relation=task.initial_relation)
    policy_observations = [task.initial_observation()]
    observed_boxes: dict[str, tuple[int, ...]] = {}
    tool, argument_key = HARNESS_TO_INSPECTION[agent_name]
    for action in order:
        payload = _action_payload(
            action, policy_observations, observed_boxes, tool, argument_key
        )
        observation = world.execute(action, payload)
        if action == "InspectBoxes":
            observed_boxes = cast(dict[str, tuple[int, ...]], observation["boxes"])
        if drop_convention and action == "SearchMessages":
            observation.pop("coordinate_convention")
        policy_observations.append(observation)
    return StudioAttempt(world, policy_observations)


def event_policy_history(
    event_will_arrive: bool, decision_time: int
) -> tuple[str, ...]:
    """Return what a policy may use at a decision, not future event state."""

    history = ["initial brief", "existing report"]
    if event_will_arrive and decision_time >= 5:
        history.append("clarification visible")
    if event_will_arrive and decision_time >= 6:
        history.append(CLARIFICATION)
    return tuple(history)
