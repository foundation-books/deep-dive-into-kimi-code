"""Book-created Chapter 25 event and media-binding checks.

These records describe what a caller must preserve. They do not encode a K3
chat template or stand in for the released processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True)
class Media:
    """One expected image payload with immutable event and pixel identity."""

    event_id: str
    pixels: bytes
    width: int
    height: int

    @property
    def digest(self) -> str:
        return sha256(self.pixels).hexdigest()


@dataclass(frozen=True)
class Event:
    """One interaction event with declared author and intervention status."""

    event_id: str
    author: str
    kind: str
    sampled_tokens: int = 0
    intervention: bool = False


def validate_binding(
    expected: Iterable[Media], supplied: Iterable[Media], *, prepared_grids: int
) -> None:
    """Reject missing, reordered, or wrong-pixel media, including equal shapes."""

    wanted = tuple(expected)
    actual = tuple(supplied)
    if len(wanted) != len(actual) or len(actual) != prepared_grids:
        raise ValueError("media count and prepared grid count disagree")
    for index, (left, right) in enumerate(zip(wanted, actual, strict=True)):
        if (
            left.event_id != right.event_id
            or left.digest != right.digest
            or (left.width, left.height) != (right.width, right.height)
        ):
            raise ValueError(f"media identity/order mismatch at position {index}")


def check_policy_request(
    text: str, supplied: Iterable[Media], *, protected_terms: Iterable[str]
) -> None:
    """Reject answer leakage and path-only claims of visual delivery."""

    if any(term in text for term in protected_terms):
        raise ValueError("protected evaluator information entered policy text")
    if not tuple(supplied):
        raise ValueError("path or image marker without supplied media")


def action_mask(events: Iterable[Event]) -> list[int]:
    """One per assistant-generated token, zero on inserted observations."""

    mask: list[int] = []
    for event in events:
        if event.sampled_tokens < 0:
            raise ValueError("negative span length")
        if event.kind == "image" and event.author == "assistant":
            raise ValueError("executor image mislabeled as assistant action")
        mask.extend(
            [int(event.author == "assistant" and not event.intervention)]
            * event.sampled_tokens
        )
    return mask


def can_continue_after_tool(mask_length: int, response_budget: int) -> bool:
    """Check whether a further image-conditioned decision fits the budget."""

    if response_budget < 0 or mask_length < 0:
        raise ValueError("negative response length")
    return mask_length < response_budget


def eligible_for_policy_batch(events: Iterable[Event]) -> bool:
    """Reject diagnostic interventions from an ordinary sampled batch."""

    return not any(event.intervention for event in events)


def terminal_success(tool_completed: bool, verifier_passed: bool) -> bool:
    """Execution is a separate predicate from terminal task success."""

    _ = tool_completed
    return verifier_passed
