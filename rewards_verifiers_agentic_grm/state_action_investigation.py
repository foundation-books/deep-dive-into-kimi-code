"""Constructed state/action fixture motivated by pinned tau2-bench evaluators."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Literal, Sequence

Action = tuple[Literal["set_blue_y", "shift_blue_y"], int]
REFERENCE_ACTIONS: tuple[Action, ...] = (("set_blue_y", 80), ("shift_blue_y", 20))
STATE_TRAJECTORIES: dict[str, tuple[Action, ...]] = {
    "S1": REFERENCE_ACTIONS,
    "S2": (("set_blue_y", 100),),
    "S3": (("shift_blue_y", 20), ("set_blue_y", 80)),
}


def final_blue_y(actions: Sequence[Action], *, initial: int = 40) -> int:
    """Replay the two book-created tools in order."""
    current = initial
    for tool, value in actions:
        if tool == "set_blue_y":
            current = value
        elif tool == "shift_blue_y":
            current += value
        else:
            raise ValueError("unknown studio tool")
    return current


def action_inclusion(
    actions: Sequence[Action], required: Sequence[Action] = REFERENCE_ACTIONS
) -> bool:
    """Check presence, neither exact transcript equality nor reference order."""
    return all(expected in actions for expected in required)


def component_results(actions: Sequence[Action]) -> dict[str, int]:
    """Return distinct final-state and required-action predicates."""
    return {
        "DB": int(final_blue_y(actions) == 100),
        "ACTION": int(action_inclusion(actions)),
    }


def selected_product(
    components: dict[str, int], basis: Sequence[str]
) -> tuple[Fraction, tuple[str, ...]]:
    """Return a selected binary product and the identities actually selected.

    An empty basis has multiplicative identity one; it makes no claim that a
    missing check ran or passed.
    """
    if len(set(basis)) != len(basis):
        raise ValueError("duplicate reward-basis component")
    result = Fraction(1)
    for name in basis:
        value = components[name]
        if value not in {0, 1}:
            raise ValueError("constructed components must be binary")
        result *= value
    return result, tuple(basis)


def hard_first_with_judge(
    gate: int | None, judge: Callable[[], Fraction], *, blocked: Fraction = Fraction(-1)
) -> Fraction | None:
    """Demonstrate that a blocked gate skips the soft callback entirely."""
    if gate == 0:
        return blocked
    if gate is None:
        return None
    if gate != 1:
        raise ValueError("gate must be zero, one, or unresolved")
    return judge()
