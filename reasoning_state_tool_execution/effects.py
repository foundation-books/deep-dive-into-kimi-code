"""Finite effect and recovery semantics for Chapter 28."""

from __future__ import annotations


EFFECT_CLASSES = (
    "pure",
    "idempotent",
    "deduplicated",
    "compensatable",
    "nonrepeatable",
)


def keyed_insert(state: frozenset[str], key: str) -> frozenset[str]:
    """Apply an idempotent keyed set insertion."""
    return state | {key}


def unkeyed_increment(state: int, _key: str) -> int:
    """Apply a deliberately non-idempotent increment."""
    return state + 1


def idempotency_witness(operation: str) -> dict[str, object]:
    """Evaluate an operation once and twice to expose idempotency."""
    if operation == "keyed-insert":
        initial_keys = frozenset({"a"})
        once_keys = keyed_insert(initial_keys, "k")
        twice_keys = keyed_insert(once_keys, "k")
        return {
            "once": sorted(once_keys),
            "twice": sorted(twice_keys),
            "idempotent": once_keys == twice_keys,
        }
    if operation == "unkeyed-increment":
        once_count = unkeyed_increment(0, "k")
        twice_count = unkeyed_increment(once_count, "k")
        return {
            "once": once_count,
            "twice": twice_count,
            "idempotent": once_count == twice_count,
        }
    raise ValueError(f"unknown operation: {operation}")


def recovery_action(knowledge: str, effect_class: str) -> tuple[str, ...]:
    """Return admissible first actions for the chapter's decision matrix."""

    if effect_class not in EFFECT_CLASSES:
        raise ValueError(f"unknown effect class: {effect_class}")
    if knowledge == "known-success":
        return ("reuse",)
    if knowledge == "known-failure-no-effect":
        if effect_class == "nonrepeatable":
            return ("retry-with-explicit-authority", "stop")
        return ("retry",)
    if knowledge != "unknown":
        raise ValueError(f"unknown effect knowledge: {knowledge}")
    return {
        "pure": ("reconcile", "recompute"),
        "idempotent": ("reconcile", "retry-same-key"),
        "deduplicated": ("reconcile", "retry-within-scope"),
        "compensatable": ("reconcile", "compensate", "escalate"),
        "nonrepeatable": ("reconcile", "escalate", "stop"),
    }[effect_class]
