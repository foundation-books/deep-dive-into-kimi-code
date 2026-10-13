"""Deterministic onboarding traces used by chapter assets and audits."""

from __future__ import annotations

from .onboarding import (
    DATE_AVAILABLE,
    DAY,
    END,
    GRANT_AVAILABLE,
    MANAGER_AVAILABLE,
    OnboardingWorld,
)


def scripted_attempt(strategy: str = "incremental") -> OnboardingWorld:
    """Two valid paths and several controlled, explicitly invalid variants."""
    if strategy not in {
        "incremental",
        "collect",
        "static",
        "calendar_only",
        "restart",
        "late",
        "collateral",
        "wrong_notice",
    }:
        raise ValueError(strategy)
    world = OnboardingWorld()
    world.step("create_calendar", day=4, manager="Mira")
    world.step("create_task", day=4, manager="Mira")
    world.step("request_provision", day=4, manager="Mira")
    world.mark("initial commitments")
    if strategy == "static":
        world.step("wait", until=END)
        world.step("finalize")
        return world
    if strategy == "late":
        world.step("wait", until=DAY + 15 * 60)
    elif strategy == "collect":
        world.step("wait", until=MANAGER_AVAILABLE)
    else:
        world.step("wait", until=DATE_AVAILABLE)
    world.step("read_messages")
    world.mark("updates observed" if strategy == "collect" else "date observed")
    if strategy not in {"late", "collect"}:
        world.step("revise_calendar", day=6)
        if strategy != "calendar_only":
            world.step("revise_task", day=6)
        world.mark("date reconciled")
        world.step("wait", until=MANAGER_AVAILABLE)
        world.step("read_messages")
    if strategy == "restart":
        world.step("create_calendar", day=6, manager="Tomas")
        world.step("create_task", day=6, manager="Tomas")
        world.step("request_provision", day=6, manager="Tomas")
    else:
        world.step("revise_calendar", day=6, manager="Tomas")
        if strategy != "calendar_only":
            world.step("revise_task", day=6, manager="Tomas")
    world.mark("manager reconciled")
    if world.now < GRANT_AVAILABLE:
        world.step("wait", until=GRANT_AVAILABLE)
    world.mark("old result available")
    world.step("inspect_task")
    world.step("inspect_access")
    if strategy not in {"calendar_only", "restart"}:
        world.step("reconcile_grant", day=6, manager="Tomas")
    world.mark("grant reconciled")
    if strategy == "collateral":
        world.step("edit_unrelated", day=6)
    world.step(
        "send_notice",
        day=6,
        manager="Tomas",
        recipient="Mira" if strategy == "wrong_notice" else "Tomas",
    )
    world.step("wait", until=END)
    world.step("finalize")
    return world
