"""Exact reward and work-ledger checks for the Chapter 26 fixtures."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence


def _status(check_id: str, predicate: bool, **fields: object) -> dict[str, object]:
    return {"check_id": check_id, "status": "pass" if predicate else "fail", **fields}


def _work_and_critical(
    orchestrator_steps: Sequence[int],
    worker_stages: Sequence[Mapping[str, int]],
) -> tuple[int, int, tuple[tuple[str, ...], ...]]:
    if len(orchestrator_steps) != len(worker_stages) or not orchestrator_steps:
        raise ValueError("aligned nonempty stage records are required")
    if any(step < 0 for step in orchestrator_steps):
        raise ValueError("orchestrator steps must be nonnegative")
    work = 0
    critical = 0
    witnesses: list[tuple[str, ...]] = []
    for main, workers in zip(orchestrator_steps, worker_stages, strict=True):
        if any(step < 0 for step in workers.values()):
            raise ValueError("worker steps must be nonnegative")
        maximum = max(workers.values(), default=0)
        work += main + sum(workers.values())
        critical += main + maximum
        witnesses.append(
            tuple(sorted(name for name, step in workers.items() if step == maximum))
            if workers
            else ()
        )
    return work, critical, tuple(witnesses)


def work_critical_check() -> dict[str, object]:
    """Reproduce the canonical and seven-step-repair ledgers."""

    main = (1, 1, 1, 2)
    canonical_workers: tuple[Mapping[str, int], ...] = (
        {"R": 3, "C": 6, "V": 4, "D": 5},
        {"RC": 3, "RV": 2, "D3_REPAIR": 2},
        {"J": 2},
        {},
    )
    long_repair_workers = (
        canonical_workers[0],
        {"RC": 3, "RV": 2, "D3_REPAIR": 7},
        canonical_workers[2],
        canonical_workers[3],
    )
    canonical = _work_and_critical(main, canonical_workers)
    long_repair = _work_and_critical(main, long_repair_workers)
    return _status(
        "CH25-D03",
        canonical[:2] == (32, 16) and long_repair[:2] == (37, 20),
        canonical_work=canonical[0],
        canonical_critical_steps=canonical[1],
        canonical_speedup=str(Fraction(32, canonical[1])),
        canonical_work_inflation=str(Fraction(canonical[0], 32)),
        long_repair_work=long_repair[0],
        long_repair_critical_steps=long_repair[1],
        unmatched_long_repair_speedup=str(Fraction(32, long_repair[1])),
        matched_long_repair_speedup=str(Fraction(37, long_repair[1])),
        matched_long_repair_inflation=str(Fraction(37, 37)),
    )


def critical_tie_check() -> dict[str, object]:
    """Preserve every maximizer in a critical-branch witness."""

    _, critical, witnesses = _work_and_critical(
        (1, 1, 1, 2),
        (
            {"R": 3, "C": 6, "V": 4, "D": 5},
            {"RC": 3, "RV": 3, "D3_REPAIR": 2},
            {"J": 2},
            {},
        ),
    )
    expected = (("C",), ("RC", "RV"), ("J",), ())
    return _status(
        "CH25-D04",
        critical == 16 and witnesses == expected,
        critical_steps=critical,
        critical_branch_sets=[list(stage) for stage in witnesses],
        tie_preserved=witnesses[1] == ("RC", "RV"),
    )


def _reward(
    components: Sequence[Fraction], lambda_parallel: Fraction, lambda_finish: Fraction
) -> Fraction:
    parallel, finish, performance = components
    return lambda_parallel * parallel + lambda_finish * finish + performance


def reward_annealing_check() -> dict[str, object]:
    """Reproduce reward arithmetic and the zero-auxiliary endpoint."""

    trajectories = {
        "serial": (Fraction(0), Fraction(1), Fraction(4, 5)),
        "valid_swarm": (Fraction(4, 5), Fraction(1), Fraction(9, 10)),
        "spurious_fanout": (Fraction(1), Fraction(0), Fraction(1, 2)),
    }
    early = {
        name: _reward(values, Fraction(1, 5), Fraction(3, 10))
        for name, values in trajectories.items()
    }
    exploration = {
        name: _reward(values, Fraction(4), Fraction(3, 10))
        for name, values in trajectories.items()
    }
    final = {
        name: _reward(values, Fraction(0), Fraction(0))
        for name, values in trajectories.items()
    }
    return _status(
        "CH25-D05",
        early["valid_swarm"] == Fraction(34, 25)
        and exploration["spurious_fanout"] == Fraction(9, 2)
        and final["valid_swarm"] > final["spurious_fanout"],
        early_totals={name: str(value) for name, value in early.items()},
        exploration_totals={name: str(value) for name, value in exploration.items()},
        final_totals={name: str(value) for name, value in final.items()},
        auxiliary_endpoint="performance_only",
    )


__all__ = ["critical_tie_check", "reward_annealing_check", "work_critical_check"]
