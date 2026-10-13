"""Required DAG versus observed order; dependencies survive witness reduction."""

from typing import Sequence


def closure(
    seeds: set[str], edges: Sequence[tuple[str, str]], ancestors: bool
) -> set[str]:
    """Reach all ancestors or descendants, including the seed events."""
    reached = set(seeds)
    changed = True
    while changed:
        old = len(reached)
        for source, target in edges:
            trigger, add = (target, source) if ancestors else (source, target)
            if trigger in reached:
                reached.add(add)
        changed = len(reached) != old
    return reached


def violations(
    order: Sequence[str], edges: Sequence[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    """Return required edges reversed by a complete observed ordering."""
    if len(set(order)) != len(order):
        raise ValueError("duplicate event identity")
    index = {event: i for i, event in enumerate(order)}
    if any(a not in index or b not in index for a, b in edges):
        raise ValueError("missing dependency endpoint")
    return tuple((a, b) for a, b in edges if index[a] >= index[b])


def reduce_witness(
    order: Sequence[str], edges: Sequence[tuple[str, str]], witness: tuple[str, str]
) -> tuple[str, ...]:
    """Keep a violated edge and its ancestors; delete unrelated dependents."""
    if witness not in violations(order, edges):
        raise ValueError("supplied edge is not a failure witness")
    # The required graph must be acyclic even though observed order is wrong.
    if any(a in closure({b}, edges, False) for a, b in edges):
        raise ValueError("required graph has a cycle")
    protected = closure(set(witness), edges, True)
    retained = list(order)
    for event in order:
        if event not in retained or event in protected:
            continue
        remove = closure({event}, edges, False)
        if remove & protected:
            continue
        trial = [e for e in retained if e not in remove]
        trial_edges = [(a, b) for a, b in edges if a in trial and b in trial]
        if witness in violations(trial, trial_edges):
            retained = trial
    return tuple(retained)


def critical_path(
    durations: dict[str, float], edges: Sequence[tuple[str, str]]
) -> float:
    """DAG longest path; resource contention must be modeled separately."""
    from math import isfinite

    if any(not isfinite(v) or v < 0 for v in durations.values()):
        raise ValueError("finite nonnegative operation durations required")
    if any(a not in durations or b not in durations for a, b in edges):
        raise ValueError("missing operation")
    remaining = set(durations)
    finish: dict[str, float] = {}
    while remaining:
        ready = [v for v in remaining if all(a in finish for a, b in edges if b == v)]
        if not ready:
            raise ValueError("dependency cycle")
        for vertex in ready:
            finish[vertex] = durations[vertex] + max(
                (finish[a] for a, b in edges if b == vertex), default=0
            )
            remaining.remove(vertex)
    return max(finish.values(), default=0)
