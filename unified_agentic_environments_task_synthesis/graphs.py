"""Graph checks for Chapter 24's concept and contamination examples."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence


def topological_order(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> tuple[str, ...]:
    """Return a stable topological order or reject a directed cycle."""

    positions = {node: index for index, node in enumerate(nodes)}
    indegree = {node: 0 for node in nodes}
    children: dict[str, set[str]] = {node: set() for node in nodes}
    for parent, child in edges:
        if parent not in indegree or child not in indegree:
            raise ValueError("edge references an unknown node")
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1
    ready = deque(
        sorted((n for n in nodes if indegree[n] == 0), key=lambda node: positions[node])
    )
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for child in sorted(children[node], key=lambda item: positions[item]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("concept graph contains a directed cycle")
    return tuple(order)


def concept_graph_check() -> dict[str, object]:
    """Return diagnostics for the concept graph invariant."""

    nodes = (
        "geometry",
        "document auditing",
        "spatial relation",
        "evidence-backed closure",
        "vertical order",
    )
    accepted = (
        ("geometry", "spatial relation"),
        ("spatial relation", "vertical order"),
        ("document auditing", "evidence-backed closure"),
    )
    order = topological_order(nodes, accepted)
    cycle_rejected = False
    try:
        topological_order(nodes, (*accepted, ("vertical order", "geometry")))
    except ValueError:
        cycle_rejected = True
    return {
        "status": "pass" if cycle_rejected and len(order) == 5 else "fail",
        "topological_order": list(order),
        "cycle_edge_rejected": cycle_rejected,
        "alias_reused": {"up/down relation": "vertical order"},
        "proposal_dispositions": {"accepted": 3, "rejected": 1, "reused": 1},
    }


def connected_components(
    nodes: Sequence[str], edges: Iterable[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    """Return sorted components of an undirected concept graph."""

    neighbors: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    components: list[tuple[str, ...]] = []
    unseen = set(nodes)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(neighbors[node] - component)
        unseen -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def contamination_check() -> dict[str, object]:
    """Return diagnostics for the contamination invariant."""

    tasks = tuple(f"t{index}" for index in range(1, 7))
    edges = (("t1", "t2"), ("t3", "t4"), ("t5", "t6"))
    components = connected_components(tasks, edges)
    train = {"t1", "t3", "t5"}
    evaluation = {"t2", "t4", "t6"}
    crossing = [c for c in components if set(c) & train and set(c) & evaluation]
    return {
        "status": "pass" if len(crossing) == 3 else "fail",
        "components": [list(component) for component in components],
        "crossing_component_count": len(crossing),
        "split_valid": not crossing,
        "prompt_only_deduplication_sufficient": False,
    }
