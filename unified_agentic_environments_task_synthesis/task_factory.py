"""Deterministic concept-to-material-to-task construction for Chapter 24.

This finite teaching factory is book-authored. It is not K3's private generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .graphs import topological_order


ROOT = Path(__file__).resolve().parent
PACKET_ID = "analysis-audit-001"
EVALUATOR_SENTINEL = "EVALUATOR_ONLY_7d1f"
_LOCAL_SOURCE_INDEX = {
    "machine learning classification evaluation class-wise recall balanced accuracy ordinary accuracy": (
        "metric-definitions",
        "methodology",
        "counts",
    )
}


@dataclass(frozen=True)
class Task:
    """One task family with its own initial world and completion contract."""

    family: str
    request: str
    files: tuple[str, ...]
    editable: tuple[str, ...]
    deliverables: tuple[str, ...]
    capabilities: tuple[str, ...]
    completion: str
    packet_id: str = PACKET_ID
    selected_concepts: tuple[str, ...] = ("balanced accuracy", "ordinary accuracy")
    source_ids: tuple[str, ...] = ("metric-definitions", "methodology", "counts")


class ConceptGraph:
    """A finite teaching graph with declared aliases and specificity ranks."""

    def __init__(self) -> None:
        self.ranks = {
            "machine learning": 0,
            "classification evaluation": 1,
            "class imbalance": 2,
            "class-wise recall": 2,
            "ordinary accuracy": 2,
            "balanced accuracy": 3,
            "precision": 2,
        }
        self.aliases = {"class recall": "class-wise recall"}
        self.edges = [
            ("machine learning", "classification evaluation"),
            ("classification evaluation", "class imbalance"),
            ("classification evaluation", "class-wise recall"),
            ("classification evaluation", "ordinary accuracy"),
            ("class-wise recall", "balanced accuracy"),
        ]

    def resolve(self, label: str) -> str:
        """Reuse only an explicitly declared alias or an existing identity."""
        node = self.aliases.get(label, label)
        if node not in self.ranks:
            raise ValueError(f"unknown concept: {label}")
        return node

    def add_edge(self, parent: str, child: str) -> None:
        """Reject a cycle and any edge that fails the declared rank rule."""
        left, right = self.resolve(parent), self.resolve(child)
        try:
            topological_order(tuple(self.ranks), (*self.edges, (left, right)))
        except ValueError as error:
            raise ValueError("cycle in concept graph") from error
        if self.ranks[left] >= self.ranks[right]:
            raise ValueError("edge reverses or lacks conceptual specificity")
        if (left, right) not in self.edges:
            self.edges.append((left, right))

    def ancestor_query(self, selected: str) -> str:
        """Form an ancestor-qualified local-index query for one selected node."""
        node = self.resolve(selected)
        parents = {child: parent for parent, child in self.edges}
        labels = [node]
        while labels[-1] in parents:
            labels.append(parents[labels[-1]])
        return " ".join(reversed(labels))

    def material_query(self, primary: str, related: tuple[str, ...] = ()) -> str:
        """Carry the primary ancestors and any declared related selections."""
        return " ".join(
            (self.ancestor_query(primary), *(self.resolve(x) for x in related))
        )


def source_index(root: Path = ROOT) -> dict[str, dict[str, object]]:
    """Verify exact local packet bytes and return records by source identity."""
    manifest = json.loads((root / "sources.json").read_text(encoding="utf-8"))
    records: dict[str, dict[str, object]] = {}
    for raw in manifest["packet_sources"]:
        record = dict(raw)
        file = root / str(record["file"])
        if not file.is_file():
            raise ValueError(f"missing source: {record['id']}")
        observed = sha256(file.read_bytes()).hexdigest()
        if observed != record["sha256"]:
            raise ValueError(f"changed source identity: {record['id']}")
        records[str(record["id"])] = record
    return records


def selected_materials(
    root: Path = ROOT, *, query: str | None = None
) -> tuple[str, ...]:
    """Use a small local index and verified evidence roles for source selection."""
    records = source_index(root)
    qualified_query = (
        ConceptGraph().material_query("balanced accuracy", ("ordinary accuracy",))
        if query is None
        else query
    )
    if qualified_query not in _LOCAL_SOURCE_INDEX:
        raise ValueError("local index has no support for the selected query")
    required = {
        "balanced-accuracy",
        "ordinary-accuracy",
        "required-metric",
        "valid-input-domain",
        "classifier-counts",
    }
    selected = tuple(
        identity
        for identity in _LOCAL_SOURCE_INDEX[qualified_query]
        if identity in records
    )
    supported: set[str] = set()
    for identity in selected:
        roles = records[identity]["supports"]
        if not isinstance(roles, list) or not all(
            isinstance(role, str) for role in roles
        ):
            raise ValueError(f"invalid source roles: {identity}")
        supported.update(roles)
    missing = sorted(required - supported)
    if missing:
        raise ValueError(f"unsupported material dependency: {', '.join(missing)}")
    return selected


def construct_task(family: str, root: Path = ROOT) -> Task:
    """Build one fully specified task from the verified shared packet."""
    selected = selected_materials(root)
    common = ("metric_definitions.md", "methodology.md", "counts.csv")
    if family == "question":
        return Task(
            family,
            "Using the methodology note and class counts, decide which fictional "
            "classifier has the higher required score. Give both balanced and "
            "ordinary accuracy scores, explain the reversed ordering, and identify "
            "the source of the metric definition.",
            common,
            (),
            ("answer",),
            ("read", "search", "calculate", "write answer"),
            "supported answer with both scores and aggregation explanation",
            source_ids=selected,
        )
    if family == "repair":
        return Task(
            family,
            "Repair score(rows) in analysis.py to compute the methodology's "
            "class-balanced score for every valid two-class input. Preserve the "
            "public interface and source data; submit the revised program and "
            "its test results.",
            (*common, "analysis.py"),
            ("analysis.py", "output/"),
            ("analysis.py", "output/test-results.json"),
            ("read", "edit program", "run tests"),
            "program agrees with independent public and changed-input checks",
            source_ids=(*selected, "buggy-program"),
        )
    if family == "report":
        return Task(
            family,
            "Create scores.csv and audit.md from the methodology and class "
            "counts. Include both class recalls, balanced and ordinary accuracy "
            "for each classifier, and explain their different rankings. Preserve "
            "the source files; submit both output files.",
            common,
            ("output/",),
            ("output/scores.csv", "output/audit.md"),
            ("read", "calculate", "write report files"),
            "parsed numerical outputs and stated conclusion agree; files exist",
            source_ids=selected,
        )
    raise ValueError(f"unknown task family: {family}")


def learner_bundle(task: Task) -> dict[str, object]:
    """Return only the initial observation and accessible file index."""
    bundle: dict[str, object] = {
        "task": task.family,
        "request": task.request,
        "file_index": list(task.files),
        "capabilities": list(task.capabilities),
        "editable": list(task.editable),
        "deliverables": list(task.deliverables),
    }
    assert_learner_boundary(bundle)
    return bundle


def assert_learner_boundary(bundle: dict[str, object]) -> None:
    """Reject protected fields and a sentinel even inside error/metadata text."""
    serialized = json.dumps(bundle, sort_keys=True).lower()
    forbidden = (
        "evaluator_only_7d1f",
        "reference_patch",
        "hidden_cases",
        "private_data",
    )
    if any(token in serialized for token in forbidden):
        raise ValueError("evaluator-only information leaked into learner view")


def validate_reference_reads(task: Task, accessed: set[str]) -> None:
    """Require a feasibility route to use only learner-accessible files."""
    unseen = accessed - set(task.files)
    if unseen:
        raise ValueError(
            f"reference used evaluator-only material: {', '.join(sorted(unseen))}"
        )


def fresh_workspace(task: Task, root: Path = ROOT) -> _FreshWorkspace:
    """Yield one independent temporary world containing only permitted files."""
    # The caller owns the context manager's lifetime; no prior task is reused.
    return _FreshWorkspace(task, root)


class _FreshWorkspace:
    def __init__(self, task: Task, root: Path) -> None:
        self.task, self.root = task, root
        self.temporary = TemporaryDirectory(prefix="ch24-task-")

    def __enter__(self) -> Path:
        world = Path(self.temporary.name)
        (world / "output").mkdir()
        for name in self.task.files:
            target = world / name
            target.write_bytes((self.root / "fixtures" / name).read_bytes())
            target.chmod(0o644 if name in self.task.editable else 0o444)
        return world

    def __exit__(self, *args: object) -> None:
        self.temporary.cleanup()


def packet_split_valid(assignments: dict[str, tuple[str, str, str]]) -> bool:
    """Keep protected packet descendants together; public definitions may recur."""
    groups: dict[str, set[str]] = {}
    for split, packet_id, _public_definition in assignments.values():
        groups.setdefault(packet_id, set()).add(split)
    return all(len(splits) == 1 for splits in groups.values())


def required_dependencies(extra_request: str = "") -> tuple[str, ...]:
    """Return supported dependencies; extra wording supplies no new evidence."""
    if "longitudinal trend" in extra_request.lower():
        raise ValueError("single snapshot cannot support a longitudinal trend")
    return selected_materials()
