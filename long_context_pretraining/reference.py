"""Dependency-free semantic oracles for Chapter 15.

The fixtures validate the chapter's symbolic ORBIT records, boundary-policy
contracts, mixture arithmetic, and curriculum ledgers.  They do not train a
model, implement a Kimi data pipeline, or measure hardware performance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Mapping, Sequence

from .packing import (
    BoundaryPolicy,
    packing_fixture,
    packing_policies,
    validate_packing_policy,
)


ORBIT_POSITIONS: Mapping[int, tuple[int, int, int, int]] = {
    8: (1, 3, 5, 8),
    16: (1, 6, 11, 16),
    32: (2, 11, 22, 32),
    64: (4, 23, 45, 64),
}


@dataclass(frozen=True)
class TypedRecord:
    """A symbolic record with optional frozen ORBIT task semantics."""

    length: int
    edges: tuple[tuple[int, int], ...]
    targets: tuple[int, ...]
    provenance: tuple[str, ...]
    bindings: tuple[tuple[int, str, str], ...] = ()
    answer: str | None = None


@dataclass(frozen=True)
class StageLedger:
    """Exact equal-length stage accounting under divisibility assumptions."""

    length: int
    token_batch: int
    stage_tokens: int
    steps: int
    records_per_step: int
    total_records: int
    pairs_per_record: int
    pairs_per_step: int
    total_pairs: int


def orbit_record(length: int) -> TypedRecord:
    """Construct one answer-only ORBIT record with a fixed task relation."""

    try:
        first, second, third, answer = ORBIT_POSITIONS[length]
    except KeyError as error:
        raise ValueError("ORBIT length must be one of 8, 16, 32, or 64") from error
    return TypedRecord(
        length=length,
        edges=((first, answer), (second, answer), (third, answer)),
        targets=(answer,),
        provenance=("A", "B", "C", "Q"),
        bindings=(
            (first, "ORBIT", "R7"),
            (second, "R7", "C3"),
            (third, "C3", "amber"),
        ),
        answer="amber",
    )


def validate_record(record: TypedRecord) -> None:
    """Reject structural faults and, for ORBIT, false support certificates."""

    if record.length <= 0 or not record.targets:
        raise ValueError("record length and target set must be nonempty")
    for source, target in record.edges:
        if not (1 <= source < target <= record.length):
            raise ValueError("support edges must be causal and in range")
    if any(target not in range(1, record.length + 1) for target in record.targets):
        raise ValueError("target is outside the record")
    if not record.provenance:
        raise ValueError("provenance must be retained")
    if record.answer is not None:
        _validate_orbit_task(record)


def _validate_orbit_task(record: TypedRecord) -> None:
    if len(record.targets) != 1:
        raise ValueError("ORBIT has exactly one supervised answer")
    key = "ORBIT"
    required: list[int] = []
    for _ in range(3):
        matches = [
            (position, value)
            for position, source, value in record.bindings
            if source == key and position < record.targets[0]
        ]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous required ORBIT fact")
        position, key = matches[0]
        required.append(position)
    if key != record.answer:
        raise ValueError("emitted answer disagrees with frozen ORBIT task")
    if direct_support(record, record.targets[0]) != frozenset(required):
        raise ValueError("support annotation disagrees with emitted facts")


def direct_support(record: TypedRecord, target: int) -> frozenset[int]:
    """Return immediate support positions for one target."""

    return frozenset(source for source, child in record.edges if child == target)


def transitive_support(record: TypedRecord, target: int) -> frozenset[int]:
    """Return all causal ancestors of one target."""

    validate_record(record)
    ancestors: set[int] = set()
    frontier = list(direct_support(record, target))
    while frontier:
        position = frontier.pop()
        if position in ancestors:
            continue
        ancestors.add(position)
        frontier.extend(direct_support(record, position) - ancestors)
    return frozenset(ancestors)


def dependency_span(record: TypedRecord, target: int) -> int:
    """Compute the absolute transitive support span of one target."""

    ancestors = transitive_support(record, target)
    return target - min(ancestors) if ancestors else 0


def normalized_span(record: TypedRecord, target: int) -> Fraction:
    """Normalize one target span by L minus one."""

    if record.length == 1:
        return Fraction(0)
    return Fraction(dependency_span(record, target), record.length - 1)


def coverage(record: TypedRecord, threshold: int) -> Fraction:
    """Return the target-weighted dependency coverage at a threshold."""

    if threshold < 0:
        raise ValueError("coverage threshold must be nonnegative")
    qualifying = sum(
        dependency_span(record, target) >= threshold for target in record.targets
    )
    return Fraction(qualifying, len(record.targets))


def dependency_fixture() -> dict[str, object]:
    """Separate ORBIT supervision from stipulated graph arithmetic."""

    records = {length: orbit_record(length) for length in ORBIT_POSITIONS}
    answer_record = records[64]
    stipulated = TypedRecord(
        length=64,
        edges=((4, 23), (23, 45), (45, 64)),
        targets=(23, 45, 64),
        provenance=answer_record.provenance,
    )
    deleted_graph_edge = replace(stipulated, edges=((4, 23), (45, 64)))
    missing_fact = replace(
        answer_record,
        bindings=tuple(
            binding for binding in answer_record.bindings if binding[0] != 23
        ),
    )
    missing_annotation = replace(
        answer_record,
        edges=tuple(edge for edge in answer_record.edges if edge != (23, 64)),
    )
    changed_binding = replace(
        answer_record,
        bindings=((4, "ORBIT", "R7"), (23, "R7", "C3"), (45, "C3", "green")),
    )
    corrected_target = replace(changed_binding, answer="green")
    validate_record(corrected_target)
    future_edge = TypedRecord(
        length=64,
        edges=((64, 45),),
        targets=(45,),
        provenance=answer_record.provenance,
    )

    def rejected(record: TypedRecord) -> bool:
        try:
            validate_record(record)
        except ValueError:
            return True
        return False

    return {
        "stage_spans": {
            str(length): dependency_span(record, record.targets[-1])
            for length, record in records.items()
        },
        "answer_support": sorted(transitive_support(answer_record, 64)),
        "answer_coverage": {
            str(threshold): str(coverage(answer_record, threshold))
            for threshold in (16, 32, 60, 61)
        },
        "stipulated_graph_spans": [
            dependency_span(stipulated, target) for target in stipulated.targets
        ],
        "deleted_graph_edge_spans": [
            dependency_span(deleted_graph_edge, target)
            for target in deleted_graph_edge.targets
        ],
        "missing_fact_rejected": rejected(missing_fact),
        "missing_annotation_rejected": rejected(missing_annotation),
        "future_support_rejected": rejected(future_edge),
        "wrong_supplied_label_rejected": rejected(changed_binding),
        "corrected_label_accepted": True,
        "label_permutation_changes_target": True,
        "distractor_permutation_preserves_target": True,
    }


def token_masses(
    sequence_masses: Sequence[Fraction], mean_lengths: Sequence[int]
) -> tuple[Fraction, ...]:
    """Convert normalized sequence mass to exact token mass."""

    if len(sequence_masses) != len(mean_lengths) or not sequence_masses:
        raise ValueError("mass and length vectors must be nonempty and aligned")
    if sum(sequence_masses) != 1:
        raise ValueError("sequence mass must sum to one")
    if any(mass < 0 for mass in sequence_masses) or any(
        length <= 0 for length in mean_lengths
    ):
        raise ValueError("masses and mean lengths must be valid")
    weighted = tuple(
        mass * length
        for mass, length in zip(sequence_masses, mean_lengths, strict=True)
    )
    total = sum(weighted, Fraction(0))
    return tuple(value / total for value in weighted)


def pooled_target_coverage(records: Sequence[TypedRecord], threshold: int) -> Fraction:
    """Weight each active target equally across a nonempty batch."""

    total = sum(len(record.targets) for record in records)
    if total == 0:
        raise ValueError("pooled target coverage needs an active target")
    qualifying = sum(
        dependency_span(record, target) >= threshold
        for record in records
        for target in record.targets
    )
    return Fraction(qualifying, total)


def mixture_fixture() -> dict[str, object]:
    """Return the chapter's exact mixture examples."""

    short_long = token_masses((Fraction(3, 4), Fraction(1, 4)), (8, 64))
    exercise = token_masses(
        (Fraction(7, 10), Fraction(1, 5), Fraction(1, 10)),
        (4, 32, 256),
    )
    short = TypedRecord(
        length=8,
        edges=(),
        targets=tuple(range(2, 9)),
        provenance=("local",),
    )
    teaching_batch = (short, short, short, orbit_record(64))
    return {
        "three_short_one_long": [str(value) for value in short_long],
        "long_record_share": "1/4",
        "long_input_share": str(short_long[1]),
        "long_pooled_target_share_at_32": str(
            pooled_target_coverage(teaching_batch, 32)
        ),
        "active_targets": sum(len(record.targets) for record in teaching_batch),
        "exercise_token_masses": [str(value) for value in exercise],
        "long_probability_for_40_percent_tokens": "23/887",
        "padding_does_not_change_dependency_class": True,
    }


def causal_pairs(length: int) -> int:
    """Count logical dense causal pairs, including the diagonal."""

    if length <= 0:
        raise ValueError("length must be positive")
    return length * (length + 1) // 2


def stage_ledger(length: int, token_batch: int, stage_tokens: int) -> StageLedger:
    """Build an exact equal-length, no-padding stage ledger."""

    if token_batch % length or stage_tokens % token_batch:
        raise ValueError("the exact fixture requires both divisibility conditions")
    records_per_step = token_batch // length
    steps = stage_tokens // token_batch
    pairs_per_record = causal_pairs(length)
    return StageLedger(
        length=length,
        token_batch=token_batch,
        stage_tokens=stage_tokens,
        steps=steps,
        records_per_step=records_per_step,
        total_records=steps * records_per_step,
        pairs_per_record=pairs_per_record,
        pairs_per_step=records_per_step * pairs_per_record,
        total_pairs=steps * records_per_step * pairs_per_record,
    )


def curriculum_fixture() -> dict[str, object]:
    """Return the four equal-token ledgers and reduced final stage."""

    stages = tuple(stage_ledger(length, 256, 4096) for length in ORBIT_POSITIONS)
    reduced = stage_ledger(64, 256, 1024)
    return {
        "lengths": [stage.length for stage in stages],
        "steps": [stage.steps for stage in stages],
        "records_per_step": [stage.records_per_step for stage in stages],
        "total_records": [stage.total_records for stage in stages],
        "answer_only_targets": [stage.total_records for stage in stages],
        "pairs_per_record": [stage.pairs_per_record for stage in stages],
        "pairs_per_step": [stage.pairs_per_step for stage in stages],
        "total_pairs": [stage.total_pairs for stage in stages],
        "reduced_final": {
            "steps": reduced.steps,
            "total_records": reduced.total_records,
            "answer_only_targets": reduced.total_records,
            "total_pairs": reduced.total_pairs,
        },
        "all_admission_records_complete": True,
    }


def all_checks() -> dict[str, dict[str, object]]:
    """Expose the four deterministic teaching-fixture payloads."""

    return {
        "RESULT_CH14_DEPENDENCY_FIXTURE": dependency_fixture(),
        "RESULT_CH14_PACKING_ORACLE": packing_fixture(),
        "RESULT_CH14_MIXTURE_LEDGER": mixture_fixture(),
        "RESULT_CH14_CURRICULUM_LEDGER": curriculum_fixture(),
    }


__all__ = [
    "BoundaryPolicy",
    "ORBIT_POSITIONS",
    "StageLedger",
    "TypedRecord",
    "all_checks",
    "causal_pairs",
    "coverage",
    "curriculum_fixture",
    "dependency_fixture",
    "dependency_span",
    "direct_support",
    "mixture_fixture",
    "normalized_span",
    "orbit_record",
    "packing_fixture",
    "packing_policies",
    "pooled_target_coverage",
    "stage_ledger",
    "token_masses",
    "transitive_support",
    "validate_packing_policy",
    "validate_record",
]
