"""Types, schedules, and identity validation for the integrated K3 oracle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
from typing import Sequence

from .reference_schedule import (
    channel_schedule as channel_schedule,
    depth_partition as depth_partition,
    full_scale_schedule_summary as full_scale_schedule_summary,
    terminal_global_schedule as terminal_global_schedule,
    validate_schedule as validate_schedule,
)


Vector = tuple[Fraction, ...]
Grid = tuple[Vector, ...]

MODEL_WIDTH = 8
VOCABULARY = 12
ROUTED_EXPERTS = 4
SELECTED_EXPERTS = 2
SHARED_EXPERTS = 1
GLOBAL_RECORD_WIDTH = 2
TOY_BLOCK_SIZE = 4
TOY_SCHEDULE = ("K", "K", "K", "G", "K", "K", "K", "G", "G")
RUNNING_LABELS = (
    "vis:red-circle-nw",
    "vis:red-circle-ne",
    "vis:blue-square-sw",
    "vis:blue-square-se",
    "text:Which",
    "text:shape",
    "text:is",
    "text:below",
    "text:red",
    "text:circle?",
)
DECODE_LABEL = "text:square"


def f(numerator: int | Fraction, denominator: int = 1) -> Fraction:
    """Construct a rational value without accidental binary conversion."""

    return Fraction(numerator, denominator)


@dataclass(frozen=True)
class TokenRow:
    """One model-width row with immutable origin metadata."""

    label: str
    position: int
    modality: str
    media_id: int | None
    values: Vector


@dataclass(frozen=True)
class RouteRecord:
    """One tagged routed-expert execution record."""

    origin: str
    position: int
    layer: int
    expert: int
    ordinal: int
    weight: Fraction


@dataclass(frozen=True)
class KdaLayerState:
    """Fixed-extent recurrent state owned by one toy KDA layer."""

    layer: int
    endpoint: int
    prefix_identity: str
    memory: Vector


@dataclass(frozen=True)
class GlobalLayerState:
    """Prefix-growing compact logical records for one toy global layer."""

    layer: int
    endpoint: int
    prefix_identity: str
    records: tuple[Vector, ...]


@dataclass(frozen=True)
class PrefixState:
    """Committed accepted-prefix state for every sequence mixer."""

    schedule: tuple[str, ...]
    endpoint: int
    accepted_labels: tuple[str, ...]
    kda: tuple[KdaLayerState, ...]
    global_layers: tuple[GlobalLayerState, ...]


@dataclass(frozen=True)
class DepthState:
    """Ephemeral Block-AttnRes history for one model forward."""

    forward_id: str
    embedding: Grid
    completed: tuple[Grid, ...] = ()
    completed_points: tuple[tuple[int, ...], ...] = ()
    partial: Grid | None = None
    partial_points: tuple[int, ...] = ()


@dataclass(frozen=True)
class LayerCheckpoint:
    """Auditable state immediately after one decoder layer."""

    layer: int
    attention_type: str
    channel_type: str
    pre_attention_sources: tuple[str, ...]
    pre_channel_sources: tuple[str, ...]
    attention_output: Grid
    channel_output: Grid
    routes: tuple[RouteRecord, ...]
    depth_points_after_layer: tuple[int, ...]
    completed_depth_groups: int


@dataclass(frozen=True)
class ForwardTrace:
    """One prefill or continuation through the integrated toy backbone."""

    forward_id: str
    input_labels: tuple[str, ...]
    start_endpoint: int
    checkpoints: tuple[LayerCheckpoint, ...]
    terminal_sources: tuple[str, ...]
    terminal_grid: Grid
    logits: Grid
    prefix_state: PrefixState
    routes: tuple[RouteRecord, ...]


def token_vector(position: int) -> Vector:
    """Create one deterministic inspectable width-eight embedding."""

    if position <= 0:
        raise ValueError("positions are one-indexed")
    return tuple(
        f(((position * (coordinate + 3) + 5 * coordinate) % 23) - 11, 10)
        for coordinate in range(MODEL_WIDTH)
    )


def running_tokens() -> tuple[TokenRow, ...]:
    """Return the four-visual/six-text accepted input rows."""

    rows = []
    for position, label in enumerate(RUNNING_LABELS, start=1):
        visual = label.startswith("vis:")
        rows.append(
            TokenRow(
                label=label,
                position=position,
                modality="visual" if visual else "text",
                media_id=1 if visual else None,
                values=token_vector(position),
            )
        )
    return tuple(rows)


def decode_token() -> TokenRow:
    """Return the accepted position-eleven token row."""

    return TokenRow(
        label=DECODE_LABEL,
        position=11,
        modality="text",
        media_id=None,
        values=token_vector(11),
    )


def splice_running_prefix(*, swap_visual_halves: bool = False) -> tuple[TokenRow, ...]:
    """Construct the integrated prefix and optionally inject an identity fault."""

    tokens = list(running_tokens())
    if swap_visual_halves:
        reordered = tokens[2:4] + tokens[0:2]
        tokens[:4] = [
            replace(token, position=position)
            for position, token in enumerate(reordered, start=1)
        ]
    return tuple(tokens)


def _zero_vector() -> Vector:
    return (f(0),) * MODEL_WIDTH


def prefix_identity(labels: Sequence[str]) -> str:
    """Return a deterministic identity for one ordered accepted prefix."""

    digest = sha256()
    for label in labels:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def empty_prefix_state(schedule: Sequence[str] = TOY_SCHEDULE) -> PrefixState:
    """Create prefix-zero sequence state for a validated schedule."""

    typed = validate_schedule(schedule)
    identity = prefix_identity(())
    return PrefixState(
        schedule=typed,
        endpoint=0,
        accepted_labels=(),
        kda=tuple(
            KdaLayerState(layer, 0, identity, _zero_vector())
            for layer, kind in enumerate(typed, start=1)
            if kind == "K"
        ),
        global_layers=tuple(
            GlobalLayerState(layer, 0, identity, ())
            for layer, kind in enumerate(typed, start=1)
            if kind == "G"
        ),
    )


def _validate_vector(vector: Sequence[Fraction], name: str) -> None:
    if len(vector) != MODEL_WIDTH:
        raise ValueError(f"{name} must have width {MODEL_WIDTH}")


def _validate_grid(grid: Sequence[Sequence[Fraction]], name: str) -> None:
    if not grid:
        raise ValueError(f"{name} must contain at least one token row")
    for row in grid:
        _validate_vector(row, name)


def _validate_kda_states(
    state: PrefixState, expected_owners: tuple[int, ...], identity: str
) -> None:
    """Validate every fixed-extent state against one accepted prefix."""

    if tuple(item.layer for item in state.kda) != expected_owners:
        raise ValueError("KDA state owners do not match the schedule")
    for kda_state in state.kda:
        if kda_state.endpoint != state.endpoint:
            raise ValueError("KDA state endpoint is stale or foreign")
        if kda_state.prefix_identity != identity:
            raise ValueError("KDA state belongs to another accepted prefix")
        _validate_vector(kda_state.memory, "KDA state")


def _validate_global_states(
    state: PrefixState, expected_owners: tuple[int, ...], identity: str
) -> None:
    """Validate every growing state against one accepted prefix."""

    if tuple(item.layer for item in state.global_layers) != expected_owners:
        raise ValueError("global state owners do not match the schedule")
    for global_state in state.global_layers:
        if (
            global_state.endpoint != state.endpoint
            or len(global_state.records) != state.endpoint
        ):
            raise ValueError("global records do not match the accepted endpoint")
        if global_state.prefix_identity != identity:
            raise ValueError("global state belongs to another accepted prefix")
        for record in global_state.records:
            if len(record) != GLOBAL_RECORD_WIDTH:
                raise ValueError("global logical record has the wrong width")


def validate_prefix_state(state: PrefixState) -> None:
    """Validate schedule ownership and one committed endpoint."""

    typed = validate_schedule(state.schedule)
    if state.endpoint != len(state.accepted_labels):
        raise ValueError("prefix endpoint and accepted-label identity disagree")
    if any(not isinstance(label, str) or not label for label in state.accepted_labels):
        raise ValueError("accepted-prefix labels must be nonempty strings")
    identity = prefix_identity(state.accepted_labels)
    expected_kda = tuple(
        layer for layer, kind in enumerate(typed, start=1) if kind == "K"
    )
    expected_global = tuple(
        layer for layer, kind in enumerate(typed, start=1) if kind == "G"
    )
    _validate_kda_states(state, expected_kda, identity)
    _validate_global_states(state, expected_global, identity)


def validate_input_segment(tokens: Sequence[TokenRow], state: PrefixState) -> None:
    """Validate origin, position, and width before a forward starts."""

    if not tokens:
        raise ValueError("a forward needs at least one new row")
    validate_prefix_state(state)
    for offset, token in enumerate(tokens, start=1):
        expected_position = state.endpoint + offset
        if token.position != expected_position:
            raise ValueError("token position does not continue the committed endpoint")
        if not token.label or token.modality not in {"visual", "text"}:
            raise ValueError("token origin metadata is incomplete")
        if token.modality == "visual" and token.media_id is None:
            raise ValueError("visual token lacks media identity")
        if token.modality == "text" and token.media_id is not None:
            raise ValueError("text token must not inherit a media identity")
        _validate_vector(token.values, "token embedding")


def validate_origin_sequence(
    tokens: Sequence[TokenRow], expected_labels: Sequence[str]
) -> None:
    """Reject a shape-valid segment whose ordered origins are not expected."""

    observed = tuple(token.label for token in tokens)
    expected = tuple(expected_labels)
    if observed != expected:
        raise ValueError("token origins are missing, duplicated, or reordered")


__all__ = [
    "DECODE_LABEL",
    "GLOBAL_RECORD_WIDTH",
    "MODEL_WIDTH",
    "ROUTED_EXPERTS",
    "RUNNING_LABELS",
    "SELECTED_EXPERTS",
    "SHARED_EXPERTS",
    "TOY_BLOCK_SIZE",
    "TOY_SCHEDULE",
    "VOCABULARY",
    "DepthState",
    "ForwardTrace",
    "GlobalLayerState",
    "Grid",
    "KdaLayerState",
    "LayerCheckpoint",
    "PrefixState",
    "RouteRecord",
    "TokenRow",
    "Vector",
    "_validate_grid",
    "_validate_vector",
    "_zero_vector",
    "channel_schedule",
    "decode_token",
    "depth_partition",
    "empty_prefix_state",
    "f",
    "full_scale_schedule_summary",
    "prefix_identity",
    "running_tokens",
    "splice_running_prefix",
    "terminal_global_schedule",
    "token_vector",
    "validate_input_segment",
    "validate_origin_sequence",
    "validate_prefix_state",
    "validate_schedule",
]
