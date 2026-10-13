"""Deterministic schema and decoder checks CH26-D01 through CH26-D05."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

from .decoder import (
    FiniteLanguageRecognizer,
    constrained_distribution,
    evaluate_token_trace,
)
from .fixture import require
from .schema import (
    LEVELS,
    STUDIO_RESPONSE_SCHEMA,
    CanonicalResult,
    Diagnostic,
    canonicalize_to_fixed_point,
    finite_semantic_relation,
    validate_schema_profile,
)


JSON = Any


def schema_profile_check() -> dict[str, JSON]:
    """Return diagnostics for the schema profile invariant."""
    unsupported = {"type": "string", "oneOf": [{"enum": ["a"]}]}
    contradiction = {"type": "integer", "minimum": 4, "maximum": 2}
    payload = {
        "studio": {
            level: not validate_schema_profile(STUDIO_RESPONSE_SCHEMA, level)
            for level in LEVELS
        },
        "unsupported": {
            level: not validate_schema_profile(unsupported, level) for level in LEVELS
        },
        "contradiction": {
            level: not validate_schema_profile(contradiction, level) for level in LEVELS
        },
    }
    require(all(payload["studio"].values()), "fixture schema must pass every level")
    require(
        payload["unsupported"]
        == {"loose": True, "lite": False, "strict": False, "ultra": False},
        "unsupported keyword expectations drifted",
    )
    require(
        payload["contradiction"]
        == {"loose": True, "lite": True, "strict": False, "ultra": False},
        "validation-level expectations drifted",
    )
    return payload


def _oscillate(candidate: Mapping[str, JSON]) -> CanonicalResult:
    return CanonicalResult(
        {"flip": not candidate.get("flip", False)},
        Diagnostic("success", ("injected-oscillation",)),
    )


def canonicalization_check() -> dict[str, JSON]:
    """Return diagnostics for the canonicalization invariant."""
    original = deepcopy(STUDIO_RESPONSE_SCHEMA)
    fixed = canonicalize_to_fixed_point(original)
    bounded = canonicalize_to_fixed_point(original, max_attempts=1)
    warning = canonicalize_to_fixed_point({"type": "string", "oneOf": []})
    oscillating = canonicalize_to_fixed_point(
        {"flip": False}, max_attempts=3, step=_oscillate
    )
    input_unmodified = original == STUDIO_RESPONSE_SCHEMA
    payload = {
        "diagnostic": fixed.diagnostic.severity,
        "diagnostic_codes": list(fixed.diagnostic.codes),
        "fixed_point_status": fixed.status,
        "attempts": fixed.attempts,
        "input_unmodified": input_unmodified,
        "one_attempt_status": bounded.status,
        "warning_status": warning.status,
        "oscillation_status": oscillating.status,
    }
    require(fixed.status == "stable", "accepted canonicalization must stabilize")
    require(fixed.attempts == 2, "fixture should expose first-pass reordering")
    require(input_unmodified, "canonicalization mutated its input")
    require(bounded.status == "max-attempts", "attempt bound was not enforced")
    require(warning.status == "rejected", "warning candidate was consumed")
    require(
        oscillating.status == "max-attempts",
        "oscillation escaped the fixed-point bound",
    )
    return payload


def semantic_relation_check() -> dict[str, JSON]:
    """Return diagnostics for the semantic relation invariant."""
    relation = finite_semantic_relation(
        {"type": "integer", "maximum": 12}, {}, [-1, 12, 13]
    )
    require(relation["relation"] == "weakened-on-domain", "relation drifted")
    require(13 not in relation["original_accepts"], "witness passed original")
    require(13 in relation["candidate_accepts"], "witness missed candidate")
    return {
        **relation,
        "witness": 13,
        "byte_stability_is_not_equivalence": True,
    }


def recognizer_fixture() -> tuple[FiniteLanguageRecognizer, dict[str, bytes]]:
    """Return the finite recognizer and its token-byte vocabulary."""
    recognizer = FiniteLanguageRecognizer((b'{"ok":true}', b'{"ok":false}'))
    vocabulary = {
        "v1": b'{"ok":',
        "v2": b"true}",
        "v3": b"false}",
        "v4": b"true",
        "v5": b"false",
        "v6": b"}",
        "v7": b"x",
    }
    return recognizer, vocabulary


def legal_token_check() -> dict[str, JSON]:
    """Return diagnostics for the legal token invariant."""
    recognizer, vocabulary = recognizer_fixture()
    prefix = b'{"ok":'
    probabilities = {
        "v2": 0.30,
        "v3": 0.10,
        "v4": 0.20,
        "v5": 0.10,
        "v6": 0.15,
        "v7": 0.15,
    }
    distribution = constrained_distribution(
        recognizer, prefix, vocabulary, probabilities
    )
    payload = {
        "empty_prefix": list(recognizer.legal_tokens(b"", vocabulary)),
        "value_prefix": list(recognizer.legal_tokens(prefix, vocabulary)),
        "true_prefix": list(recognizer.legal_tokens(b'{"ok":true', vocabulary)),
        "distribution": distribution,
        "boundary_spanning": ["v2", "v3"],
    }
    require(payload["empty_prefix"] == ["v1"], "initial mask drifted")
    require(
        payload["value_prefix"] == ["v2", "v3", "v4", "v5"],
        "value-prefix mask drifted",
    )
    require(math.isclose(sum(distribution.values()), 1.0), "mask did not normalize")
    require(math.isclose(distribution["v2"], 3 / 7), "v2 mass drifted")
    return payload


def decoder_failure_check() -> dict[str, JSON]:
    """Return diagnostics for the decoder failure invariant."""
    recognizer, vocabulary = recognizer_fixture()
    failures: list[str] = []
    for prefix, probabilities in (
        (b"x", {name: 1.0 for name in vocabulary}),
        (b'{"ok":', {name: 0.0 for name in vocabulary}),
    ):
        try:
            constrained_distribution(recognizer, prefix, vocabulary, probabilities)
        except ValueError as error:
            failures.append(str(error))
    outcomes = {
        "dead": evaluate_token_trace(recognizer, (b"x",), end_requested=True),
        "premature": evaluate_token_trace(
            recognizer, (b'{"ok":', b"tru"), end_requested=True
        ),
        "interrupted": evaluate_token_trace(
            recognizer, (b'{"ok":', b"tru"), end_requested=False
        ),
        "accepted": evaluate_token_trace(
            recognizer, (b'{"ok":', b"true}"), end_requested=True
        ),
    }
    require(len(failures) == 2, "both decoder mask failures must surface")
    require(outcomes["dead"].status == "dead-prefix", "dead prefix accepted")
    require(outcomes["premature"].status == "premature-end", "early stop accepted")
    require(outcomes["interrupted"].status == "interrupted", "interrupt collapsed")
    require(outcomes["accepted"].status == "accepted", "complete value rejected")
    return {
        "failures": failures,
        "outcomes": {name: outcome.status for name, outcome in outcomes.items()},
    }
