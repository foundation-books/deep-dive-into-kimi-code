"""Finite byte-language decoder fixture for Chapter 27."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class FiniteLanguageRecognizer:
    """Exact recognizer for a finite set of byte strings."""

    language: tuple[bytes, ...]

    def status(self, prefix: bytes) -> str:
        if prefix in self.language:
            return "complete"
        if any(word.startswith(prefix) for word in self.language):
            return "viable"
        return "dead"

    def legal_tokens(
        self, prefix: bytes, vocabulary: Mapping[str, bytes]
    ) -> tuple[str, ...]:
        return tuple(
            name
            for name, expansion in vocabulary.items()
            if self.status(prefix + expansion) != "dead"
        )

    def end_is_legal(self, prefix: bytes) -> bool:
        return prefix in self.language


@dataclass(frozen=True)
class DecodeOutcome:
    """Typed result of replaying exact token-byte expansions."""

    status: str
    prefix: bytes
    token_count: int


def evaluate_token_trace(
    recognizer: FiniteLanguageRecognizer,
    token_bytes: tuple[bytes, ...],
    *,
    end_requested: bool,
) -> DecodeOutcome:
    """Replay a token trace without treating a viable prefix as complete."""

    prefix = b""
    for index, expansion in enumerate(token_bytes, start=1):
        prefix += expansion
        if recognizer.status(prefix) == "dead":
            return DecodeOutcome("dead-prefix", prefix, index)
    if not end_requested:
        return DecodeOutcome("interrupted", prefix, len(token_bytes))
    if recognizer.end_is_legal(prefix):
        return DecodeOutcome("accepted", prefix, len(token_bytes))
    return DecodeOutcome("premature-end", prefix, len(token_bytes))


def constrained_distribution(
    recognizer: FiniteLanguageRecognizer,
    prefix: bytes,
    vocabulary: Mapping[str, bytes],
    probabilities: Mapping[str, float],
) -> dict[str, float]:
    """Mask and renormalize, surfacing empty and zero-mass legal sets."""

    legal = recognizer.legal_tokens(prefix, vocabulary)
    if not legal:
        raise ValueError("dead-prefix: empty legal-token set")
    denominator = sum(probabilities.get(name, 0.0) for name in legal)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("zero-or-nonfinite legal probability mass")
    return {
        name: probabilities.get(name, 0.0) / denominator
        for name in vocabulary
        if name in legal
    }
