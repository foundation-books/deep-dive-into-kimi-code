"""Finite DSpark teaching extensions to the Chapter 45 reference.

No model, tensor framework or serving engine is executed here. Matrices use
value-first layout. Records are valid only for the prefix that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import exp, isfinite, prod
from typing import Mapping, Sequence

from .reference import Distribution, inverse_cdf, overlap_mass, positive_residual


Matrix = tuple[tuple[float, ...], ...]


def markov_proposal_law(
    base_logits: Mapping[str, float],
    transition_bias: Mapping[str, float],
    temperature: float = 1.0,
    allowed_tokens: Sequence[str] | None = None,
) -> dict[str, float]:
    """Normalize the coupled logits after temperature and an explicit mask."""
    if not base_logits or set(base_logits) != set(transition_bias):
        raise ValueError("base and transition vocabularies must agree")
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("stochastic temperature must be finite and positive")
    if any(not isfinite(x) for x in (*base_logits.values(), *transition_bias.values())):
        raise ValueError("fixture logits must be finite")
    allowed = set(base_logits if allowed_tokens is None else allowed_tokens)
    if not allowed or not allowed <= set(base_logits):
        raise ValueError("mask must retain a nonempty vocabulary subset")
    logits = {t: (base_logits[t] + transition_bias[t]) / temperature for t in allowed}
    maximum = max(logits.values())
    weights = {
        t: exp(logits[t] - maximum) if t in allowed else 0.0 for t in base_logits
    }
    total = sum(weights.values())
    return {t: weight / total for t, weight in weights.items()}


def markov_draft(
    base_block: Sequence[Mapping[str, float]],
    transitions: Mapping[str, Mapping[str, float]],
    anchor: str,
    draws: Sequence[float],
) -> tuple[tuple[str, ...], tuple[dict[str, float], ...]]:
    """Sample a fixed block left to right while retaining each actual law."""
    if len(base_block) != len(draws):
        raise ValueError("one independent draw is required per proposal")
    previous = anchor
    tokens, laws = [], []
    for base, draw in zip(base_block, draws, strict=True):
        law = markov_proposal_law(base, transitions[previous])
        previous = inverse_cdf(law, draw)
        tokens.append(previous)
        laws.append(law)
    return tuple(tokens), tuple(laws)


def selected_proposal_output_law(
    target: Distribution,
    actual_draft: Distribution,
    admit: Mapping[str, float],
    verifier_draft: Distribution | None = None,
) -> dict[str, float]:
    """Enumerate an admitted proposal, or a fresh target draw if not admitted.

    Token-dependent admission and a wrong verifier law are intentional negative
    controls. A constant admission probability and the actual law are controls.
    Acceptance and correction randomness are independent in this finite model.
    """
    overlap_mass(target, actual_draft)
    assumed = actual_draft if verifier_draft is None else verifier_draft
    alpha = overlap_mass(target, assumed)
    if set(admit) != set(target) or any(not 0 <= x <= 1 for x in admit.values()):
        raise ValueError("admission probabilities must cover the vocabulary")
    residual = positive_residual(target, assumed) if alpha < 1.0 else dict(target)
    output = dict.fromkeys(target, 0.0)
    for token, mass in actual_draft.items():
        if mass == 0:
            continue
        if assumed[token] <= 0:
            raise ValueError("verifier assigns zero mass to a possible proposal")
        acceptance = min(1.0, target[token] / assumed[token])
        output[token] += mass * admit[token] * acceptance
        for emitted in output:
            output[emitted] += mass * (
                (1 - admit[token]) * target[emitted]
                + admit[token] * (1 - acceptance) * residual[emitted]
            )
    return output


def prefix_survival(conditional_confidence: Sequence[float]) -> tuple[float, ...]:
    """Convert calibrated conditional estimates to cumulative prefix estimates."""
    if any(not isfinite(c) or not 0 <= c <= 1 for c in conditional_confidence):
        raise ValueError("confidence must be a finite probability")
    return tuple(
        prod(conditional_confidence[:j])
        for j in range(1, len(conditional_confidence) + 1)
    )


def predicted_output_rate(
    survival: Sequence[Sequence[float]], lengths: Sequence[int], step_cost: float
) -> tuple[float, float]:
    """Return nonterminal expected output and rate for a predeclared allocation."""
    if len(survival) != len(lengths) or not isfinite(step_cost) or step_cost <= 0:
        raise ValueError("require aligned requests and positive finite elapsed cost")
    output = float(len(survival))
    for estimates, length in zip(survival, lengths, strict=True):
        if not 0 <= length <= len(estimates):
            raise ValueError("verification length exceeds the proposed block")
        if any(not isfinite(s) or not 0 <= s <= 1 for s in estimates):
            raise ValueError("survival must be a finite probability")
        if any(a < b for a, b in pairwise(estimates)):
            raise ValueError("prefix survival must be non-increasing")
        output += sum(estimates[:length])
    return output, output / step_cost


@dataclass(frozen=True)
class CorrectionRecord:
    """Already computed value correction, key and diagonal decay for one input."""

    decay: tuple[float, ...]
    key: tuple[float, ...]
    correction: tuple[float, ...]


def kda_step_record(
    state: Matrix,
    decay: tuple[float, ...],
    key: tuple[float, ...],
    value: tuple[float, ...],
    beta: float,
) -> tuple[Matrix, CorrectionRecord]:
    """Compute u=beta(v-SDk), then SD+u k^T; key normalization is an input."""
    if not state or not key or len(decay) != len(key) or len(value) != len(state):
        raise ValueError("incompatible value-first shapes")
    if any(len(row) != len(key) for row in state):
        raise ValueError("state must have shape d_v by d_k")
    decayed = tuple(
        tuple(s * d for s, d in zip(row, decay, strict=True)) for row in state
    )
    correction = tuple(
        beta * (v - sum(s * k for s, k in zip(row, key, strict=True)))
        for v, row in zip(value, decayed, strict=True)
    )
    result = tuple(
        tuple(s + u * k for s, k in zip(row, key, strict=True))
        for row, u in zip(decayed, correction, strict=True)
    )
    return result, CorrectionRecord(decay, key, correction)


def reconstruct_kda(
    initial: Matrix, records: Sequence[CorrectionRecord], accepted_inputs: int
) -> Matrix:
    """Direct reverse-order reconstruction, without replaying correction work."""
    if not initial or not initial[0] or not 0 <= accepted_inputs <= len(records):
        raise ValueError("require a state and a valid accepted input count")
    width = len(initial[0])
    if any(len(row) != width for row in initial):
        raise ValueError("state must be rectangular")
    suffix_decay = [1.0] * width
    updates = [[0.0] * width for _ in initial]
    for record in reversed(records[:accepted_inputs]):
        if (
            len(record.decay) != width
            or len(record.key) != width
            or len(record.correction) != len(initial)
        ):
            raise ValueError("record does not match state dimensions")
        for row, correction in zip(updates, record.correction, strict=True):
            for column in range(width):
                row[column] += correction * record.key[column] * suffix_decay[column]
        for column in range(width):
            suffix_decay[column] *= record.decay[column]
    return tuple(
        tuple(s * d + u for s, d, u in zip(row, suffix_decay, update, strict=True))
        for row, update in zip(initial, updates, strict=True)
    )


def recover_convolution_history(
    before: Sequence[str], input_tokens: Sequence[str], consumed: int
) -> tuple[str, ...]:
    """Restore the last h consumed inputs; rejected suffix inputs cannot enter."""
    if not 0 <= consumed <= len(input_tokens):
        raise ValueError("consumed input count lies outside the window")
    history = tuple(before) + tuple(input_tokens[:consumed])
    return history[-len(before) :] if before else ()


@dataclass(frozen=True)
class WindowPositions:
    """A nonterminal fixed window with one incoming and one outgoing anchor."""

    draft_queries: int
    target_inputs: int
    emitted_tokens: int
    recovered_inputs: int
    consumed_frontier: int
    visible_frontier: int
    cache_boundaries: tuple[int, ...]


def window_positions(
    width: int, accepted: int, consumed_before: int = 0, cache_block: int = 4
) -> WindowPositions:
    """Map A accepted drafts to A+1 recovered inputs and one pending output.

    Excludes EOS and output caps: those use reference.stochastic_commit and
    actual emitted/consumed counts. The incoming anchor is already visible.
    """
    if (
        width < 1
        or not 0 <= accepted <= width
        or consumed_before < 0
        or cache_block < 1
    ):
        raise ValueError("invalid nonterminal window geometry")
    consumed = consumed_before + accepted + 1
    next_boundary = (consumed_before // cache_block + 1) * cache_block
    return WindowPositions(
        width,
        width + 1,
        accepted + 1,
        accepted + 1,
        consumed,
        consumed + 1,
        tuple(range(next_boundary, consumed + 1, cache_block)),
    )
