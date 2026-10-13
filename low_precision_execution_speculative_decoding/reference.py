"""Small correctness and accounting oracles for Chapter 45.

These functions model finite teaching distributions and state actions.  They
are not a production sampler, quantization kernel, or performance benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log2
from typing import Mapping, Sequence


Distribution = Mapping[str, float]


def _validate_distribution(distribution: Distribution) -> None:
    if not distribution:
        raise ValueError("distribution must not be empty")
    if any(not isfinite(value) or value < 0.0 for value in distribution.values()):
        raise ValueError("probabilities must be finite and nonnegative")
    if abs(sum(distribution.values()) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")


def overlap_mass(target: Distribution, draft: Distribution) -> float:
    """Return sum_v min(target[v], draft[v]) on a common vocabulary."""

    _validate_distribution(target)
    _validate_distribution(draft)
    if set(target) != set(draft):
        raise ValueError("target and draft vocabularies must match")
    return sum(min(target[token], draft[token]) for token in target)


def positive_residual(target: Distribution, draft: Distribution) -> dict[str, float]:
    """Return the normalized positive target-minus-draft residual."""

    overlap_mass(target, draft)
    raw = {token: max(target[token] - draft[token], 0.0) for token in target}
    normalizer = sum(raw.values())
    if normalizer <= 0.0:
        raise ValueError("positive residual is undefined when target equals draft")
    return {token: value / normalizer for token, value in raw.items()}


def exact_one_step_output_law(
    target: Distribution, draft: Distribution
) -> dict[str, float]:
    """Enumerate accepted mass plus rejected residual mass for each token."""

    alpha = overlap_mass(target, draft)
    if all(target[token] == draft[token] for token in target):
        return dict(target)
    residual = positive_residual(target, draft)
    return {
        token: min(target[token], draft[token]) + (1.0 - alpha) * residual[token]
        for token in target
    }


def greedy_commit(
    proposals: Sequence[str], target_choices: Sequence[str]
) -> tuple[tuple[str, ...], int]:
    """Commit matching draft prefix plus first target mismatch or bonus."""

    if len(target_choices) < len(proposals) + 1:
        raise ValueError("target choices must include a mismatch or bonus token")
    accepted = 0
    committed: list[str] = []
    for proposal, target in zip(
        proposals, target_choices[: len(proposals)], strict=True
    ):
        if proposal != target:
            committed.append(target)
            return tuple(committed), accepted
        committed.append(proposal)
        accepted += 1
    committed.append(target_choices[len(proposals)])
    return tuple(committed), accepted


@dataclass(frozen=True)
class StateAction:
    """Finite ledger for one speculative window's committed state action."""

    accepted_draft_tokens: int
    final_token_kind: str
    committed_tokens: int
    rejected_draft_tokens: int


def state_action(proposal_length: int, accepted_length: int) -> StateAction:
    """Return the bonus-included commit and rejected-suffix ledger."""

    if proposal_length < 0 or not 0 <= accepted_length <= proposal_length:
        raise ValueError("accepted length must lie inside the proposal")
    return StateAction(
        accepted_draft_tokens=accepted_length,
        final_token_kind="bonus"
        if accepted_length == proposal_length
        else "correction",
        committed_tokens=accepted_length + 1,
        rejected_draft_tokens=proposal_length - accepted_length,
    )


def normalized_speedup(
    mean_commit_length: float,
    draft_cost: float,
    verification_cost: float,
    repair_cost: float,
    ordinary_target_cost: float = 1.0,
) -> float:
    """Return the chapter's matched-work speculative speed ratio."""

    denominator = draft_cost + verification_cost + repair_cost
    if mean_commit_length <= 0.0 or ordinary_target_cost <= 0.0 or denominator <= 0.0:
        raise ValueError("commit length and costs must be positive")
    return mean_commit_length * ordinary_target_cost / denominator


E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_mxfp4(
    values: Sequence[float], scale: float
) -> tuple[int, tuple[int, ...]]:
    """Encode one finite 32-element fixture block, nearest-even and saturating.

    Scale selection and physical packing are deliberately caller choices.
    Nonfinite inputs are unsupported; this is not a general OCP converter.
    """

    if len(values) != 32 or not isfinite(scale) or scale <= 0:
        raise ValueError("require 32 values and a positive finite scale")
    exponent = log2(scale)
    if not exponent.is_integer() or not -127 <= exponent <= 127:
        raise ValueError("scale must be a finite E8M0 power of two")
    codes = []
    for value in values:
        if not isfinite(value):
            raise ValueError("fixture supports finite inputs only")
        normalized = min(abs(value) / scale, 6.0)
        magnitude = min(
            range(8),
            key=lambda code: (abs(E2M1_MAGNITUDES[code] - normalized), code % 2),
        )
        codes.append(magnitude | (8 if value < 0 else 0))
    return int(exponent) + 127, tuple(codes)


def decode_mxfp4(scale_code: int, codes: Sequence[int]) -> tuple[float, ...]:
    """Decode the finite fixture representation; no physical layout implied."""

    if not 0 <= scale_code <= 254 or len(codes) != 32:
        raise ValueError("require a finite E8M0 scale and 32 elements")
    if any(not 0 <= code <= 15 for code in codes):
        raise ValueError("E2M1 codes must fit in four bits")
    scale = 2.0 ** (scale_code - 127)
    return tuple(
        scale * E2M1_MAGNITUDES[code & 7] * (-1 if code & 8 else 1) for code in codes
    )


def inverse_cdf(law: Distribution, draw: float) -> str:
    """Use mapping insertion order and half-open CDF intervals."""

    _validate_distribution(law)
    if not 0.0 <= draw < 1.0:
        raise ValueError("uniform draw must lie in [0, 1)")
    cumulative = 0.0
    last_positive = ""
    for token, probability in law.items():
        cumulative += probability
        if probability > 0:
            last_positive = token
        if draw < cumulative:
            return token
    return last_positive  # Floating summation may finish just below one.


def stochastic_commit(
    proposals: Sequence[str],
    targets: Sequence[Distribution],
    drafts: Sequence[Distribution],
    acceptance_draws: Sequence[float],
    final_draw: float,
    output_cap: int,
    eos: str = "eos",
) -> tuple[tuple[str, ...], int]:
    """Trace a chain, assuming proposals were drawn from the supplied laws.

    Targets include the bonus row. Rows must be evaluated at aligned causal
    prefixes. Explicit draws are independent realizations supplied by a test.
    The returned suffix is eagerly materialized by the caller before commit.
    """

    if output_cap < 0:
        raise ValueError("output cap must be nonnegative")
    if output_cap == 0:
        return (), 0
    count = len(proposals)
    if (
        len(targets) != count + 1
        or len(drafts) != count
        or len(acceptance_draws) != count
    ):
        raise ValueError("unaligned proposal, law or draw rows")
    committed: list[str] = []
    for index, proposal in enumerate(proposals):
        target, draft = targets[index], drafts[index]
        overlap_mass(target, draft)
        draw = acceptance_draws[index]
        if not 0 <= draw < 1 or draft.get(proposal, 0) <= 0:
            raise ValueError("invalid acceptance draw or impossible proposal")
        if draw >= min(1.0, target[proposal] / draft[proposal]):
            committed.append(inverse_cdf(positive_residual(target, draft), final_draw))
            return tuple(committed), index
        committed.append(proposal)
        if proposal == eos or len(committed) == output_cap:
            return tuple(committed), index + 1
    committed.append(inverse_cdf(targets[-1], final_draw))
    return tuple(committed), count


def replay_scalar(
    initial: int, tokens: Sequence[str], embeddings: Mapping[str, int]
) -> int:
    """Consume committed tokens in the original scalar recurrence s'=2s+e(t)."""

    state = initial
    for token in tokens:
        state = 2 * state + embeddings[token]
    return state


@dataclass(frozen=True)
class StateIdentity:
    """Identity and eager materialization frontier for one state object."""

    request_id: str
    prefix_digest: str
    contract: str
    consumed_tokens: int


def _identity_key(identity: StateIdentity) -> tuple[str, str, str, int]:
    """Return every identity component that must match before publication."""

    return (
        identity.request_id,
        identity.prefix_digest,
        identity.contract,
        identity.consumed_tokens,
    )


def validate_state_bundle(
    owner: StateIdentity, objects: Sequence[StateIdentity]
) -> None:
    """Reject mismatched ownership/frontiers before any visible publication."""

    owner_key = _identity_key(owner)
    if any(_identity_key(item) != owner_key for item in objects):
        raise ValueError("state identity or materialization frontier mismatch")
