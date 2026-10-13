"""A tiny studio policy: exact arithmetic, genuine sampling, and explicit updates.

All dimensions, rewards, probabilities, and optimizer choices are illustrative.
This program uses only the Python standard library and never runs a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
import random
from typing import Sequence


STUDIO_LENGTHS = (8, 12, 8, 20)
STUDIO_REWARDS = (1.0, 1.0, 0.0, 1.0)
OBSERVATION_OFFSET = math.log(9.0)
Gradient = tuple[float, float]


def sigmoid(value: float) -> float:
    """Evaluate a logistic probability without overflowing its exponential."""
    if not math.isfinite(value):
        raise ValueError("logits must be finite")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class Policy:
    """Two learned logits; the fixed observation changes the answer history."""

    u: float = 0.0
    v: float = -math.log(3.0)

    def probabilities(self) -> tuple[float, float, float]:
        """Return inspect, direct-blue, and inspected-blue probabilities."""
        return sigmoid(self.u), sigmoid(self.v), sigmoid(self.v + OBSERVATION_OFFSET)


@dataclass(frozen=True)
class Episode:
    """A complete route/answer sample on the fixed studio canvas."""

    inspect: bool
    blue: bool


def score(episode: Episode) -> float:
    """Use the declared answer-only rule, which does not require tool use."""
    return float(episode.blue)


def sample(policy: Policy, generator: random.Random) -> Episode:
    """Draw two actual choices; inspection deterministically returns the boxes."""
    q, direct, inspected = policy.probabilities()
    inspect = generator.random() < q
    blue = generator.random() < (inspected if inspect else direct)
    return Episode(inspect, blue)


def action_logprobs(policy: Policy, episode: Episode) -> tuple[float, float]:
    """Evaluate only assistant choices, conditioned on the saved history."""
    q, direct, inspected = policy.probabilities()
    answer = inspected if episode.inspect else direct
    route_probability = q if episode.inspect else 1 - q
    answer_probability = answer if episode.blue else 1 - answer
    if min(route_probability, answer_probability) <= 0:
        raise ValueError("this floating-point policy lacks support for the episode")
    return math.log(route_probability), math.log(answer_probability)


def masked_log_probability(
    logprobs: Sequence[float | None], mask: Sequence[int]
) -> float:
    """Exclude external observations; a missing assistant likelihood is invalid."""
    if not logprobs or len(logprobs) != len(mask):
        raise ValueError("log probabilities and mask need matching nonempty lengths")
    total = 0.0
    for value, owned in zip(logprobs, mask, strict=True):
        if owned not in (0, 1):
            raise ValueError("mask entries must be zero or one")
        if owned:
            if value is None or not math.isfinite(value) or value > 0:
                raise ValueError("assistant log probabilities must be finite and <= 0")
            total += value
    if not any(mask):
        raise ValueError("a complete policy example needs an assistant choice")
    return total


def score_gradient(policy: Policy, episode: Episode) -> Gradient:
    """Differentiate the complete action log likelihood at retained histories."""
    q, direct, inspected = policy.probabilities()
    answer = inspected if episode.inspect else direct
    return float(episode.inspect) - q, float(episode.blue) - answer


def expected_reward(policy: Policy) -> float:
    """Direct finite expectation, used as an oracle and optional fixed baseline."""
    q, direct, inspected = policy.probabilities()
    return q * inspected + (1 - q) * direct


def exact_gradient(policy: Policy) -> Gradient:
    """Differentiate the finite expectation directly, without sampled scores."""
    q, direct, inspected = policy.probabilities()
    return (
        q * (1 - q) * (inspected - direct),
        q * inspected * (1 - inspected) + (1 - q) * direct * (1 - direct),
    )


def enumerated_gradient(policy: Policy, baseline: float = 0.0) -> Gradient:
    """Weight all four episode score contributions by their probabilities."""
    totals = [0.0, 0.0]
    for inspect, blue in product((False, True), repeat=2):
        episode = Episode(inspect, blue)
        probability = math.exp(sum(action_logprobs(policy, episode)))
        coefficient = score(episode) - baseline
        for index, component in enumerate(score_gradient(policy, episode)):
            totals[index] += probability * coefficient * component
    return totals[0], totals[1]


def policy_loss(policy: Policy, episodes: Sequence[Episode], baseline: float) -> float:
    """Mean over episodes, sum over action tokens; baseline is held fixed."""
    if not episodes or not math.isfinite(baseline):
        raise ValueError("need a nonempty batch and a finite frozen baseline")
    return -sum(
        (score(episode) - baseline) * sum(action_logprobs(policy, episode))
        for episode in episodes
    ) / len(episodes)


def batch_gradient(
    policy: Policy, episodes: Sequence[Episode], baseline: float
) -> Gradient:
    """Return the ascent gradient; collect the elementary batch at this policy."""
    if not episodes or not math.isfinite(baseline):
        raise ValueError("need a nonempty batch and a finite frozen baseline")
    totals = [0.0, 0.0]
    for episode in episodes:
        coefficient = score(episode) - baseline
        for index, component in enumerate(score_gradient(policy, episode)):
            totals[index] += coefficient * component / len(episodes)
    return totals[0], totals[1]


def update(policy: Policy, gradient: Gradient, step_size: float) -> Policy:
    """Apply one ascent step, leaving the immutable behavior policy intact."""
    if step_size <= 0 or not all(map(math.isfinite, (*gradient, step_size))):
        raise ValueError("need finite gradients and a finite positive step size")
    return Policy(
        policy.u + step_size * gradient[0], policy.v + step_size * gradient[1]
    )


def group_coefficients(
    rewards: Sequence[float | None],
    *,
    expected_count: int,
    leave_one_out: bool = False,
    standardize: bool = False,
) -> tuple[float, ...]:
    """Compare a complete group; standardization uses population variance."""
    if expected_count < 2 or len(rewards) != expected_count:
        raise ValueError("need every member of a declared group of size >= 2")
    if any(value is None or not math.isfinite(value) for value in rewards):
        raise ValueError("missing or nonfinite rewards are not zero-reward failures")
    if leave_one_out and standardize:
        raise ValueError("choose one named baseline/normalization construction")
    values = tuple(float(value) for value in rewards if value is not None)
    mean = sum(values) / expected_count
    centered = tuple(value - mean for value in values)
    if leave_one_out:
        return tuple(
            value * expected_count / (expected_count - 1) for value in centered
        )
    if standardize:
        deviation = math.sqrt(sum(value * value for value in centered) / expected_count)
        return tuple(value / deviation for value in centered) if deviation else centered
    return centered


def sequence_ratio(current: Sequence[float], behavior: Sequence[float]) -> float:
    """Multiply action ratios through a difference of complete log likelihoods."""
    if not current or len(current) != len(behavior):
        raise ValueError("current and behavior likelihoods must align")
    if any(not math.isfinite(value) or value > 0 for value in (*current, *behavior)):
        raise ValueError("finite log probabilities required on common support")
    return math.exp(sum(current) - sum(behavior))


def ppo_score(ratio: float, advantage: float, epsilon: float = 0.2) -> float:
    """PPO's reward surrogate, not a projection of model probabilities."""
    if (
        ratio <= 0
        or not 0 < epsilon < 1
        or not all(map(math.isfinite, (ratio, advantage, epsilon)))
    ):
        raise ValueError("need positive ratio, finite advantage, and 0 < epsilon < 1")
    clipped = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped * advantage)


def ppo_ratio_derivative(ratio: float, advantage: float, epsilon: float = 0.2) -> float:
    """Choose zero at the active clipping kink; other points have unique slopes."""
    ppo_score(ratio, advantage, epsilon)
    if (advantage > 0 and ratio >= 1 + epsilon) or (
        advantage < 0 and ratio <= 1 - epsilon
    ):
        return 0.0
    return advantage


def categorical_kl(current: Sequence[float], reference: Sequence[float]) -> float:
    """Compute KL(current || reference) exactly on a finite common support."""
    if not current or len(current) != len(reference):
        raise ValueError("distributions must align")
    for distribution in (current, reference):
        if any(not math.isfinite(value) or value < 0 for value in distribution):
            raise ValueError("invalid probability")
        if not math.isclose(sum(distribution), 1.0):
            raise ValueError("probabilities must sum to one")
    if any(p > 0 and q == 0 for p, q in zip(current, reference, strict=True)):
        raise ValueError("reference lacks current-policy support")
    return sum(
        p * math.log(p / q) for p, q in zip(current, reference, strict=True) if p > 0
    )


def learning_demo(seed: int = 2026, steps: int = 20, batch_size: int = 32) -> Policy:
    """Fresh on-policy batches; the exact starting value is a frozen baseline."""
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    generator = random.Random(seed)
    policy = Policy()
    for _ in range(steps):
        baseline = expected_reward(policy)
        episodes = [sample(policy, generator) for _ in range(batch_size)]
        policy = update(policy, batch_gradient(policy, episodes, baseline), 0.4)
    return policy


def main() -> None:
    """Print the worked episode separately from fixed and sampled demonstrations."""
    behavior = Policy()
    episode = sample(behavior, random.Random(7))
    gradient = batch_gradient(behavior, [episode], 0.5)
    current = update(behavior, gradient, 0.4)
    learned = learning_demo()
    print(
        json.dumps(
            {
                "scope": "illustrative finite policy; no K3 or generalization result",
                "fixed_group": {
                    "tokens": sum(STUDIO_LENGTHS),
                    "mean": sum(STUDIO_REWARDS) / 4,
                    "centered": group_coefficients(STUDIO_REWARDS, expected_count=4),
                },
                "seed_7_episode": {"inspect": episode.inspect, "blue": episode.blue},
                "sampled_loss": policy_loss(behavior, [episode], 0.5),
                "sampled_ascent_gradient": gradient,
                "exact_expected_gradient": exact_gradient(behavior),
                "updated_logits": [current.u, current.v],
                "updated_probabilities": current.probabilities(),
                "updated_expected_reward": expected_reward(current),
                "episode_ratio_to_behavior": sequence_ratio(
                    action_logprobs(current, episode),
                    action_logprobs(behavior, episode),
                ),
                "fresh_batch_demo": {
                    "seed": 2026,
                    "steps": 20,
                    "batch_size": 32,
                    "final_expected_reward": expected_reward(learned),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
