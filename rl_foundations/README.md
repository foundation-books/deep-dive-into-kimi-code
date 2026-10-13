# RL foundations

This companion belongs to numbered Volume II Chapter 18, between Chapters
17 and 19. All policies and outcomes are teaching constructions; no K3 trainer
or model performance is reproduced.

From the companion root (`code/` in the book repository):

```sh
python3 -B -m rl_foundations.reference
python3 -B -m unittest rl_foundations.test_rl_foundations_reference
```

The program uses only the standard library. It needs no model, GPU, network,
test runner dependency, or dataset. `make smoke` also runs its registered test.

The fixed studio arithmetic has lengths `(8,12,8,20)` and answer-only rewards
`(1,1,0,1)`. A direct correct answer earns one under this evaluator; it does not
satisfy the earlier SFT task's stricter requirement for executed evidence.
Those four paths are not claimed to be draws from the tiny policy.

The separate tiny policy makes two choices: inspect/direct, then blue/red.
The inspect logit starts at 0; the blue logit starts at `-log(3)`. Inspection
returns the fixed correct boxes and adds a fixed `log(9)` offset to the answer
logit. There is no discount, inspection cost, sampling truncation, or model
update during tool execution. Temperature is one. The initial probabilities
are `q=1/2`, `p_direct=1/4`, `p_inspected=3/4`.

The output separates three objects:

- `fixed_group`: deterministic mean, centered coefficients, and action-token
  count; no sampling claim.
- `seed_7_episode` and the update fields: an actual sampled inspected-blue
  episode, frozen baseline `1/2`, loss `0.4904146265`, ascent gradient
  `(1/4,1/8)`, step size `0.4`, and exact post-update reward `0.5218582523`.
  `updated_probabilities` lists inspect, direct-blue, and inspected-blue in
  that order. `episode_ratio_to_behavior` retains the old denominator.
- `fresh_batch_demo`: 20 fresh batches of 32 episodes, seed 2026, with each
  batch collected at the policy being differentiated. The exact expected
  reward at collection is a fixed baseline for that batch. This is a toy
  stochastic learning demonstration, not a held-out generalization result.

The loss averages episodes and sums action log probabilities. Observations
are excluded from action likelihood terms while still changing subsequent
answer probabilities. A token-mean PPO comparison is a separate surrogate.
`ppo_score` covers the reward term only; `ppo_ratio_derivative` chooses zero
at the active clipping kink and tests the two one-sided slopes. Standardized
group coefficients use population variance and return zeros for equal rewards;
this fallback is explicitly illustrative.

The checks use exact enumeration, direct analytic differentiation, and central
differences as independent oracles. They exhaust all four tiny episodes and
all 16 size-four groups from the separate two-answer policy. They reject
missing group members and missing assistant probabilities. No stored evidence
campaign or benchmark result is created.
