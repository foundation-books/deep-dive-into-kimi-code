# Chapter 22 semantic reference

This dependency-free companion reproduces the finite contracts in
**Rewards, Verifiers, and Agentic Generative Reward Models**:

- four verifier dispositions and hard-gate precedence;
- exact, tolerance-aware, visual, and studio relation checks;
- hard-first composition with typed missingness;
- public--hidden proxy gaps and hacking witnesses;
- pairwise records, cycles, ties, abstentions, and order retention;
- rubric, scorepad, and strict visible-verbosity boundaries; and
- the complete four-candidate studio handoff.

The revised fixture uses `studio-profile-v2`: C's downstream cells are
`not_evaluated`, D's residual is `inapplicable`, and timeouts are `unresolved`.
The relation vocabulary is `below`, `above`, or `level`, including equality.
`compare_candidates` applies the strict 8/16/18 verbosity branches before
calling a supplied judge. It preserves automatic and learned decision sources
and leaves two crossings or a timeout unresolved.

The residual inputs 0.9 and 0.7 are stipulated arithmetic inputs, not computed
judge outputs. `studio-admission-v1` admits C's negative evidence-violation
outcome only with the stipulated valid trajectory, sampling-probability, and
slot provenance. `centered_studio_group` requires all four original eligible
slots; it computes mean 0.2675 and coefficients
`(0.9125, 0.6225, -1.2675, -0.2675)` without dropping blocked or missing slots.

The added investigations use separate, constructed fixtures:

- `math_investigation.py` contrasts extraction fallback, directional
  relation/interval comparison, and candidate versus evaluator faults.
- `state_action_investigation.py` scores three small trajectories under final
  state, required-action inclusion, and selected products. A callback check
  distinguishes a zero product from a judge that was never called.
- `judge_investigation.py` validates ordered rubric verdicts and contrasts
  shared versus artifact-only judge workspaces. Its evidence-link check is a
  book extension beyond structural verdict validation.
- `reward_bridge.py` models last-response callback input and routes eligible
  scalars separately from retry, defer, or quarantine actions. Its bounded
  child process runs only the local rational fixture in `math_worker.py`.

The sources are immutable comparison anchors: Math-Verify
`ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b`, tau2-bench
`b7ea9074c1cba482b30687fecdb5c8425fd6f619`, and the book's existing
Prime Intellect and veRL source pins cited in the book.
The dependency-free checks do not import or execute those projects. In
particular, the thread-signal and child-process checks demonstrate a Python
execution boundary, not a measured veRL plus Math-Verify integration.

It is a semantic teaching reference. It is not Moonshot AI code, a learned
judge, a model evaluation, or a production verifier.

Run the unit tests from the companion-code root (`code/`):

    python3 -B _maintenance/chapter_test_runner.py \
      rewards_verifiers_agentic_grm/test_rewards_verifiers_agentic_grm_reference.py

Print the current seven deterministic records from `code/`, without refreshing
stored experimental evidence:

    python3 -B -c 'import json; from rewards_verifiers_agentic_grm.reference import all_checks; print(json.dumps(all_checks(), indent=2))'

## October 2026 reward-error example

`reward_error_update.py` computes the three-response raw centered coefficients,
mean log-softmax derivative, finite differences, SGD logits and normalized
probabilities. `artifact_candidates.py` defines varied count packets, correct
and pooled-metric deliveries, a whitelisted judge view and independent artifact
acceptance. The registered chapter test also checks protected-field rejection,
missing/stale verdicts, candidate-order mapping and gate failures. These are
dependency-free checks; they do not load a judge or train a model.

The complete source repository separately retains the bounded frozen judge
diagnostic and generated table under `experiments/rewards_verifiers_agentic_grm/`.
Essential task, failure and update explanations remain in Chapter 22.
