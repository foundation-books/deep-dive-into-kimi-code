# Chapter 19 companion: long-CoT RL and localized updates

This dependency-free companion makes six Chapter 19 teaching surfaces
executable:

- complete same-prompt group validation, exact centering, translation
  invariance, and the equal-outcome null case;
- finite-support KL-regularized exponential tilting;
- token/sequence likelihood-ratio factorization and cancellation examples;
- the chapter-created inclusive log-ratio locality rule, boundary-held forward
  factor, restoring penalty, and derivative checks;
- total-token, per-trajectory, and reward-only active-token ledgers, including
  empty-denominator rejection; and
- a complete 48-token score and coefficient ledger, plus a separately
  normalized binary policy that performs one actual parameter step.
- a distinct-rollout adapter that preserves prompt identity when rollout IDs
  collide across groups, plus token and trajectory reducers that preserve a
  chosen objective across physical row partitions;
- an eight-choice normalized-policy witness whose scalar score stays fixed
  when a wrong fragment mean changes the parameter gradient.

The public facade is **reference.py**. Arithmetic fixtures and reductions live
in **reference_core.py**; typed group validation and exact centering live in
**reference_group.py** to keep each module within the companion size budget.
The existing public imports remain available.
The new `implementation_insights.py` is a separate finite adapter.
`test_implementation_insights.py` runs through the existing chapter smoke
manifest. It includes masked padding, endpoint, identity, and partition
checks. Named reverse-mode tests recompute a graph for each physical fragment,
accumulate separate backward calls at fixed parameters, then take one step.
They cover all 128 contiguous partitions, reverse order, a noncontiguous
partition, empty fragments, and an intentionally wrong equal-fragment mean.
A separate negative control differentiates the incorrect whole-score locality
mask at both exterior tails and both advantage signs. These are finite local
checks; no upstream trainer is executed by them.

Three outputs have different endpoints:

- **STUDIO_LOG_RATIOS** and **studio_trace()** preserve the historical 16-token
  representative diagnostic. Its 13 eligible values do not determine a
  full 48-token objective.
- **STUDIO_COMPLETE_LOG_RATIOS** and **complete_studio_score()** define
  **studio-zero-tail-v1**: append 4, 8, 4, and 16 zeros to the four diagnostic
  rows. All 48 values are specified; 45 are reward-eligible. The score is
  0.09072106790975874. The normalized scalar coefficient sum is
  0.08922245719576709, not a model parameter gradient.
- **tiny_policy_step()** uses the shared binary-answer logit from the printed
  example. Behavior probabilities are (0.25, 0.75), current probabilities
  are (0.3, 0.7), and the saved group contains blue and red once each.
  Its analytic parameter gradient is 0.26516770487981467; an ascent step
  of size 0.4 yields probabilities approximately (0.322735065, 0.677264935).

**matched_reduction_fixture()** is a different, fully specified 48-token
comparison: eligible counts (8, 6, 8, 10), eligible log ratios zero, and
other log ratios 0.3. All reduction columns use the same surviving reward
numerators. **length_ledger(..., active_counts=...)** now applies those counts
to every reward column; **ratio** means the common eligible-token ratio,
and supplied advantages stay frozen. The regularizer keeps its full support
and total-token denominator.

Run the focused unit tests from the companion-code root (`code/`):

    python3 -B _maintenance/chapter_test_runner.py \
      long_cot_rl_policy_optimization/test_long_cot_rl_policy_optimization_reference.py

The pinned Slime comparison retained in the author's research archive selects
context-parallel size one, `calculate_per_token_loss=False`, no custom reducer,
and no correction that
changes the supplied loss masks. That source inspection is separate from the
local tests above.

The binary example takes one teaching parameter step; no model-training
campaign or Moonshot AI pipeline is implemented or reproduced.
The fixtures test only the chapter's declared mathematics, typing, and
accounting. They supply no model- or environment-measured policy outcome.
