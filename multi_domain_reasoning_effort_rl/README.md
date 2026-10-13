# Chapter 21 semantic reference

This dependency-free companion reproduces the finite contracts accompanying
**Multi-Domain and Reasoning-Effort Reinforcement Learning**. The compact print
chapter uses selected examples. The finite checks cover:

- the three-domain by three-effort policy-cell product;
- disjoint resource coordinates and the domain-dependent token projection;
- positive baseline and bound-multiplier validation;
- below, equality, and strict-overrun reward arithmetic;
- max-to-high-to-low curriculum order and finite max-stage validation;
- coordinate-wise allocation conservation;
- protocol-scoped Pareto dominance; and
- all nine evidence-qualified studio handoff schemas.

It is a semantic teaching reference. It is not Moonshot AI code, a Kimi
trainer, an implementation of a private budget estimator, a benchmark, or
evidence of model-level quality beyond the separately stated finite study.

The separate `finite_policy.py` and `train_fixture.py` modules implement the
Chapter 21 four-route learning example. They use the existing budget and
reward functions, a stable softmax, exact logit gradients, and fresh sampled
groups with the documented inclusive-mean correction. The declared route
costs and success probabilities are teaching assumptions. A split sampled
run can resume from saved logits, update count, route counts, and random state.
The executed 30-seed study, configuration, and figure generator live in
`experiments/multi_domain_reasoning_effort_rl/`; no model weights or external
dependencies are needed.

Run the unit tests from the companion-code root (`code/`):

    python3 -B _maintenance/chapter_test_runner.py \
      multi_domain_reasoning_effort_rl/test_multi_domain_reasoning_effort_rl_reference.py

    python3 -B _maintenance/chapter_test_runner.py \
      multi_domain_reasoning_effort_rl/test_finite_learning.py
