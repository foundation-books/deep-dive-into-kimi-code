# Chapter 20 semantic reference

This dependency-free companion reproduces the finite arithmetic and state
contracts in **Sampling, Curricula, and Partial Rollouts**:

- raw and Beta-smoothed model-relative difficulty;
- target/acquisition normalization, support, and exact task-level correction;
- legal rollout transitions and pause-checkpoint validation;
- four-coordinate reuse and gap-free segment provenance;
- strict settled/ready group predicates and a finite fairness witness; and
- the complete three-task, four-slot mixed-version studio trace.

It is a semantic teaching reference. It is not Moonshot AI code, a Kimi
sampler or trainer, or a throughput measurement.

Run the unit tests from the companion-code root (`code/`):

    python3 -B _maintenance/chapter_test_runner.py \
      sampling_curricula_partial_rollouts/test_sampling_curricula_partial_rollouts_reference.py
