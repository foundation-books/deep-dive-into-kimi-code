# Chapter 23 semantic reference

This dependency-free companion reproduces the finite contracts in
**Capability Consolidation and Deployment-Aware Post-Training**:

- nine domain--effort routes and mixture/count conservation;
- reverse-KL, token log-ratio, clipping, support, and top-k arithmetic;
- strict sampled-score refusal before clipping or any supplied optimizer
  callback, with late failure and unresolved-score fixtures; committed-use
  counts are returned only after an admitted updater succeeds;
- unique sampled/admitted token identities versus repeated committed uses;
- retention-matrix and evidence-status structure;
- quantization contracts and ordered precision fingerprints;
- rollout--training parity dispositions;
- acceptance mass, total variation, LK loss, and seven-step recurrence; and
- the illustrative studio handoff with absent model-level evaluations and a
  constructed parity failure that keeps readiness blocked.

It is a semantic teaching reference. It does not train or evaluate a model,
reproduce private Moonshot AI code, emulate an undisclosed K3 quantizer, or
supply an integrated K3 retention or serving evaluation. The literal `pending`
machine value represents an evaluation not supplied in this example; it does
not schedule author work. Every required field and check must still satisfy
the readiness gate.

The finite checks distinguish the pinned NeMo RL score-to-loss path from
SpecForge recurrence and loss selection.
The separate author-side Chapter 23 public-source experiment adds an actual
small NeMo score-to-update trace, the SpecForge LK helper, and a TorchAO MX
module update. Its records live under
`experiments/capability_consolidation_deployment_aware_post_training/` in the
complete book source and are outside this companion export; the volume states
their tested boundary.
The string-valued `recurrent_inputs` fixture illustrates token feedback;
it does not identify K3's undisclosed token-input construction or implement
SpecForge's hidden-state recurrence with shifted recorded tokens.

Run the unit tests from the companion-code root (`code/`):

    python3 -B _maintenance/chapter_test_runner.py \
      capability_consolidation_deployment_aware_post_training/test_capability_consolidation_deployment_aware_post_training_reference.py
