# Chapter 17 companion: Supervised Cold Start for Agentic RL

This dependency-free companion makes four Chapter 17 teaching artifacts
executable:

- the valid studio trajectory and its eight frozen variants, including seven
  rejections and one accepted low-efficiency trace;
- a canonical symbolic serialization, parser, malformed structures,
  structural token spans, and SFT mask;
- the weighted masked-loss and six-ledger numerical examples, including empty
  denominator rejection; and
- the qualified sequential-error arithmetic and two recovery-record fixtures.

Run the unit tests from the companion-code root (`code/`):

```sh
python3 -B _maintenance/chapter_test_runner.py \
  k3_post_training_data_sft/test_k3_post_training_data_sft_reference.py
```

The fixtures neither train a model nor implement or reproduce a Moonshot AI
pipeline. They test only the chapter's declared record, arithmetic, and
serialization contracts. They supply no model- or environment-measured
training result.

## Record and validation reference

The chapter's five-section print treatment focuses on selection, the objective,
distribution shift, and executed readiness. This reference retains the record
administration needed to implement its examples. `reference.py` re-exports the
types, validation, variants, serialization, loss, and accounting functions from
the neighboring `reference_*.py` modules.

A supervised agentic record needs the following conceptual components. The
fixture represents them through separate functions and structures; it does not
claim a unified K3 record schema.

| Component | Contents and fixture representation |
| --- | --- |
| Task | Instruction, allowed actions and schemas, observation types, terminal predicate, call budget; `task_contract()` |
| Environment | Initial state, geometry, and permitted transitions; `environment_contract()` and the deterministic executor |
| Events | Ordered `Event(ordinal, kind, actor, payload, correlation)` objects; instruction, assistant calls, executor results/errors, state changes, answer |
| Verification | Named `ValidationResult`, validator version, and content identity from `certificate_for()` |
| Provenance | Seed/task lineage, teacher and executor identity, generation and repair history; the fixture demonstrates shared-seed accounting rather than a full provenance database |
| Training view | Canonical serialization, token ownership, target mask, and record weight; `serialize_events()`, `structural_tokens_and_mask()`, and the loss fixture |

The validator uses zero-based event ordinals; the chapter's six-event table uses
one-based presentation numbers. Original red and blue boxes are `(20,20,40,40)`
and `(20,80,40,40)`. The crop moves the origin by `(10,10)`, giving boxes
`(10,10,40,40)` and `(10,70,40,40)`. Vertical centers change from 40/100 to 30/90;
their difference remains 60.

`validate_record()` performs a deterministic pass in dependency order:

1. Require a nonempty event sequence with consecutive ordinals.
2. Check each event's kind, actor, payload, and call schema. Require declared
   observation dependencies to refer to completed results. Track unique call
   keys and the call budget.
3. Evaluate calls in the current fixture state. Match results or errors to
   unmatched calls. Compare successful observations with deterministic execution;
   apply only the declared crop transition and reject stale comparison boxes.
4. Require completed call accounting and one supported terminal claim. Retain
   unresolved errors until a successful retry supplies the needed evidence.
5. If a certificate is supplied, verify its validator version and content hash.

This is a small executable oracle for the frozen cases, not a complete validator
for arbitrary agent interactions. Its content hash identifies the accepted
event bytes; it is not an external attestation of executor authenticity.
`corrupted_variant()` and `record_fixture()` distinguish seven hard rejections
from C8's valid redundant call. Efficiency is a preference among valid paths.

`recovery_events("executor_error")` retains a transient executor error, retries
the observation with a fresh call key, and then compares. The `"state_refresh"`
variant re-observes after a crop before comparing. Both are validated event
sequences. The short recovery descriptions in `sequential_error_fixture()` are
illustrative learner-error scenarios; its numerical check covers only the
declared horizon terms.

For dependency-aware repair, retain the original record and identify the first
changed state, event, or argument. Re-execute affected calls and downstream
consumers; regenerate the supported answer. Rebuild changed serialization and
mask spans, run validation, and issue a certificate for the repaired bytes.
Preserve old and new identities and the repair reason in provenance. Changing
only the final answer cannot correct an unsupported upstream observation.

## Serialization, masks, and normalization

The canonical JSON serializer is a teaching format. Its parser checks exact
fields and canonical round-trip equality. The symbolic tokenizer assigns five
structural tokens to each event; the valid record has 30 tokens in six spans,
with 15 assistant-owned targets. These counts are separate from the numerical
A/B loss example and the D/G mixture example. They are not XTML token counts.

`validate_mask()` checks span coverage, target ownership, and mask length.
Observations with zero direct target loss still condition later predictions.
The loss fixture computes global weighted-token normalization and an unweighted
mean of record-normalized losses. They give 1.0925138800068108 and
1.004345880271493 for the fixed A/B probabilities. Both reject empty target
mass. With fixed masks and weights, a record-mean gradient's coefficient in the
global objective is `w_i S_i / sum_j(w_j S_j)`. Vector magnitudes and directions
still determine the resulting gradient.

## Six accounting measures

For each selected record let `L_i` be serialized length, `S_i` supervised-token
mass, `A_i` action count, `u_i` seed lineage, and `C_i` a declared compute proxy.
Choose a corpus or sampling window before totaling these quantities. The table
uses one direct record D and one agentic record G at unit record weights.

| Measure | Family numerator | D / G raw values | D / G normalized shares |
| --- | --- | --- | --- |
| Records | Number of selected records | 1 / 1 | 1/2 / 1/2 |
| Serialized tokens | Sum of `L_i` | 20 / 200 | 1/11 / 10/11 |
| Supervised tokens | Sum of `S_i` | 10 / 50 | 1/6 / 5/6 |
| Actions | Sum of `A_i` | 0 / 8 | 0 / 1 |
| Source lineage | Distinct seeds and reuse counts | One shared seed used twice | Family shares undefined without an allocation rule |
| Approximate compute | Sum of declared `C_i` | 20 / 320 | 1/17 / 16/17 |

`mixture_fixture()` retains raw totals of two records, 220 serialized tokens,
60 supervised tokens, eight actions, one unique source, and 340 declared compute
units. `normalized_shares()` uses exact fractions and rejects negative inputs
or zero total mass. Shared seed sets can overlap between families, so counting
their sizes independently does not produce additive family shares.

Record weights modify the supervised objective: use weighted target mass
`sum(w_i S_i)` to discuss its coefficients. Sampling probabilities modify
expected raw exposure: with fixed record lengths, family contributions are
proportional to `q_i w_i S_i`. Equal expected raw target mass can be obtained by
`q_D=5/6, q_G=1/6` at unit weights, or by `w_D=1, w_G=1/5` at equal sampling.
These choices can differ in finite-batch normalization, context processed, and
optimization behavior. No one accounting measure establishes gradient norm,
learning quality, throughput, memory, energy, or actual production cost.

## Source and evidence boundary

The manuscript's source comparisons assign different roles to public
sources: K2 explains synthetic task/tool construction; K2.5 supplies the
text-only SFT case after multimodal pretraining; ToolBench illustrates path
filtering and conversation conversion; LLaMA-Factory exposes label masking;
Molmo2 exposes sampling, message weighting, and packing. These public-source
comparisons do not disclose K3's private masks, mixtures, teachers, or verifier.

The existing deterministic checks establish finite-record semantics and
arithmetic. The chapter's additional acceptance-conditioning and gradient
derivations explain their learning implications; executed policy studies would
be needed to establish an improvement in behavior. Detailed QAT mechanisms
belong to Chapter 23.
