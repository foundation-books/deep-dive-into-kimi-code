# Low-Precision Execution and Speculative Decoding

This original, dependency-free finite reference makes Chapter 45's exactness,
commit, state, and cost arithmetic executable. It is not a production sampler,
quantization kernel, serving runtime, or performance benchmark.

From the companion root:

```sh
python3 -m unittest low_precision_execution_speculative_decoding.test_low_precision_execution_speculative_decoding_reference
```

`reference.py` checks target--draft overlap, positive-residual correction and
the one-step target law, including an exhaustive 225-pair small-law check.
It also models greedy and stochastic chain commitment, EOS/output caps,
eager scalar recovery and request-state identity checks before publication.
The finite MXFP4 converter encodes 32 E2M1 elements with a caller-chosen E8M0
scale, ties-to-even rounding and saturation. It rejects nonfinite input and
does not implement the standard's complete exceptional-value behavior.

The original fourteen tests cover M44-A's bit patterns, dot-product/error arithmetic,
ties and clipping; M44-B's binary32-order, greedy-flip and recovery examples;
R44-A's rejection/all-accepted/stopped traces; and M44-C's population cost.
Explicit random draws are supplied trace realizations. Target/draft rows are
assumed to be causal processed laws; these functions cannot establish parity
of a real model's batched kernel or prove a tree selection policy exact.

The numerical example is teaching arithmetic rather than a K3 measurement.
It supplies no matched runtime comparison of acceptance cost, proposer shape,
or state recovery; the finite functions are not hardware benchmarks.

`dspark.py` adds eight tests to the same suite (22 total). The DSpark cases
extend the existing example without changing its target law or commit trace:

| Chapter concept | Functions and finite checks |
| --- | --- |
| Actual coupled proposal (§45.4–6; Exercises 3–4) | `markov_proposal_law`, `markov_draft`; temperature/mask order, actual-law mass enumeration and wrong-law negative control; R44-A still proposes `(a,b,c)` and commits `(a,a)` |
| Admission (§45.5; Exercise 3) | `selected_proposal_output_law`; DSpark Appendix A adaptation gives `.80` instead of `.60`; 75 constant-admission controls preserve the target law |
| Direct recovery (§45.7; Exercise 7) | `kda_step_record`, `reconstruct_kda`; all four prefixes of a dyadic matrix window exactly, all nine of a nondyadic window within absolute `1e-12`; stale correction negative control |
| Input and state ownership (§45.6–7; Exercise 8) | `window_positions`, `recover_convolution_history`; 21 nonterminal prefix geometries at widths 3/7/8, cache-boundary positions, convolution histories and request-remapping controls |
| Confidence and cost (§45.8; Exercise 9) | `prefix_survival`, `predicted_output_rate`; conditional products, budgets `(2,1)` versus `(3,3)`, and ratios of total work to total cost |

These DSpark functions are finite checks of a declared example, not an
executed Inferact/vLLM configuration. They do not load models, execute GPU
kernels, test engine support,
establish a runtime random-stream contract, or measure throughput. EOS and
output-cap branches are covered by the original stopped-chain oracle; the
separate position arithmetic assumes a nonterminal window with a live final
correction or bonus.
