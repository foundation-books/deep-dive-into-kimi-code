# Chapter 35 hybrid-attention and depth-mixing reference

`reference.py` is a dependency-free semantic oracle for owner validation,
compact-record layout transport, request-local causal mixing, exact hybrid
dispatch, Full and Block AttnRes, stable-summary merging, token sharding, and
accepted-prefix transactions. `faults.py` executes the six shape-plausible
negative controls from the running trace and records the boundary that rejects
each one. Those semantic checks use exact rational arithmetic. The companion
does not imitate an accelerator kernel or predict performance.

The accepted-prefix transaction oracle projects each KDA entry to a scalar
recurrence stand-in and each global record to two scalars. It checks endpoint
atomicity, not convolution execution or the chapter fixture's four-scalar
record. The manuscript's complete KDA bundle additionally includes query,
key, and value convolution histories; their ownership and last readers remain
required although this finite arithmetic projection does not execute them.

Run the focused checks from the companion-code root:

```sh
python3 -B _maintenance/chapter_test_runner.py \
  hybrid_attention_depth_mixing_kernels/test_hybrid_attention_depth_mixing_kernels_reference.py
```

The reviewed deterministic evidence record lives in
`experiments/hybrid_attention_depth_mixing_kernels/`. EXP33.01/02 retain bounded
public-PyTorch Gated-MLA results; EXP33.05 retains its bounded public-FLA
Full-style result. The eight protocol identifiers are unchanged.

## Phase-by-object validation coverage

This is the complete comparison plan for a changed hybrid execution path. A
check listed here is an obligation for that phase, not a claim that K3's
production kernel was tested. The dependency-free reference supplies semantic
checks; the bounded public-PyTorch/FLA records retain their stated numerical
scope. Optional measurement designs supply no broader production outcome.

| Observable | Training | Packed prefill | Decode | Discriminating edge and check home |
| --- | --- | --- | --- | --- |
| Row origins and causal mask | required | required | owner | Unequal request lengths and tail tile; `packed_isolation`, `faults.py` |
| KDA state candidate | required | required | required | Nonzero input and rejected proposal; `accepted_continuation` |
| Gated-MLA logical records | required | required | required | Non-symmetric layout and terminal global layer; `layout_round_trip`, `hybrid_dispatch` |
| Ungated and gated global output | required | required | required | FP32 attention output and nontrivial gate; EXP33.01/02 protocol and result |
| Depth source multiset | required | required | required | First target, group boundary, final partial; `block_sources`, `faults.py` |
| Stable `(m,L,U)` summary | required | required | required | Unequal maxima and empty identity; `stable_merge`, `two_phase_block` |
| Shard origins and collectives | topology | required | topology | Missing or duplicate row; `sequence_shards` and protocol for actual collectives |
| Side-stream producer revision | n/a | when overlapped | required | Delayed producer; `faults.py` semantic witness, no device-overlap result |
| Input and parameter gradients | required | if training prefill | n/a | Nonzero upstream gradient; `backward_checks.py`, EXP33.01/02 and EXP33.05 |
| Allocation, traffic, and time | required | required | required | Fallback and unsupported shapes; scoped experimental protocols, not semantic checks |

Identity, layout, source-set, collective-instance, and event-order checks use
exact comparisons. Numeric tensors use the result's declared tolerance and
nonfinite rule. A supported-path comparison names the executed candidate; an
unsupported path records its fallback or error separately.

`backward.py` supplies a separate floating-point AttnRes example with explicit
value and score paths, per-token shared-parameter partials, optional output
normalization, and saved/recomputed mixtures. `backward_checks.py` checks all
source, query, norm-scale, and optional output-scale coordinates by centered
differences. It also checks a wrapper that assembles Block sources and scatters
their adjoints. Missing score paths and missing token contributions fail.
These focused checks do not execute FLA or promote the Block wrapper to a
released K3 implementation; the historical ten-oracle API is unchanged.
