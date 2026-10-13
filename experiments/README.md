# Selected book experiments

Run these commands from the companion repository root, the directory containing
the top-level `Makefile`. Python 3.10–3.12 is sufficient; no installation,
manuscript files, network access, model weights, or accelerator is required.

```sh
python3 -B -m experiments all --check
```

Omit `--check` to print the computed values and summaries. Commands print to
standard output and do not replace recorded evidence. The validation experiment
uses temporary authored Python files and local subprocesses. Its measured wall
times depend on your CPU.

| Command argument | Book connection | What you can inspect |
| --- | --- | --- |
| `kda` | Chapter 4, Kimi Delta Attention | Ten exact/float recurrence, chunk-equivalence, causality, gradient and accounting comparisons. |
| `moba` | Chapter 11, MoBA | Ten dense-masked/partitioned comparisons, finite-difference derivatives, causal boundaries, online softmax merges and route costs. |
| `gated-mla` | Chapter 6, Gated MLA | Independent scalar float64 reconstruction of 36 recorded FP32 cases, and paired allocation summaries for 48 recorded cache cases. |
| `validation` | Chapter 29, test-grounded software engineering | Three test-selection policies on nine authored Python patches, including a defect missed by every available policy test. |
| `recovery` | Chapter 29, interrupted coding sessions | All 1,200 simulated crash/policy/repetition cells, including duplicate effects, missing receipts and unresolved outcomes. |

For example:

```sh
python3 -B -m experiments gated-mla
python3 -B -m experiments validation
python3 -B -m experiments recovery
make experiment-checks
```

The KDA and MoBA calculations use the existing [KDA](../kda/README.md) and
[MoBA](../moba/README.md) reference implementations. Their small inputs and
outputs establish toy correctness and accounting only.

The Gated MLA command analyzes supplied observations from one H200. It
reconstructs materialized attention and outputs from the actual recorded
inputs, compares them with recorded compact-path outputs under the frozen
numerical gates, and recalculates allocator increments and medians. The cache
analysis consumes the recorded parity checks; it does not rerun GPU cache
construction or device measurements. Allocator peaks cover the declared
synthetic cache lifetime and include construction temporaries. They do not
establish K3 production residency, serving performance or training stability.

The validation command executes each patch/policy once, then compares its
non-timing outcomes with all 270 historical cells (ten repetitions per
patch/policy). Historical wall times and local replay wall times have
separate summaries. Predicate coverage and the hidden duplicate oracle explain
why a green test ladder can miss a defect. Neither population represents
production repositories or Kimi coding quality.

The recovery command recreates every retained state, effect attempt, receipt,
event and metric. Its milliseconds are modeled costs. No real tool effect,
repository mutation or external post occurs. Passing the replay means the
simulation matches the recorded outcomes, including unsafe and unresolved
cases; it does not mean every simulated policy is safe.

The [manifest](manifest.json) records original archive hashes and checksums for
all six supplied data packets: [KDA](data/kda.json), [MoBA](data/moba.json),
[Gated MLA numerics](data/gated_mla_numerics.json),
[Gated MLA cache](data/gated_mla_cache.json),
[validation](data/validation.json), and [recovery](data/recovery.json).
Archive identifiers preserve source provenance independently of runtime paths;
machine-specific archive directories use portable artifact labels. The public
packets retain numerical values and
observed outcomes, with portable metadata. Original source verification,
machine identity, cluster job controls, operational failure logs and the
historical protocol backlog remain in the author archive. These adaptations
do not change an original study's result.

The correctness/fault tests run with `make experiment-checks` and as part of
`make smoke`. The selected files are also included in the companion exporter.
