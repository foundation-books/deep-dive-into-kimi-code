# Chapter 37 MoonEP semantic companion

This dependency-free package models the Chapter 37 placement contract:
immutable routes, semantic and home loads, perfectly balanced allocation,
bounded redundant expert types, an exact small-instance oracle, aligned
segments, plan and alias lifetimes, layout accounting, and exact gradient
repatriation.

It is a CPU correctness and accounting oracle. It does not implement MoonEP
communication, GPU planning, symmetric virtual memory, expert kernels, or
performance measurement.

Run its focused tests from the companion-code root (`code/` in the book source):

```sh
python3 -B _maintenance/chapter_test_runner.py \
  moonep/test_moonep_reference.py
```

`checks.all_checks()` exposes the stable twelve-check evidence surface used by
the historical deterministic campaign. `reference.py` is the public semantic
facade; `planning.py` owns constructive and exact allocation, while
`runtime.py` owns buffer, lifetime, accounting, and gradient checks.

`temporal_balance.py` adds the two-batch reflection of the existing fixture,
a fixed-replica witness, and common semantic checks separated from an optional
perfect-balance claim. The focused tests also reject wrong-batch plans,
stale expert maps, outdated weights, incompatible quantization tags, unsafe
placement transitions, and undeclared backend fallback. These are finite CPU
contract checks; no EPLB implementation or serving framework is executed.
The historical twelve-check evidence surface remains unchanged.

`backward_lifecycle.py` adds a differentiable two-coordinate quadratic expert
under the existing route plan. Exact differences check every input, parameter,
and route coefficient across two successive microbatches; home return consumes
temporary partials once. Missing/doubled returns and an overwritten alias are
explicit faults. This remains a CPU ownership model: the chapter's source-matched
MoonEP adapter is illustrative and its device synchronization is not executed.

`reader_walkthrough.py` reconstructs the printed per-origin values, logical
route/payload counts, signed destination worksheet, reduce-storage alias,
165/85/11/177 populations, and the fully specified four-rank exercise.
Its separate allocation binding and payload lease model rejects every changed
binding field and an overwritten view, while allowing copied-payload replay
through unchanged bindings. The original conservative epoch checker remains
unchanged. Run the additional dependency-free checks from `code/`:

```sh
python3 -B -m unittest moonep.test_moonep_reader_walkthrough -v
```
