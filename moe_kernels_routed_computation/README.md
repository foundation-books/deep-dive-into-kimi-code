# Chapter 36 semantic companion

This dependency-free Python package implements exact route-record,
count/scan, stable-permutation, restoration, schedule-equivalence, and scalar
backward oracles for Chapter 36. A small floating-point fixture follows the
SiTU gate/up/product/down dependency graph. It is not an accelerator kernel or
a performance model.

Run the direct tests from the companion-code root (`code/` in the book source):

```sh
python3 -B _maintenance/chapter_test_runner.py \
  moe_kernels_routed_computation/test_moe_kernels_routed_computation_reference.py
```

`checks.all_checks()` is the stable ten-check evidence API used by the chapter
experiment campaign.

`backward.py` extends the twenty-route fixture to analytic capped-SiTU input,
gate/up/down parameter, and route-weight derivatives. It includes a fifth,
empty expert and optional ownerless padding. `backward_checks.py` compares
every coordinate with centered differences of the token-major forward map,
checks saved/replayed grouping, and rejects wrong-origin and double-weight
mutants. This extension isolates the nonlinear expert; the full latent
projection, RMSNorm, shared branch, and router chain remain caller obligations.
It is separate from the historical ten-check evidence API and does not run
SonicMoE or an accelerator comparison.
