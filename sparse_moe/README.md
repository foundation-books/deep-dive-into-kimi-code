# Chapter 3 CPU reference

The reference module implements the chapter's scaled-down dense-FFN and
sparse-MoE contracts with Python's standard library. The dense baseline is an
explicit bias-free ReLU map with a typed `d -> h -> d` interface. The sparse
path covers routing, full-width expert execution, dispatch/combine, balancing,
and the controlled latent comparison. These routines are arithmetic,
identity, shape, and accounting oracles, not optimized model implementations
or performance models.

Run its unit tests from the companion-code root:

    python3 -B _maintenance/chapter_test_runner.py \
      sparse_moe/test_sparse_moe_reference.py

The evidence runner under experiments/sparse_moe records the six named checks,
the environment, claim boundary, and source hashes.
