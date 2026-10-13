# Chapter 10 integrated K3 reference

This dependency-free companion composes exact-rational toy versions of the
Chapter 4--9 interfaces. It checks the Chapter 10 claims that require
whole-model agreement:

- the ordered terminal-global schedule and the 69/24/93 identity;
- one dense channel layer followed by routed layers;
- two depth-retrieval points per decoder layer and partial-tail coverage;
- token and media-origin conservation;
- fixed KDA state, growing global records, ephemeral depth history, and
  transient route records;
- exact full-prefix versus cached one-token continuation; and
- context, logical-state, route, and precision-range ledgers.

Section 10.6 prints the input, depth, sequence, channel, router, and readout
rules used here. The first-row fractions, actual layer-2 route pairs, and
rounded terminal logits are asserted in
`test_integrated_k3_transformer_reference.py`. The executed global stand-in
record has two scalar components, a latent summary and a shared summary;
the 44-element KDA complete-state ledger counts source-mechanism components
that the eight-scalar recurrence stand-in does not execute.

Run the focused checks from the companion-code root (`code/`):

```bash
python3 -B _maintenance/chapter_test_runner.py \
  integrated_k3_transformer/test_integrated_k3_transformer_reference.py
```

The routines are mathematical fixtures. They are not K3 weights, production
operators, a physical cache allocator, or a performance implementation.

## Volume I implementation investigations

The `implementation_insights.py` checks connect six local choices to the
Chapter 10 composition check. Run all seven small
comparisons from `code/`:

```bash
python3 -B -m integrated_k3_transformer.implementation_insights
```

Check their numerical expectations with:

```bash
python3 -B _maintenance/chapter_test_runner.py \
  integrated_k3_transformer/test_implementation_insights.py
```

The comparison command prints results without updating stored experiment
records or requiring external checkouts.
