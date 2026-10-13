# Chapter 1 Transformer foundations reference

`reference.py` is a dependency-free, one-block decoder whose weights, cache,
loss, scalar reverse-mode graph, and AdamW step are all visible. It is a finite
arithmetic witness for the chapter's causal-decoder contracts, not a trained
model, production kernel, or performance benchmark.

Run the unit tests from the companion-code root:

```sh
python3 -B _maintenance/chapter_test_runner.py \
  transformer_foundations/test_transformer_foundations_reference.py
```

The suite checks cached and full-prefix equivalence, causality, printed logits
and loss, all 144 central-difference gradients, one AdamW update, and shared
scalar-graph accumulation. A cache belongs to one unchanged model version and
must be discarded after an update.
