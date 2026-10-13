# Chapter 12 companion: Moonlight and Muon

`reference.py` gives a dependency-free, complete two-step Muon/AdamW trace plus
matrix, optimizer-partition, state-ledger, distributed-equivalence, and
scaling-law calculations for the chapter's small fixtures. The
`global_mean_gradient` helper combines rank-local mean gradients with
their normalization masses before sharded Muon semantics are compared.
Focused tests check the printed Nesterov expansion, diagonal polynomial
trace, query endpoints, base-rate decay, and weighted gradient assembly. It
deliberately does not train a language model or measure hardware performance.

Run the checks from the companion-code root (`code/`):

```sh
python3 -B _maintenance/chapter_test_runner.py \
  moonlight_muon/test_moonlight_muon_reference.py
```

The evidence campaign in `experiments/moonlight_muon/` records the reviewed
outputs and their claim boundaries.
