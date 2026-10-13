# Chapter 4 KDA reference

`reference.py` is a dependency-free correctness oracle for the chapter's
retain--predict--correct--read recurrence, exact rational running trace,
affine segment summaries, chunk-boundary checks, finite-difference helpers,
and analytical cost ledger. It is deliberately not a kernel implementation or
a production benchmark.

Run its tests from the companion-code root:

```sh
python3 -B _maintenance/chapter_test_runner.py \
  kda/test_kda_reference.py
```

The reviewed evidence campaign lives in `experiments/kda/`.
