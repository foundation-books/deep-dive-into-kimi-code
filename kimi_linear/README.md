# Chapter 5 Kimi Linear reference

The dependency-free reference models the chapter's typed layer schedule,
fixed-extent KDA ownership, ordered global-cache records, complete
accepted-prefix state, causal access matrix, decode continuation, and narrow
logical-memory and attention-work ledgers.

`contracts.py` owns the immutable schedule, value types, and accounting
functions; `reference.py` re-exports that API and implements the causal state
transitions.

It is deliberately an abstract correctness oracle. It does not implement
production KDA, MLA projection or reconstruction, a GPU kernel, a cache
manager, model training, or a performance benchmark.

Run the tests from the companion-code root (`code/`):

    python3 -B -m unittest kimi_linear.test_kimi_linear_reference

The reviewed evidence campaign lives in experiments/kimi_linear/.
