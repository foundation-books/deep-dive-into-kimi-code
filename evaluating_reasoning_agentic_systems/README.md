# Chapter 30 finite evaluation-methodology oracles

This package retains finite checks for the revised print chapter. Existing
D30.*, I30.* and CH30-* identifiers name its original fixtures, not current
print numbering.


`reference.py` implements ten small deterministic checks for the chapter's
`studio-eval-001` teaching fixture: record validation, protocol comparison,
pass@k, paired effects, disposition denominators, aggregation reversal, judge
order consistency, Pareto comparison, swarm accounting, and dossier claim
admission.

Run the tests from the parent `code/` directory with:

```sh
python3 -B -m unittest -v evaluating_reasoning_agentic_systems.test_evaluating_reasoning_agentic_systems_reference
```

The code performs no model inference, network access, repository mutation, or
empirical evaluation. Passing results establish only the exact finite
properties encoded here; they are not evidence about Kimi quality, benchmark
performance, robustness, latency, or safety.
