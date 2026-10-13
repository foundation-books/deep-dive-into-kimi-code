# K3 validation checklist companion

These dependency-free finite oracles implement the former Chapter 45's original
tolerance, recurrence, routed-gradient, law, causal-order, state, service
and resource examples. They are not KDA/MLA kernels or GPU benchmarks.

Volume III, Appendix A, *Checklist for Validating K3 Implementation Changes*,
collects optional reference and test choices. The mechanism chapters explain
the comparisons locally. This package retains the longer arithmetic, law,
trace and service fixtures, including examples removed from the printed
checklist. Its stable name reflects their historical source and does not
reserve a printed chapter.

From the local companion root (`code/` in this checkout):

```sh
python3 -B -m unittest discover -s system_correctness_performance -p 'test_*.py'
```

The former printed Section A.5's operational instructions are retained here
beside the executable fixture. This command is verified against this local
layout; companion packaging must recheck the working directory and invocation
against its release tree. This handoff does not establish public package
availability or a completed release.

`reference.py` makes arithmetic and counting conventions explicit.
`state.py` models a durable record as an atomic Python assignment and an
effect-boundary epoch comparison. State objects carry exact history and
role identities rather than model tensors. The fixture reconstructs the
same committed prefix on both sides of a crash, with a fixed decision tape;
it tests specified transitions, not numerical KDA/MLA execution, physical
storage durability, concurrent device fencing or random-generator quality.
`trace.py` validates required DAG edges against
an observed ordering and retains dependency-complete witnesses.

The tests include deliberately corrupted identities, missing readiness,
shared-route mistakes, non-finite values, unfinished requests and duplicate
logical outcomes. Empirical campaigns remain unrun; wall time of these
checks has no performance interpretation.
