# Open Systems Problems

This historical Chapter 46 companion checks original finite models, not K3 kernels,
million-token service performance or production recovery. It uses only the
Python standard library. The former numbered chapter is now the unnumbered
Synthesis and Outlook. The numerical migration witness appears in the
numerical-execution chapter. Historical formal-model identifiers below label
the finite checks. From the companion root:

```sh
python3 -B -m unittest open_systems_problems.test_open_systems_problems_reference
```

`reference.py` provides exact rational resource floors, feasible Pareto
filtering, conditional retention, FCFS placement, scalar rounding, serial
migration costs and cohort/recovery accounting. `state.py` models two required
KDA checkpoint groups with MLA coverage, explicit input/model/numerical
identity and auxiliary readiness. Ready sets assume truthful payloads; the
model cannot inspect a real device's visibility or state contents.

Tests reproduce M46-A–E, reject incompatible diagram/preprocessing/version
and missing-group variants, check fixed-history online decisions, and show
both controller-only failure and atomic effect-boundary fencing. The effect
store starts at committed position 14/epoch 7. The complete branch trajectory
is specified in the manuscript; the code focuses on the restoration and
takeover cuts rather than emulating a model's token computation. It enumerates
both late/current write orders after takeover. The atomic transition is an
assumption, not proof of a hardware implementation. Tool outputs are supplied
events and no real tool is called.

The scalar check also exhausts E46.4's 341 input sequences over {-2,0,2,6}
at lengths 0–4. An independent closed-form sum and nearest-grid oracle check
every state; mismatching readouts remain counterexamples to token preservation.
This completes the finite enumeration, not the empirical migration study.

The scalar recurrence is deliberately not KDA. Decimal GB and seconds in
the arithmetic are hypothetical inputs. Resource floors are not runtimes;
the 1.25 serial speedup is a separate calculation. Cohort counts mean unique
requests, with correct completion by deadline and cutoff. Undefined empty
attainment and zero-lost-work amplification return `None`.

No datasets, dependencies, model weights or hardware runs are needed for these
finite checks. Their passing results do not establish empirical performance.
