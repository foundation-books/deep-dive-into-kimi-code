# Chapter 29 finite repository-engineering oracles

The historical finite algorithms and formal IDs in this package predate the
current print numbering. The print chapter contains the worked examples and
answers; these checks exercise their bounded software-engineering contracts.

The revised print chapter calls the vertical-scale variant `studio-code-002`,
separating it from Chapter 25's missing-offset case. The original
`reference.py` retains its historical `studio-code-001` identity.
`revision_case.py` adds finite A/C patch-selection arithmetic, independent B,
the scale witness, edit-match contrasts, and the main correctness-qualified
candidate-timing condition. Its edit and timing helpers are explicitly
schematic; they do not execute Kimi Code or KernelBench.

`reference.py` implements ten small deterministic checks for the historical
`studio-code-001` teaching fixture: repository identity, instruction scope,
context selection, localization, action admission, patch preservation,
validation coverage, failure attribution, recovery, and conflict-aware
parallelism.

Run the tests from the parent `code/` directory with:

```sh
python3 -B -m unittest -v kimi_code_test_grounded_software_engineering.test_kimi_code_test_grounded_software_engineering_reference
```

For a source-bound check, set `KIMI_DEV_CHECKOUT`,
`KIMI_CODE_CH29_CHECKOUT`, and `KIMI_KERNELBENCH_CHECKOUT` to clean
checkouts at the commits recorded in `source_probe.py`, then run:

```sh
python3 -B -m kimi_code_test_grounded_software_engineering.source_probe
```

The probe checks exact source hashes, captures prompts from the pinned
BugFixer function, and runs the pinned TestWriter controller with local
stubs to check file selection, the oracle branch, and patch-shape retry.
It inspects Kimi Code TypeScript and KernelBench branches at their pins;
those two implementations are not executed natively.

The finite fixture performs no repository mutation, command execution,
network access, or model inference. The source probe invokes read-only Git
commands and writes a temporary local JSONL record while exercising
TestWriter; it makes no repository edits, network calls, or model requests.
Passing results establish only the exact finite properties encoded here;
they are not evidence about Kimi Code reliability or model quality.
