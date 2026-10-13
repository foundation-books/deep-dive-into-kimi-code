# Chapter 33 FlashKDA and KDA-kernel reference

`reference.py` is a dependency-free semantic oracle for accepted-prefix KDA
state, ordered affine segments, packed variable-length boundaries, logical and
V-first layout conversion, versioned dispatch, state/communication byte
ledgers, and projected-input replay. It deliberately does not imitate the
FlashKDA CUDA source or predict hardware performance.

Run the focused checks from the companion-code root:

```sh
python3 -B _maintenance/chapter_test_runner.py \
  flashkda_kda_kernels/test_flashkda_kda_kernels_reference.py
```

The reviewed deterministic evidence record lives in
`experiments/flashkda_kda_kernels/`. Its immutable campaign identifiers retain
their reviewed outcomes and scope limits, including negative and blocked runs.

`backward.py` adds the recurrence pullback, an explicit reverse chunk boundary,
save/replay alternatives, and a public-stage ledger. `backward_checks.py`
checks every recurrence-ready coordinate of the 19-token, 16+3 fixture under
output-only, final-state-only, and combined losses using exact centered
differences. A separate smooth floating-point fixture checks L2, exponential
gate, and sigmoid-beta pullbacks away from normalization floors. Deliberately
dropping the later chunk's adjoint fails. These checks extend the focused test;
they do not change the historical ten-oracle evidence API or run FLA/FlashKDA.
