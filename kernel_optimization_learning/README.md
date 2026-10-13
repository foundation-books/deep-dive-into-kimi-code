# Chapter 30: kernel-optimization learning

This dependency-free CPU companion implements the book-authored one-head,
forward-only normalized-and-gated operator. It makes numerical and controller
decisions inspectable; it is neither a GPU kernel nor a K3 training record.

| Chapter result | Executable role |
|---|---|
| Equation (30.1), tiny row | `reference.py::normalized_gated_rows` |
| Padding and gate diagnosis | `reference.py::{fused_padded_rows,padded_divisor_defect_rows,unmasked_padding_defect_rows,combined_padding_defect_rows}` and the reference test |
| C3/C4 finite meanings | `reference.py::partitioned_reduction_rows` with two/four partial square sums combined before normalization |
| Candidate selection and final check | `selection.py::select_fixture`, `final_check`; assigned normalized costs, four revisions, incumbent-on-tie |
| Score disposition, returns, binary update | `learning_example.py::teaching_score`, `discounted_returns`, `first_decision_policy_step` |

The oracle converts each input scalar to Python float, accumulates squared
values with `math.fsum`, and returns Python floats. The partitioned variants
also combine partial sums with `math.fsum`. They do not simulate FP32/FP16
accumulation or physical GPU scheduling. The tiny row uses
`U=(3,4,0)`, `G=(0,log(3),-log(3))`, `gain=(1,2,1)`, and `epsilon=1`;
its correct denominator is `sqrt(28/3)`. The finite tests cover the four
zero/sentinel padding predictions and widths 64/65.

The selection fixture checks A = 128 rows of width 1024, B = one row of width
4096, and a tail case of width 65 before consuming any assigned cost. It
retains C0 on C2's score tie, selects C3 at `Q=3/4`, and keeps C3 after C4
scores `13/21` despite improving B. It records `last=C4` separately from
`selected=C3`; the selected source/model snapshot is then rechecked at
held-out widths 127 and 257 with signed, near-zero, and zero values. These
are Float64 finite checks at absolute/relative tolerance `1e-12`, not
held-out GPU performance. A missing or nonpositive assigned cost is unscored.

`learning_example.py` retains the separate three-turn score streams
`D=(0.4,0.4,0.4)` and `E=(0,0.4,1)`. At discount one, the first-decision
objective gives baseline `1.3`, gradient `0.05`, new logit `0.025`, and
probability `0.5062496745`. The independent turn-group calculation remains
a finite teaching check; the optional pinned-source probe in
`experiments/kernel_optimization_learning/source_probes.py` tests the
selected KernelGYM routine itself with controlled tensor substitutes.

Run from `code/`:

```sh
python3 -B -m kernel_optimization_learning.reference
python3 -B -m kernel_optimization_learning.selection
python3 -B -m kernel_optimization_learning.learning_example
python3 -B _maintenance/chapter_test_runner.py \
  kernel_optimization_learning/test_kernel_optimization_learning_reference.py
```

The separate `experiments/kernel_optimization_learning/` harness contains the
existing controlled upstream probes and an optional CUDA measurement design.
No compiled GPU-candidate timing result is supplied for the finite controller;
its selection costs are assigned inputs.
