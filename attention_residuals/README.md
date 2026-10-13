# Chapter 7 Attention Residuals companion

This dependency-free companion is a correctness oracle for the typed contracts
in Chapter 7. It evaluates a ten-token, width-eight request through fourteen
toy transformation outputs, with separate pre-attention and pre-MLP depth
mixers in seven decoder layers.

The reference includes:

- standard residual unrolling;
- Full AttnRes and an independent source-major oracle;
- Block AttnRes with completed summaries, a current partial, and a partial
  final block;
- exact source conservation and boundary validation;
- stable online-softmax summaries and two-phase merging;
- toy and K3 logical resource ledgers; and
- failure checks for axis, source, boundary, and state-lifetime errors.

The deterministic transforms are arbitrary and do not claim to solve the
image-text request. The code does not reproduce unpublished training code,
production kernels, GPU timing, physical memory, or pipeline communication.

Run the focused checks from the companion-code root (`code/`):

~~~text
python3 -B _maintenance/chapter_test_runner.py \
  attention_residuals/test_attention_residuals_reference.py
~~~
