# Chapter 11 MoBA reference

`reference.py` is a dependency-free correctness oracle for the Chapter 11
running example. It defines sequence-local blocks, mean-key representatives,
deterministic historical selection, the exact accessible set, dense-masked and
partitioned attention, online-softmax merging, route records, finite-difference
gradient checks, and analytical pair/count ledgers.

It is intentionally not a GPU kernel, a model conversion utility, a training
experiment, or a performance benchmark. The deterministic tie rule and toy
tensors are book conventions.

The reported finite-difference comparison uses the first four rows of the
running $Q,K,V$ fixture, $b=2$, and total block budget 3. For output
$Y$, its scalar objective is

$$
F(Y)=\sum_{t=0}^{3}\sum_{h=0}^{1}\sum_{c=0}^{3}
  (1+t+h/10+c/100)Y_{t,h,c}.
$$

Each coordinate of $Q,K,V$ is perturbed by $\pm10^{-5}$ in turn. The
centered differences from `dense_masked_attention` and
`partitioned_attention` are compared coordinate by coordinate. All eligible
historical blocks are selected on this four-token prefix, so membership stays
fixed under these perturbations. The recorded maximum absolute difference is
$3.56\times10^{-10}$ against a $10^{-8}$ comparison tolerance. This
checks numerical derivatives of two forward paths; it does not call or
validate an automatic-differentiation backward kernel.

Run the unit tests from the companion-code root:

```sh
python3 -B _maintenance/chapter_test_runner.py \
  moba/test_moba_reference.py
```
