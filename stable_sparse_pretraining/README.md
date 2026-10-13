# Chapter 14 semantic companion

`reference.py` fixes the chapter's tiny route, gradient, optimizer-owner, and
restart fixtures. It also includes a two-block, shape-scaled finite
Newton--Schulz example that distinguishes Per-Head Muon from one global stacked
transform.

Run the dependency-free checks from the companion-code root (`code/`):

```bash
python3 -B _maintenance/chapter_test_runner.py \
  stable_sparse_pretraining/test_stable_sparse_pretraining_reference.py
```

The companion checks mathematical and state-machine equivalence only. It does
not implement MoonEP, reproduce K3 pretraining, or make throughput, memory,
failure-rate, or model-quality claims.
