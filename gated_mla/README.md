# Chapter 6 companion: Gated MLA

`reference.py` is the dependency-free public entry point for the chapter's
semantic oracle. `reference_types.py` defines the typed records,
`reference_runtime.py` implements reconstructed and compact-latent causal
paths plus cache continuation, and `reference_proofs.py` isolates the exact
key/value identities. Together they cover the shared NoPE key branch, the
channel-wise output gate, cache identity, and logical cache ledgers.

Run the focused checks from the companion-code root (`code/`):

```sh
python3 -B _maintenance/chapter_test_runner.py \
  gated_mla/test_gated_mla_reference.py
```

The oracle uses Python double precision and toy dimensions. It does not claim
to reproduce the production K3 kernel, its physical cache allocation, timing,
or the report's FP32 attention-output training path.
