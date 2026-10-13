# Chapter 13 reference semantics

This companion contains dependency-free logical oracles for *MuonClip and
Training Stability*. It is not an official K2 implementation and does not
model fused attention kernels, distributed topology, optimizer sharding, or
training quality.

`reference.py` keeps five records distinct: the forward observation, the
inherited post-Muon candidate, the post-clip commit, a hypothetical same-state
replay, and the next forward checkpoint. This is one clipping step followed
by an observation, with candidate values supplied as an optimizer stand-in.
Exact `Fraction` arithmetic protects
the chapter's 400/80, 100/80, and 96/72 values.
The focused test also constructs the full two-token causal score matrices,
checks all six admissible entries and their \(1/6\) tail fraction, and verifies
that the separately supplied candidate uses the stored factors.

Run the focused checks from the companion-code root (`code/`):

```bash
python3 -B _maintenance/chapter_test_runner.py \
  muonclip_training_stability/test_muonclip_training_stability_reference.py
```

The experiment wrapper in `experiments/muonclip_training_stability/` records
the existing deterministic results. It supplies no matched local training
threshold sweep or systems measurement.

## Outline-to-implementation traceability

The `C12-I` identifiers are historical labels from the pre-renumbering outline;
the section references below use printed Chapter 13.

| Planned term | Implemented anchor |
| --- | --- |
| Scaled pre-softmax head score | `def:muonclip-scaled-head-score` |
| Per-head positive maximum | `def:muonclip-scaled-head-score` |
| Idealized versus finite Muon direction | `def:muonclip-exact-finite` |
| Same-state replay contraction | `prop:muonclip-mha-contraction` and `prop:muonclip-mla-contraction` |
| Lagged feedback transition | `def:muonclip-lagged-transition` |
| C12-I1 — Score convention | Sections 13.2 and 13.12 |
| C12-I2 — Endpoint provenance | Sections 13.2 and 13.8 |
| C12-I3 — Suppressive range | `def:muonclip-trigger-factor` |
| C12-I4 — No-op below threshold | `inv:muonclip-mha-locality` |
| C12-I5 — Per-head locality | `inv:muonclip-mha-locality` |
| C12-I6 — MHA exponent conservation | `prop:muonclip-mha-contraction` |
| C12-I7 — MLA path conservation | `inv:muonclip-mla-paths` |
| C12-I8 — Shared-key protection | `inv:muonclip-mla-paths` |
| C12-I9 — Current-step noninterference | `inv:muonclip-current-step` |
| C12-I10 — Replay boundary | Sections 13.6--13.8 |
| C12-I11 — Inherited-Muon integrity | Sections 13.1, 13.4, and 13.8 |
| C12-I12 — Replica consensus | `prop:muonclip-consensus` |
| C12-I13 — Metric separation | Sections 13.2 and 13.10 |
| C12-I14 — Threshold provenance | Sections 13.10--13.11 |
| C12-I15 — Hypothesis boundary | Section 13.4 |
| C12-I16 — Evidence closure | Section 13.11 |
| C12-I17 — Terminal exercises | Section 13.13 and the Volume I answer appendix |
