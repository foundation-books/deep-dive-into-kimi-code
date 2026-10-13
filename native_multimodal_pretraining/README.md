# Chapter 16 companion: Native Multimodal Pretraining

This dependency-free companion makes the chapter's small studio examples
executable.  It checks exact serialization arrays and checksums for all five
studio records, target masks, media ownership, coordinate transformations and
negative cases, record/token/target/visual/compute mixtures, source-reuse and
rejection conservation, spatial and temporal target validity, and four
complete pedagogical stage ledgers with precision states, gates, transitions,
and handoffs.

The historical preparation-first stage fixture remains in this code for
comparison with the print chapter's joint-from-start illustration.
`reference_joint_stages.py::joint_stage_fixture` now checks all four printed
stages, their low-grid record counts and budgets, and the exact changed/fixed
fields at the first transition. It records retention as not applicable at
initialization and leaves measured gate verdicts unset. The existing
`STAGE_INPUTS` and recorded transition campaign still describe the preserved
staged example. The new fixture and historical results are deterministic
contract checks, not training observations.

Run the checks from the companion-code root (`code/`):

```text
python3 -B _maintenance/chapter_test_runner.py \
  native_multimodal_pretraining/test_native_multimodal_pretraining_reference.py
```

The values are teaching fixtures.  They do not implement Moonshot AI's private
data pipeline, train a model, reproduce Kimi-VL, Kimi K2.5, or Kimi K3
pretraining, or establish quality, throughput, memory, energy, or production
cost. Source-reported claims and revisions remain in the book's citations.
