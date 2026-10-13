# Chapter 39 deterministic companion

The fixed four-example fixture checks geometry, packing, attention and its
gradients, objective weighting, bounded admission, lifetimes, version identity,
and a two-rank/four-chunk interleaved schedule. All durations are normalized
synthetic units; transfers in the complete trace have zero cost. These modules
are semantic and accounting oracles, not a K3 trainer or performance benchmark.

From `code/`:

```sh
python3 -B -m unittest multimodal_encoder_input_pipeline_optimization.test_multimodal_encoder_input_pipeline_optimization_reference
```
