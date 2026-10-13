# Chapter 9 MoonViT and native-multimodality companion

This dependency-free reference makes the chapter's coordinate and identity
contracts executable. It reproduces the pinned public image resize-and-padding
arithmetic, row-major variable grids, packed offsets and isolation masks,
two-by-two spatial rearrangement, report-level full-versus-factorized attention
counts, explicit temporal tail handling, a bias-free toy projector, ordered
media splicing, the ten-position running trace, and seven fault oracles.

The fixtures are not learned Kimi weights. They do not reproduce MoonViT-V2,
the unreleased training path, public video preprocessing, GPU kernels, quality,
memory traffic, latency, or throughput.

Run the focused checks from the companion-code root (`code/`):

```text
python3 -B _maintenance/chapter_test_runner.py \
  moonvit_native_multimodality/test_moonvit_native_multimodality_reference.py
```
