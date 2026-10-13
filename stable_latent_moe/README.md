# Chapter 8 Stable LatentMoE companion

This dependency-free reference evaluates the chapter's ten-token,
width-eight Stable LatentMoE fixture. It preserves Chapter 3's twenty route
identities while making the routed path explicit at widths
`8 -> 4 -> 3 -> 4 -> 8`.

The companion checks:

- full-width routing and latent-width routed execution;
- selection with correction bias and weighting without it;
- post-aggregate RMSNorm, including the epsilon-qualified scale statement;
- SiTU-GLU's pointwise SwiGLU limit and intermediate-coordinate bound;
- exactly balanced assignment and one Quantile Balancing coordinate update;
- pooled histogram recovery and invariance to token partitioning;
- separate logical resource ledgers; and
- six injected semantic faults.

Forward fixture F supplies the ten origin and route records and a complete
width-eight row in the chapter. Balancing fixture Q keeps those origins,
current selected tuples, and load vector, but changes most raw score rows;
its margins and weights must be recomputed. The four-token QB example uses
exact decimal inputs. The float64 oracle canonicalizes adjusted routing
scores to 12 decimal places before applying the increasing-index tie rule,
so representational noise in decimal ties does not select a different expert.
Frozen-cutoff counts compare stored margins with the selected endpoint and
report near-equality at a 1e-12 absolute tolerance; midpoint witnesses keep
the same token cutoffs and expose collapsed intervals.

The matrices are deterministic fixtures, not learned Kimi weights. The code
does not reproduce K3 training, GPU kernels, elapsed time, physical memory, or
collective performance.

Run the focused checks from the companion-code root (`code/`):

```text
python3 -B _maintenance/chapter_test_runner.py \
  stable_latent_moe/test_stable_latent_moe_reference.py
```
