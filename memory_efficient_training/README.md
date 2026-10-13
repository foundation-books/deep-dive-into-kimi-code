# Chapter 39 deterministic companion

This dependency-free package checks the `mem-k3-37-a` teaching contracts:
versioned object identity, half-open lifetime sweeps, policy typing,
recomputation closure, compression tolerance, restoration deadlines, MoE
saved-output elimination, AttnRes reference counts, pipeline loans, gradient
buffer epochs, owner-directed Muon assembly, and end-to-end refinement.

`finite_fixture.py` builds all sixteen region records with explicit holders,
capacity domains, allocation/readiness/release suborder, and reserved aliases.
The exact/compressed manifests contain 255/243 operations; their HBM peaks are
32/32 and 32/30, and their host peaks (including momentum and metadata) are
14/16 and 10/12. The exact core's four input records cost 32 out/back payload
units; twelve checkpoint seeds add 24. The compressed core costs eight,
plus the same 24 seed units. The helpers in `specialized.py` retain the
payload-frontier counterexample; they do not establish a full physical peak.

`numerical_fixture.py` derives the printed encoder/decoder worksheet,
complete reduce-scatter inputs and receipts, all transformed Muon row slices,
once-only updates, and both parameter publications. The constructed local
diagnostic is 113/4, with scalar tolerance 1/4. It supersedes the old assigned
447/16 checksum; selected-consumer errors do not certify a full K3 step.

`render_fixture.py` prints either complete operation table for optional trace
inspection. Section 39.13 specifies every event family and its invocation order,
with net/prefix accounting; its shorter representation contains the same
schedule and can be expanded without an external runtime.
`schedule_spec.py` independently expands those printed families into every
physical object/domain delta and composes their net/prefix bounds. The focused
test compares every expanded event with the original manifest in both arms;
identity, lease, and numerical checks remain separate in the existing verifier.

It is a semantic and accounting oracle. It does not allocate accelerator
memory, implement a distributed trainer, or support a performance claim.

`backward_lifetimes.py` adds an explicit live-buffer timeline and a backward
consumer-order check. Its constructed 256-byte units show how removing four
saved units can leave the same 19-unit backward peak once replay overlaps
workspace, communication, and gradients. Retaining a redundant weighted output
raises this example's peak to 21 units. These are logical sweep results, not
allocator or accelerator-memory measurements.

Run from `code/`:

```sh
python3 -B -m unittest memory_efficient_training.test_memory_efficient_training_reference
```
