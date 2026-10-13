# KDA-aware prefix caching

Original finite models for Chapter 43. No external packages, model weights,
GPU execution, traffic measurements or production K3 bindings are used.

From the companion root:

```sh
python3 -m unittest kda_aware_prefix_caching.test_kda_aware_prefix_caching_reference
```

- `reference.py`: candidate intersection, exact scalar continuation with
  auxiliary history, typed physical allocations, atomic reservations and
  capped FIFO/cyclic scheduling.
- `protocol.py`: complete bundle validation, explicit write completion,
  all-component leases, invalidation and delayed reclamation.
- The test module uses all 512 combinations of three endpoint subsets,
  known scalar outputs at every split and adversarial lifecycle/capacity cases.

All owner calls are serialized abstract operations. Completion is supplied
as an event, not inferred from real device state. The owner receives the
model's expected group set independently of bundle metadata and requires an
exact match. Pending descriptors retain their allocation owner through the
gap between final copy completion and publication; invalidation is required
before retirement. Bundle payloads use fresh
slots and one allocation owner per bundle; shared cross-bundle page ownership
is outside this small protocol model. Immutable scalar values demonstrate
copy isolation. Initialized ranges are supplied facts, not tensor inspection.
The service model counts bounded quanta, not FLOPs, tokens or wall time.
