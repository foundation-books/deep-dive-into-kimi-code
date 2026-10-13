# Chapter 40: Checkpoint Engine and Online Weight Publication

This dependency-free companion models the chapter's 28-coordinate snapshot,
row-to-column reconstruction, ideal pipeline recurrence, allocation lifetimes,
and a synchronous whole-replica publication state machine.

Run from the companion root:

```sh
python3 -m unittest checkpoint_engine_online_weight_publication.test_checkpoint_engine_online_weight_publication_reference
```

The tests use independently enumerated destination columns and invalid traces:
missing and mixed values, stale epochs, premature activation, lost acknowledgments,
and retirement while an old request remains live. Model values are exact integers
representable in float32. Dictionary copies stand for immutable version handles;
callers must not mutate the model's internal dictionaries.

This is a finite teaching oracle, not Checkpoint Engine, a distributed controller,
or a benchmark. Device completion and epoch authority are explicit assumptions.
The state machine models both workers as one installation predicate, uses no
persistent storage, and does not model cache compatibility, actual crash-durable
writes, rollback, DMA cancellation, or real network membership. A restart loses
requests and keeps admission closed until reconstruction. No RL quality or
hardware performance result is claimed.
