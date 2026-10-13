# Chapter 15 semantic companion

`reference.py` checks the answer-only ORBIT task against emitted bindings,
keeps stipulated graph arithmetic separate, and computes exact record,
input-atom, target, and curriculum ledgers. `packing.py` holds the three
16-atom boundary policies and their first active prediction rows.

Run the dependency-free checks from the companion-code root (`code/`):

```bash
python3 -B _maintenance/chapter_test_runner.py \
  long_context_pretraining/test_long_context_pretraining_reference.py
```

The companion validates symbolic construction and arithmetic only. It does
not train a model, implement a Moonshot AI pipeline, reproduce K2 or K3
pretraining, or measure model quality, throughput, or memory.
