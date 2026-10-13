# Million-Token Agentic-RL Infrastructure

This dependency-free companion implements Chapter 41's finite teaching models.
It is not K3 or AgentENV code, a sandbox deployment, or a performance benchmark.

`reference.py` covers the G/H completion gate, eight- and twelve-token records,
cache compatibility, physical allocation identities, reservation bounds,
device access intervals, and ordered snapshot deltas. `recovery.py` models a
serialized durable authority and an execution gateway that enforces generations.
Its takeover operation assumes old execution has been fenced and drained;
it does not implement a distributed lease, actual GPU fencing, or storage.
Its `admit` method checks the current committed generation, a complete cache
for adopted policy v8, and the 224-byte reservation needed to grow G4 to twelve
tokens. Other reservations and capacity are authoritative inputs from the
serialized resource ledger. Duplicate admission retains one reservation;
takeover releases it only under the model's completed fencing/draining assumption.
Bare transition messages cannot bypass these admission checks. This models a
cache budget; other runtime allocations require separately checked budgets.

`accounting.py` tracks tier-local physical allocation creation and retirement.
Its event test instantiates the phase ledger and independently checks GPU,
host, NVMe, and durable peaks, including old/new spill overlap and restore
scratch. Retirement events assume last readers and transfers have completed;
the separate device-interval oracle checks conflicting buffer lifetimes.

Run from the companion root:

```sh
python3 -B -m unittest million_token_agentic_rl_infrastructure.test_million_token_agentic_rl_infrastructure_reference
```

Tests use independent expected masks, pages, and event intervals; enumerate
small reservation populations and inject crashes at each resume transition.
Snapshot identities are trusted inputs after integrity/provenance validation;
the fixture validator only recognizes the supplied E8/u1 boundary. It cannot
inspect a real filesystem or prove a remote effect occurred. Finite tests supply no empirical
throughput, isolation, million-token scalability, or convergence evidence.
