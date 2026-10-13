# Fleet Scheduling, Admission, and Failure Recovery

This original, dependency-free finite reference makes Chapter 44's assignment,
memory, service and recovery accounting executable. It does not implement a
production controller, consensus, transport, failure detection or a model.

From the companion root:

```sh
python3 -m unittest fleet_scheduling_admission_failure_recovery.test_fleet_scheduling_admission_failure_recovery_reference
```

`reference.py` supplies fixed SHA-256 rendezvous assignment, acquired-candidate
placement, fallback/readiness gates and exact arithmetic. IDs use UTF-8 with a
NUL separator; embedded NUL is forbidden. Physical cluster IDs are unique.
The six-session secondary table is a separate declared oracle, not hash output.

`ledger.py` assumes a serialized owner and epoch enforcement at every sink.
Memory is in integral GiB fixture units. Physical grants persist across slots;
immediate service quanta are charged to a named slot and never refunded after
execution. One memory grant owns a request attempt; distinct service quantum
IDs provide dispatch deduplication. Released allocation IDs remain tombstoned.
Prepared timeout is retire then release; committed cancellation additionally
needs actual quiescence. A caller cannot infer that fact from a timeout.
Immutable Contract values bound an attempt's total work and allowed classes.
Dispatch credits service at its ten-ms slot endpoint, strictly before the
absolute deadline. Per-attempt counters persist across slots; retries charge
re-execution again without enlarging the logical output cap or deadline.
Output publication takes an explicit time; stored-payload delivery remains
possible after expiry. The six-sequence reconnect fork uses its own cap of
eight, separate from the main fixture's four-unit contracts.

`simulation.py` runs the declared six-session recovery from admission at 10 ms.
Each request retains one GiB, gets protected FIFO replay, then suffix/decode at
the next endpoint. Healthy S traffic completes at 20 ms. Completion records
include state, work and occupancy; final pools retain background four GiB each.
The trace's one output is a marker, not generated language.
Request age and admission wait are distinct fields. Transition calls execute
with or without Python assertions; a subprocess regression compares both modes.

Tests use independent arithmetic, all 729 probability assignments, fixed hash
outputs, monotonicity, and adversarial grant/output transitions. These finite
checks have no statistical or production-latency interpretation. Optional
comparison methods are designs, not implemented benchmarks or author commitments.
