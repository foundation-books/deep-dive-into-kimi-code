# Mooncake and Disaggregated Serving

Chapter 42's dependency-free finite reference model checks prefix identity,
full-page allocation, layer timelines, route costs, forecast reservations and
conditional publication. It implements book assumptions, not Mooncake APIs.
Assigned milliseconds are arithmetic inputs, not measured hardware timings.

Run `python3 -m unittest discover -s mooncake_disaggregated_serving -t .` from the
companion root, or run `make smoke` for the standalone suite.

The model assumes exact token/context identity, trusted payloads, serialized
admission and publication, and enforced buffer quiescence before retirement.
Partial groups, expired protection and stale generations must fail publication.
No physical-serving comparison is supplied by these checks; no models, datasets, network
services or accelerator dependencies are required by these checks.

## Implementation mechanisms

The September 20 revision adds six finite teaching modules. Run the same
chapter test command above; its named test module checks both the original
request and the new mechanisms. There are no added dependencies.

| Module | Declared model and expected result |
| --- | --- |
| `transfer_scheduling.py` | Eligible NIC scores 3/8 ms reverse to 11/8 under local load; inverse-score integer shares place a 17-unit folded-tail transfer as local 13 / remote 4. Every hundredth modeled multi-slice call probes. |
| `transfer_qos.py` | One posted unit slice at a time: urgent completion is 11 ms under FIFO and 3 ms under priority, with 11 ms total makespan in both. A posted 10 ms bulk slice cannot be preempted. |
| `prefetch_stopping.py` | `prefetch-41-stop-a`: 20-token prompt, local boundary 4, remote target 16; first-token times 18/15/16 ms reverse to 28/23/16 when suffix costs change. |
| `resource_pooling.py` | An explicitly provisioned 128 MiB arena holds two 64 MiB leases; a third 32 MiB lease either waits/fails under a cap or registers a separate overflow. |
| `ssd_pipeline.py` | A 3+3 MiB partial bucket waits for another 2 MiB; a remote restore has lookup 8--9, staging wait 9--11, disk 11--15, network 15--17 and installation 17--18 ms. |
| `endpoint_cache.py` | Insert A/B/C/D, hit A/C, insert E: evict B after two inspections. Hit D then insert F: evict E after three. Destruction requires separately declared drained resource identities. |

The slice companion omits topology discovery, jitter, EWMA clamps, concurrent
updates and hardware posting. Scores are time estimates, not measured link
capacities. The final slice may be larger than the nominal size after a short
tail is folded into it. The conservation sweep covers all 98 combinations of
1--7 slices, 1--7 units in the final slice, and ordinary/probe allocation.

The priority companion isolates a single queueing decision; the real worker
can dequeue lists and post multiple slices. Promotion counts distinguish age
predicates from full-queue traversal: head-only is not constant total worker
work. The finite arrival test establishes progress only for its supplied
finite arrivals. Device-candidate fallback does not bypass worker shared slots.

The stopping model observes arrived pages at scheduler opportunities. Its
deadline is an ideal continuous-time threshold, while `timeout_reached`
separately checks the source's capped timeout in seconds and strict `>`
comparison. The full retrievable boundary is 16, not the 20-token prompt end.
One millisecond of serial GPU installation is charged after stopping; actual
layer overlap requires a different schedule. Required rank/state groups are
explicit model inputs; their minimum does not establish connector support.

The pooling model assumes an unfragmented arena. Internal Store occupancy and
external leases share its capacity; the external `max_bytes` bound does not
limit all physical registered bytes when overflow and internal occupancy coexist.
It counts successful registration events, not OS calls. Pooling and dummy/real
topology are independent: dummy setup can allocate caller-specific shared
buffers. Eight 64 MiB arenas versus one 128 MiB arena are alternative declared
provisioning choices, without a claim of equal peak concurrency.

The SSD model fixes input and sample orders for reproducibility. Runtime bucket
input is unordered; existing disk keys and failed writes require additional
handling. Bucket metadata/file headers and alignment are omitted from payload
arithmetic. Closing a partial bucket when the next object would exceed its
limit is distinct from reaching that limit exactly. Empty heartbeats do not
magically make incomplete groups eligible. Placement is one-replica selection
over declared preferred/sample/fallback lists. Unknown SSD metrics receive
score 1.0, matching the pin, without guaranteeing available physical disk space.

The SIEVE list uses linear lookup for clarity; the C++ store uses a map and
synchronization. The model counts scan steps, not CPU time, and distinguishes
successive endpoint resources with the same peer name. It does not simulate
queue-pair draining or validate concurrent lifetime safety.

## Frozen sources and optional physical checks

Mooncake implementation study:
`68ce23861197f7df3751274cfdf56771d61534b7`; SGLang HiCache:
`e635577431cbdfb8ce5fafb0fcd8a4ac074062c6`. Exact source paths and hashes are
registered in the book's source-basis metadata. The original chapter's
Mooncake snapshot remains separately cited for its historical explanations.
No compatibility-tested executable pair is claimed.

Optional comparison-design dimensions include static/adaptive NIC allocation, foreground
latency under background load, pooled/per-operation registration, cold/warm
endpoint working sets, bucketed/file-per-key I/O and the three prefetch
policies. Freeze arrivals, payloads, initial cache, runtime branch, topology
and timing endpoints. Observe per-NIC traffic and estimates, priority wait,
registration counts and staging peaks, endpoint misses/creations, disk/network
intervals or jointly usable prefixes respectively. A serving comparison must
also report completed-request goodput. None is a prerequisite for these
deterministic examples or an executed result of the chapter tests.
