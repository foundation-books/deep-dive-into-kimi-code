# LLM Parallelism Foundations

Chapter 33 uses CPU arithmetic to explain what each rank stores, computes,
exchanges, and reconstructs. Run from the companion root:

```sh
python3 -B -m llm_parallelism_foundations.reference
python3 -B -m llm_parallelism_foundations.communication
python3 -B -m unittest llm_parallelism_foundations.test_llm_parallelism_foundations_reference
```

No GPU, networking, external package, or distributed runtime is required.
Exact fractions establish these constructed logical results; they do not test
collective liveness, floating-point kernels, network behavior, or speedup.

| Manuscript | Executable object | Expected result |
| --- | --- | --- |
| 33.2–33.8 collective semantics | `collectives.py` | ordered gather; sum (11,22); scattered (11),(22); tagged destination buckets |
| 33.3 DP and the Volume I refresher | `mean_gradient`, `sgd` | equal masses (3,5); unequal masses (4,6); update (9.7,19.5) |
| 33.4 TP | Chapter 38 matrices, `tp_forward_backward`, `tp_both_active` | Case A at (1,1): partials (3,7)+(-2,4)=(1,11), input gradient (6,6); Case B at (1,-1): output (-2,4), input gradient (2,0) |
| 33.5 PP | `flush_schedule`, `validate_schedule` | one microbatch: 1/2; four: 4/5 busy stage-slot fraction |
| 33.6 EP | `routes`, `expert_round_trip`, `combine` | A returns 5 to origin 0; B returns 3 to origin 1 |
| 33.7 CP | `causal_prefix_means`, `attention_shards` | position 5 returns 3; position 8 returns 9/2 |
| Chapter 38 comparison retained in the companion | `kda_fixture` | document order 1, reversed order -3; 34.4 uses the four-segment fixture below |
| 33.9 TP × DP | `check_axis_group`, `four_rank_forward`, `four_rank_step` | each TP group reconstructs (1,11); matching parameter shards update to 9.7 and 19.5 in both replicas |

The communication expansion adds these exact fixtures:

| Manuscript | Executable object | Expected result |
| --- | --- | --- |
| 33.2 four-rank collectives | `four_rank_partials`, `tagged_gather`, existing `collectives` | sum `(1111,...,8888)`; rank 1 owns `(3333,4444)`; exact reduce-scatter/all-gather composition |
| 35.9 row-local depth mixing | `depth_sandwich` | completed rows produce `10t`; premature nonlinear mixing produces `12t` |
| 33.4 all-gather pullback | `allgather_pullback` | independent quadratic-loss finite differences match sum-reduce-scatter; mean divides by four |
| 33.6 unequal route buckets | `irregular_dispatch`, `irregular_round_trip` | counts `(2,1,1,0)`; restored A/B remain 5/3 |
| 34.4 exclusive affine scan | `affine_segments`, `exclusive_prefixes` | incoming `(0,1,7,5)`, final 10; identity preserves a nonzero initial state |
| 33.8 completion | `communication_traces.py` | reject wrong order/group/count/type, early read, and early overwrite |
| 33.8 one-transfer cost | `transfer_microseconds` | 4000 bytes: 3.2 us; 4,000,000 bytes: 203 us under stipulated latency/bandwidth |

The rational depth mixer is a constructed nonlinear analogue, not AttnRes.
Packing keeps complete source pairs together for each token row; conceptual
`[S,T,d]` shape alone does not determine a backend's contiguous slice order.
The irregular extension keeps the original route values and weights but assigns
expert homes `(0,0,2,1)` for experts 1–4. Route ordinals 0/1 travel with the
origin, token, expert, weight, and payload. Tests distinguish harmless whole-record
reordering from payload/tag misassociation, which requires a numerical oracle.
The lifetime checker validates only the stipulated finite dependency chain.

The TP calculation imports the existing Chapter 38 matrices rather than
duplicating their source. Cases A and B retain the output-row split of `W1`
and matching input-column split of `W2`; the trace exposes preactivations,
pre-ReLU derivatives, chosen multipliers, and input contributions. At zero
the multiplier is zero, a convention rather than an ordinary derivative.
Case C retains B's input but splits `W1` by input columns: separately rectified
partials sum to `(1,3,0,2)` instead of the correct `(0,0,0,2)`. The tests
independently differentiate the dense loss at smooth Case B and reject treating
its input gradient as Case A's. The Muon gather–transform–slice witness remains in
`moonlight_muon/`; Chapter 33 explains its communication prerequisites and
does not implement another matrix optimizer. The new attention witness fixes
all logits to zero, so causal softmax becomes an exactly checkable prefix mean.

Notation crosswalk: the text numbers sequence positions 1–8, while Python
indices are 0–7. Expert identifiers stay 1–4 here and in Chapter 3; subtract
one when comparing with Chapter 38's expert identifiers. The TP witness uses
column vectors (`y = W x`), so output features index matrix rows; Chapter 1's
row-vector convention (`Y = X W`) transposes that description. The overview
keeps Chapter 1's hidden width four and head count two; the isolated TP witness
instead declares its own two-entry input and four-entry hidden activation.

The tests compare independent serial objectives or derivatives and inject
missing participants, reversed shard order, wrong equal-size groups, duplicated
or misidentified routes, missing positions, stage conflicts, and premature
updates. They cover the declared finite witnesses, not every possible layout.

The affine-map question and its complete answer are merged into Exercise 34.4;
the four-rank depth-mixer question and answer are merged into Exercise 35.7(a).
Their checks remain in these stable modules. Chapter 33 keeps ordinary addition
prefixes and an optional all-gather-gradient derivation, with Exercise 33.4
now devoted to that derivation. No companion prose is required to understand them.
