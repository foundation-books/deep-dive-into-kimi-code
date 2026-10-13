"""MOBA correctness and accounting experiments on the book's fixed CPU inputs."""

from __future__ import annotations

from types import ModuleType

from moba import reference as chapter_reference
from ._data import protocol


def forward_oracle(reference: ModuleType) -> dict[str, object]:
    """Compare partitioned execution with the direct accessible-set oracle."""

    queries, keys, values = reference.running_tensors()
    block_size = reference.RUNNING_BLOCK_SIZE
    budget = reference.RUNNING_TOTAL_BLOCK_BUDGET
    direct = reference.dense_masked_attention(queries, keys, values, block_size, budget)
    grouped = reference.partitioned_attention(queries, keys, values, block_size, budget)
    error = reference.max_abs_tensor_difference(direct, grouped)
    return {
        "status": "pass" if error <= 1e-12 else "fail",
        "protocol": protocol(
            input_description="ten tokens, two heads, head width four, b=2, total k=3",
            reference="direct dense-masked attention over each exact accessible set",
            comparison="current/history partition execution with online-state merging",
            metric="maximum absolute output-coordinate error",
            tolerance="1e-12 in Python double precision",
            claim_boundary="scaled-down forward semantics, not a GPU-kernel result",
        ),
        "output_shape": [10, 2, 4],
        "max_abs_error": error,
        "tolerance": 1e-12,
        "final_query_outputs": direct[-1],
    }


def backward_oracle(reference: ModuleType) -> dict[str, object]:
    """Compare finite-difference Q/K/V derivatives on a stable small route case."""

    queries, keys, values = reference.running_tensors()
    inputs = (queries[:4], keys[:4], values[:4])
    block_size = reference.RUNNING_BLOCK_SIZE
    budget = reference.RUNNING_TOTAL_BLOCK_BUDGET
    direct = reference.finite_difference_gradients(
        reference.dense_masked_attention, *inputs, block_size, budget
    )
    grouped = reference.finite_difference_gradients(
        reference.partitioned_attention, *inputs, block_size, budget
    )
    errors = tuple(
        reference.max_abs_tensor_difference(left, right)
        for left, right in zip(direct, grouped, strict=True)
    )
    maximum = max(errors)
    return {
        "status": "pass" if maximum <= 1e-8 else "fail",
        "protocol": protocol(
            input_description="four-token prefix with stable routes and centered step 1e-5",
            reference="finite differences of the direct dense-masked scalar objective",
            comparison="finite differences of the partitioned-attention objective",
            metric="maximum absolute Q, K, or V derivative error",
            tolerance="1e-8; routing membership must stay unchanged under perturbations",
            claim_boundary="numerical toy derivatives, not autograd or training validation",
        ),
        "q_k_v_max_abs_errors": errors,
        "max_abs_error": maximum,
        "finite_difference_step": 1e-5,
    }


def causality_and_isolation(
    reference: ModuleType,
) -> tuple[dict[str, object], dict[str, object]]:
    """Check future perturbation and separate-sequence evaluation."""

    queries, keys, values = reference.running_tensors()
    block_size = reference.RUNNING_BLOCK_SIZE
    budget = reference.RUNNING_TOTAL_BLOCK_BUDGET
    baseline = reference.dense_masked_attention(
        queries, keys, values, block_size, budget
    )
    perturbed = reference.dense_masked_attention(
        queries,
        reference.mutate_suffix(keys, 6, 7.0),
        reference.mutate_suffix(values, 6, -9.0),
        block_size,
        budget,
    )
    prefix_error = reference.prefix_difference(baseline, perturbed, 5)
    causal: dict[str, object] = {
        "status": "pass" if prefix_error == 0.0 else "fail",
        "protocol": protocol(
            input_description="positions 6-9 perturbed after a fixed six-token prefix",
            reference="unchanged outputs at query positions 0-5",
            comparison="baseline and future-perturbed prefix outputs",
            metric="maximum absolute prefix output error",
            tolerance="exact zero in the deterministic Python oracle",
            claim_boundary="accessible-set causality for the toy oracle only",
        ),
        "last_compared_query": 5,
        "max_abs_prefix_error": prefix_error,
    }

    short = (queries[:6], keys[:6], values[:6])
    altered = (
        queries,
        reference.mutate_suffix(keys, 0, 3.0),
        reference.mutate_suffix(values, 0, -4.0),
    )
    batched = reference.batch_attention((short, altered), block_size, budget)
    independent = reference.dense_masked_attention(*short, block_size, budget)
    isolation_error = reference.max_abs_tensor_difference(batched[0], independent)
    isolation: dict[str, object] = {
        "status": "pass" if isolation_error == 0.0 else "fail",
        "protocol": protocol(
            input_description="one six-token sequence paired with an unrelated ten-token sequence",
            reference="the six-token sequence evaluated independently",
            comparison="the first result from the sequence-isolated batch wrapper",
            metric="maximum absolute first-sequence output error",
            tolerance="exact zero in the deterministic Python oracle",
            claim_boundary="logical sequence isolation, not packed GPU layout validation",
        ),
        "max_abs_error": isolation_error,
    }
    return causal, isolation


def boundary_and_limit_checks(
    reference: ModuleType,
) -> tuple[dict[str, object], dict[str, object]]:
    """Check a non-divisible tail and the full-attention limiting budget."""

    queries, keys, values = reference.running_tensors()
    blocks = reference.partition_blocks(9, 4)
    representatives = reference.mean_key_representatives(keys[:9], 4)
    partial_error = max(
        abs(observed - expected)
        for observed_head, expected_head in zip(
            representatives[-1], keys[8], strict=True
        )
        for observed, expected in zip(observed_head, expected_head, strict=True)
    )
    partial: dict[str, object] = {
        "status": "pass"
        if blocks == ((0, 1, 2, 3), (4, 5, 6, 7), (8,)) and partial_error == 0.0
        else "fail",
        "protocol": protocol(
            input_description="N=9, b=4 with no padding member in the last block",
            reference="blocks of sizes 4, 4, and 1; last mean equals the real last key",
            comparison="constructed intervals and computed last representative",
            metric="exact intervals and maximum absolute representative error",
            tolerance="exact intervals and exact zero floating-point error",
            claim_boundary="partial-tail book semantics, not padding-kernel behavior",
        ),
        "blocks": blocks,
        "block_sizes": [len(block) for block in blocks],
        "last_representative_max_abs_error": partial_error,
    }

    full_budget = len(reference.partition_blocks(len(queries), 2))
    sparse_limit = reference.dense_masked_attention(
        queries, keys, values, 2, full_budget
    )
    dense = reference.full_causal_attention(queries, keys, values)
    limit_error = reference.max_abs_tensor_difference(sparse_limit, dense)
    full_limit: dict[str, object] = {
        "status": "pass" if limit_error <= 1e-12 else "fail",
        "protocol": protocol(
            input_description="ten tokens with every completed history block selected",
            reference="ordinary dense causal attention",
            comparison="MoBA accessible-set attention at total block budget five",
            metric="maximum absolute output-coordinate error",
            tolerance="1e-12 in Python double precision",
            claim_boundary="mathematical dense limit, not a kernel-mode equivalence claim",
        ),
        "total_block_budget": full_budget,
        "max_abs_error": limit_error,
    }
    return partial, full_limit


def merge_check(reference: ModuleType) -> dict[str, object]:
    """Check the chapter's exact 27/7 online-softmax merge."""

    values = reference.online_merge_microexample()
    error = abs(values["output"] - values["expected"])
    return {
        "status": "pass" if error <= 1e-12 else "fail",
        "protocol": protocol(
            input_description="logits (0, ln 2) and (ln 4), scalar values (1,3) and 5",
            reference="direct global weights proportional to (1,2,4), giving 27/7",
            comparison="merged local max/denominator/numerator states",
            metric="absolute scalar-output error",
            tolerance="1e-12 in Python double precision",
            claim_boundary="exact algebraic merge example, not a fused-kernel reduction study",
        ),
        **values,
        "abs_error": error,
    }


def route_and_cost_checks(
    reference: ModuleType,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Check route identity/load conservation and analytical pair counts."""

    queries, keys, _ = reference.running_tensors()
    block_size = reference.RUNNING_BLOCK_SIZE
    budget = reference.RUNNING_TOTAL_BLOCK_BUDGET
    routes = reference.block_routes(queries, keys, block_size, budget)
    identities = {(route.query, route.head, route.block) for route in routes}
    loads = reference.route_loads(routes, 5)
    conserved = len(routes) == len(identities) == sum(loads) == 28
    route_result: dict[str, object] = {
        "status": "pass" if conserved else "fail",
        "protocol": protocol(
            input_description="all historical query-head-block routes in the ten-token trace",
            reference="28 unique route identities and a load sum of 28",
            comparison="expanded routes, identity set, and block-load reduction",
            metric="exact route, identity, and load counts",
            tolerance="exact integer equality",
            claim_boundary="logical route conservation, not physical permutation traffic",
        ),
        "historical_route_records": len(routes),
        "unique_route_identities": len(identities),
        "load_sum": sum(loads),
    }
    load_result: dict[str, object] = {
        "status": "pass" if loads == (14, 7, 5, 2, 0) else "fail",
        "protocol": protocol(
            input_description="the same 28 historical routes grouped by five source blocks",
            reference="hand-audited load vector (14,7,5,2,0)",
            comparison="computed route counts by block identifier",
            metric="exact five-entry integer load vector",
            tolerance="exact integer equality",
            claim_boundary="toy route skew only; no scheduler or utilization claim",
        ),
        "historical_block_loads": loads,
        "maximum_to_mean_load_ratio": max(loads) / (sum(loads) / len(loads)),
    }
    ledger = reference.pair_and_cost_ledger(queries, keys, block_size, budget)
    expected = {
        "dense_causal_query_key_pairs": 110,
        "moba_query_key_pairs": 86,
        "gate_query_block_pairs": 100,
        "historical_route_records": 28,
        "mandatory_current_assignments": 20,
    }
    matches = all(ledger[name] == value for name, value in expected.items())
    cost_result: dict[str, object] = {
        "status": "pass" if matches else "fail",
        "protocol": protocol(
            input_description="N=10, H=2, dh=4, b=2, total k=3 running trace",
            reference="hand-derived causal, accessible, gate, route, and storage counts",
            comparison="analytical ledger emitted by the reference",
            metric="exact integer counts and derived realized density/sparsity",
            tolerance="exact integer equality; serialized double precision for ratios",
            claim_boundary="operation and storage counts only; this experiment performs no timing",
        ),
        "values": ledger,
        "performance_claim_supported": False,
    }
    return route_result, load_result, cost_result


def run_all_checks(
    reference: ModuleType | None = None,
) -> dict[str, dict[str, object]]:
    """Run ten fixed MoBA correctness and accounting comparisons."""

    api = reference or chapter_reference
    causality, isolation = causality_and_isolation(api)
    partial, full_limit = boundary_and_limit_checks(api)
    route_result, load_result, cost_result = route_and_cost_checks(api)
    return {
        "RESULT_MOBA_FORWARD_ORACLE": forward_oracle(api),
        "RESULT_MOBA_BACKWARD_ORACLE": backward_oracle(api),
        "RESULT_MOBA_CAUSALITY": causality,
        "RESULT_MOBA_BATCH_ISOLATION": isolation,
        "RESULT_MOBA_PARTIAL_BLOCK": partial,
        "RESULT_MOBA_FULL_LIMIT": full_limit,
        "RESULT_MOBA_ONLINE_MERGE": merge_check(api),
        "RESULT_MOBA_ROUTE_CONSERVATION": route_result,
        "RESULT_MOBA_COST_LEDGER": cost_result,
        "RESULT_MOBA_ROUTE_LOAD": load_result,
    }
