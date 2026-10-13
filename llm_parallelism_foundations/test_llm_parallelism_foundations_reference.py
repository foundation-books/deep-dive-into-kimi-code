"""Check serial equivalence and reject semantically wrong distributed examples."""

from dataclasses import replace
from fractions import Fraction as F
import unittest

from distributed_execution_k3_class_models.semantics import (
    compose_fragments,
    kda_fixture,
    matvec,
    relu,
    tp_fixture,
)
from .collectives import (
    all_gather,
    all_reduce,
    all_to_all,
    broadcast,
    gather,
    reduce_scatter,
    scatter,
    send_receive,
)
from .reference import (
    attention_shards,
    causal_prefix_means,
    check_axis_group,
    combine,
    expert_home,
    expert_round_trip,
    four_rank_forward,
    four_rank_step,
    mean_gradient,
    routes,
    sgd,
    tp_both_active,
    tp_forward_backward,
)
from .schedule import flush_schedule, validate_schedule
from .communication import (
    GROUP,
    X,
    affine_segments,
    allgather_pullback,
    depth_sandwich,
    exclusive_prefixes,
    four_rank_partials,
    irregular_dispatch,
    irregular_round_trip,
    tagged_gather,
    transfer_microseconds,
)
from .communication_traces import (
    CollectiveCall,
    buffer_trace,
    validate_buffer_trace,
    validate_collective_order,
)


class FoundationsTests(unittest.TestCase):
    def test_collectives_preserve_values_order_and_destination(self) -> None:
        partials = {0: (F(1), F(2)), 1: (F(10), F(20))}
        self.assertEqual(all_reduce(partials, (0, 1)), {0: (11, 22), 1: (11, 22)})
        self.assertEqual(reduce_scatter(partials, (0, 1)), {0: (11,), 1: (22,)})
        shards = {0: (1, 2), 1: (3, 4)}
        self.assertEqual(all_gather(shards, (0, 1)), {0: (1, 2, 3, 4), 1: (1, 2, 3, 4)})
        self.assertNotEqual(all_gather(shards, (1, 0))[0], (1, 2, 3, 4))
        self.assertEqual(broadcast((7, 8), 0, (0, 1)), {0: (7, 8), 1: (7, 8)})
        self.assertEqual(send_receive("activation m0", 0, 1), {1: "activation m0"})
        buckets = {0: {0: ("A1",), 1: ("A3",)}, 1: {0: ("B2",), 1: ("B4",)}}
        self.assertEqual(
            all_to_all(buckets, (0, 1)), {0: ("A1", "B2"), 1: ("A3", "B4")}
        )
        for group in ((0, 0), (0,), (0, 2)):
            with self.assertRaises(ValueError):
                all_reduce(partials, group)

    def test_dp_matches_serial_objective_and_one_update(self) -> None:
        gradients = {0: (F(1), F(3)), 1: (F(5), F(7))}
        equal = mean_gradient(gradients, {0: 1, 1: 1}, (0, 1))
        self.assertEqual(equal, (3, 5))
        self.assertEqual(sgd((F(10), F(20)), equal), (F(97, 10), F(39, 2)))
        weighted = mean_gradient(gradients, {0: 1, 1: 3}, (0, 1))
        # The serial objective is the mean of four sample-linear losses.
        samples = (gradients[0], gradients[1], gradients[1], gradients[1])
        serial = tuple(sum(g[j] for g in samples) / len(samples) for j in range(2))
        self.assertEqual(weighted, serial)
        self.assertEqual(weighted, (4, 6))
        self.assertEqual(sgd((F(10), F(20)), weighted), (F(48, 5), F(97, 5)))
        self.assertNotEqual(weighted, equal)
        self.assertNotEqual(all_reduce(gradients, (0, 1))[0], equal)
        with self.assertRaises(ValueError):
            mean_gradient({0: gradients[0]}, {0: 1, 1: 1}, (0, 1))

        # Owner sharding changes the final gathered object to updated parameters.
        sums = reduce_scatter(gradients, (0, 1))
        self.assertEqual(sums, {0: (6,), 1: (10,)})
        updated = {
            owner: sgd((F(old),), tuple(g / 2 for g in sums[owner]))
            for owner, old in enumerate((10, 20))
        }
        self.assertEqual(all_gather(updated, (0, 1))[0], (F(97, 10), F(39, 2)))
        self.assertNotEqual(all_gather(gradients, (0, 1))[0], (F(97, 10), F(39, 2)))

    def test_tp_forward_and_backward_match_dense_reference(self) -> None:
        result = tp_both_active()
        self.assertEqual(result["preactivation"], ((3, 7), (0, 2)))
        self.assertEqual(result["local"], ((3, 7), (0, 2)))
        self.assertEqual(result["partials"], ((3, 7), (-2, 4)))
        self.assertEqual(result["output"], (1, 11))
        self.assertEqual(result["hidden_gradient"], ((1, 1), (3, 1)))
        self.assertEqual(result["relu_multiplier"], ((1, 1), (0, 1)))
        self.assertEqual(result["preactivation_gradient"], ((1, 1), (0, 1)))
        self.assertEqual(result["input_partials"], ((4, 6), (2, 0)))
        self.assertEqual(result["input_gradient"], (6, 6))
        _, first, second = tp_fixture()
        eps = F(1, 1000)
        # Use a separate smooth point to check the full transpose formula.
        x = (F(2), F(1))
        h = matvec(first, x)
        dy = (F(1), F(1))
        dz = tuple(
            sum(second[k][j] * dy[k] for k in range(2)) if h[j] > 0 else F(0)
            for j in range(4)
        )
        analytic = tuple(sum(first[j][i] * dz[j] for j in range(4)) for i in range(2))
        for i in range(2):
            plus = tuple(v + (eps if j == i else 0) for j, v in enumerate(x))
            minus = tuple(v - (eps if j == i else 0) for j, v in enumerate(x))
            difference = (
                sum(matvec(second, relu(matvec(first, plus))))
                - sum(matvec(second, relu(matvec(first, minus))))
            ) / (2 * eps)
            self.assertEqual(difference, analytic[i])

    def test_tp_changed_input_has_different_activation_gradient(self) -> None:
        x = (F(1), F(-1))
        result = tp_forward_backward(x)
        self.assertEqual(result["preactivation"], ((-1, -1), (-2, 2)))
        self.assertEqual(result["local"], ((0, 0), (0, 2)))
        self.assertEqual(result["partials"], ((0, 0), (-2, 4)))
        self.assertEqual(result["hidden_gradient"], ((1, 1), (3, 1)))
        self.assertEqual(result["relu_multiplier"], ((0, 0), (0, 1)))
        self.assertEqual(result["preactivation_gradient"], ((0, 0), (0, 1)))
        self.assertEqual(result["input_partials"], ((0, 0), (2, 0)))
        self.assertEqual(result["input_gradient"], (2, 0))
        self.assertNotEqual(
            result["input_gradient"], tp_both_active()["input_gradient"]
        )

        def dense_loss(value: tuple[F, ...]) -> F:
            # Independently reconstruct sum(y) from the printed dense matrices.
            a, b = value
            return (
                max(F(0), a + 2 * b)
                + max(F(0), 3 * a + 4 * b)
                + (3 * max(F(0), -a + b) + max(F(0), 2 * a))
            )

        epsilon = F(1, 1000)  # Case B is away from every ReLU kink.
        numeric = tuple(
            (
                dense_loss(
                    tuple(v + (epsilon if j == i else 0) for j, v in enumerate(x))
                )
                - dense_loss(
                    tuple(v - (epsilon if j == i else 0) for j, v in enumerate(x))
                )
            )
            / (2 * epsilon)
            for i in range(2)
        )
        self.assertEqual(numeric, result["input_gradient"])

    def test_partial_preactivations_reduce_before_relu(self) -> None:
        _, first, _ = tp_fixture()
        contributions = (
            tuple(row[0] for row in first),
            tuple(-row[1] for row in first),
        )
        self.assertEqual(contributions, ((1, 3, -1, 2), (-2, -4, -1, 0)))
        completed = all_reduce(dict(enumerate(contributions)), (0, 1))[0]
        self.assertEqual(relu(completed), (0, 0, 0, 2))
        premature = all_reduce(
            {r: relu(partial) for r, partial in enumerate(contributions)}, (0, 1)
        )[0]
        self.assertEqual(premature, (1, 3, 0, 2))
        self.assertNotEqual(premature, relu(completed))
        self.assertEqual(relu((F(-2), F(0), F(3))), (0, 0, 3))

    def test_pipeline_dependencies_capacity_and_update_boundary(self) -> None:
        for count, expected in ((1, F(1, 2)), (4, F(4, 5))):
            events = flush_schedule(count)
            self.assertEqual(
                validate_schedule(events, count, 2 * (count + 1)), expected
            )
            with self.assertRaises(ValueError):
                validate_schedule(events, count, 2 * count)
        events = list(flush_schedule(4))
        self.assertEqual(
            sorted(
                (event.slot, event.stage, event.direction)
                for event in events
                if event.microbatch == 3
            ),
            [(3, 0, "F"), (4, 1, "F"), (5, 1, "B"), (6, 0, "B")],
        )
        events[0] = replace(events[0], slot=9)
        with self.assertRaises(ValueError):
            validate_schedule(events, 4, 10)
        with self.assertRaises(ValueError):
            validate_schedule(flush_schedule(4)[:-1], 4, 10)
        early_consumer = list(flush_schedule(4))
        early_consumer[0] = replace(early_consumer[0], slot=1)
        early_consumer[2] = replace(early_consumer[2], slot=0)
        with self.assertRaisesRegex(ValueError, "producer"):
            validate_schedule(early_consumer, 4, 10)

    def test_expert_routes_return_once_to_their_own_origin(self) -> None:
        selected = routes()
        self.assertEqual(expert_round_trip(), {(0, "A"): 5, (1, "B"): 3})
        self.assertEqual(sum(r.origin == expert_home(r.expert) for r in selected), 2)
        outputs = [(r, r.expert * r.value) for r in selected]
        self.assertEqual(combine(outputs[::-1], selected), expert_round_trip())
        for faulty in (
            outputs[:-1],
            outputs + outputs[:1],
            outputs[1:2] + outputs[1:],
            [(replace(outputs[0][0], origin=1), outputs[0][1])] + outputs[1:],
            [(replace(outputs[0][0], weight=F(1, 2)), outputs[0][1])] + outputs[1:],
        ):
            with self.assertRaises(ValueError):
                combine(faulty, selected)

    def test_context_mask_positions_and_sequence_isolation(self) -> None:
        values = tuple(map(F, range(1, 9)))
        docs = {"A": values, "B": tuple(F(100) for _ in range(8))}
        pieces = [
            (doc, positions)
            for doc in docs
            for positions in ((0, 1, 2, 3), (4, 5, 6, 7))
        ]
        result = attention_shards(docs, pieces[::-1])
        for doc, sequence in docs.items():
            self.assertEqual(
                tuple(result[doc, j] for j in range(8)), causal_prefix_means(sequence)
            )
        self.assertEqual(result["A", 4], 3)
        self.assertEqual(causal_prefix_means(values[4:])[0], 5)
        self.assertEqual(result["A", 7], F(9, 2))
        for bad in (pieces[:-1], pieces + pieces[:1]):
            with self.assertRaises(ValueError):
                attention_shards(docs, bad)
        self.assertEqual(kda_fixture()["ordered"], 1)
        self.assertEqual(kda_fixture()["reversed"], -3)

    def test_composition_preserves_shard_identity(self) -> None:
        self.assertEqual(four_rank_forward(), dict.fromkeys(range(4), (F(1), F(11))))
        for group in ((0, 1), (2, 3)):
            check_axis_group(group, "TP")
            with self.assertRaises(ValueError):
                check_axis_group(group, "DP")
        for group in ((0, 2), (1, 3)):
            check_axis_group(group, "DP")
        for group in ((1, 0), (0, 0), (0, 3)):
            with self.assertRaises(ValueError):
                check_axis_group(group, "TP")
        result = four_rank_step()
        self.assertEqual(
            result, {0: (F(97, 10),), 2: (F(97, 10),), 1: (F(39, 2),), 3: (F(39, 2),)}
        )


class CommunicationTests(unittest.TestCase):
    def test_four_rank_values_rooted_results_and_exact_composition(self) -> None:
        shards = {r: X[2 * r : 2 * r + 2] for r in GROUP}
        self.assertEqual(all_gather(shards, GROUP), dict.fromkeys(GROUP, X))
        self.assertEqual(gather(shards, 2, GROUP), {2: X})
        self.assertEqual(scatter(shards, 2, GROUP), shards)
        self.assertEqual(broadcast(X, 2, GROUP), dict.fromkeys(GROUP, X))
        partials = four_rank_partials()
        expected = tuple(map(F, (1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888)))
        self.assertEqual(all_reduce(partials, GROUP), dict.fromkeys(GROUP, expected))
        pieces = reduce_scatter(partials, GROUP)
        self.assertEqual(pieces, {r: expected[2 * r : 2 * r + 2] for r in GROUP})
        self.assertEqual(all_gather(pieces, GROUP), all_reduce(partials, GROUP))
        self.assertEqual(len(all_gather(partials, GROUP)[0]), 32)
        averaged = mean_gradient(partials, dict.fromkeys(GROUP, 1), GROUP)
        self.assertEqual(averaged, tuple(value / 4 for value in expected))
        self.assertNotEqual(averaged, expected)  # Missing normalization.
        self.assertNotEqual(averaged, tuple(value / 16 for value in expected))
        with self.assertRaises(ValueError):
            all_reduce({r: partials[r] for r in (0, 1, 2)}, GROUP)
        with self.assertRaises(ValueError):
            gather(shards, 9, GROUP)

    def test_coordinate_identity_with_equal_payloads_and_nonworld_group(self) -> None:
        group = (2, 5, 8, 11)
        shards = {
            r: tuple((2 * j + k, F(7)) for k in (0, 1)) for j, r in enumerate(group)
        }
        self.assertEqual(tagged_gather(shards, group, range(8)), (7,) * 8)
        for bad in (
            {**shards, 2: ((0, F(7)), (0, F(7)))},
            {**shards, 2: ((0, F(7)),)},
            {**shards, 2: shards[5], 5: shards[2]},
        ):
            with self.assertRaises(ValueError):
                tagged_gather(bad, group, range(8))
        with self.assertRaises(ValueError):
            tagged_gather(shards, group[::-1], range(8))

    def test_row_partition_keeps_nonlinearity_after_complete_sources(self) -> None:
        correct, premature = depth_sandwich()
        # Serial source sums are (10t,10t); their normalized weights are 1/2.
        self.assertEqual(correct, tuple(10 * t for t in X))
        self.assertEqual(premature, tuple(12 * t for t in X))
        self.assertNotEqual(correct, premature)
        self.assertEqual(correct[4:6], (50, 60))  # Rank 2's token rows.

    def test_allgather_pullback_against_independent_quadratic_loss(self) -> None:
        # L=sum_j ((j+1)/2)*sum_i x_i^2: every consumer sees every coordinate.
        gradients = {r: tuple((r + 1) * v for v in X) for r in GROUP}
        derivative = allgather_pullback(gradients, GROUP)

        def loss(values: tuple[F, ...]) -> F:
            return sum((F(r + 1, 2) * sum(v * v for v in values) for r in GROUP), F(0))

        epsilon = F(1, 100)
        for i in range(8):
            plus = tuple(v + (epsilon if j == i else 0) for j, v in enumerate(X))
            minus = tuple(v - (epsilon if j == i else 0) for j, v in enumerate(X))
            numeric = (loss(plus) - loss(minus)) / (2 * epsilon)
            self.assertEqual(derivative[i // 2][i % 2], numeric)
        averaged = allgather_pullback(gradients, GROUP, divisor=4)
        self.assertEqual(
            averaged, {r: tuple(v / 4 for v in derivative[r]) for r in GROUP}
        )
        self.assertNotEqual(averaged, derivative)

    def test_affine_prefix_identity_regrouping_and_order(self) -> None:
        fragments = affine_segments()
        self.assertEqual(exclusive_prefixes(fragments), ((0, 1, 7, 5), 10))
        for incoming in (F(0), F(2), F(-3, 2)):
            state = incoming
            direct = []
            for f in fragments:
                direct.append(state)
                state = f.apply(state)
            self.assertEqual(
                exclusive_prefixes(fragments, incoming), (tuple(direct), state)
            )
            regrouped = compose_fragments(
                (compose_fragments(fragments[:2]), compose_fragments(fragments[2:]))
            )
            self.assertEqual(regrouped.apply(incoming), state)
        self.assertEqual(compose_fragments(()).apply(F(9)), 9)
        self.assertEqual(compose_fragments(fragments[:2]).shift, 7)
        self.assertEqual(compose_fragments(fragments[:2][::-1]).shift, 9)
        with self.assertRaises(ValueError):
            exclusive_prefixes(fragments[::-1])

    def test_irregular_buckets_and_keyed_return(self) -> None:
        blocks = {s: {d: ((s, d),) for d in GROUP} for s in GROUP}
        self.assertEqual(
            all_to_all(blocks, GROUP), {d: tuple((s, d) for s in GROUP) for d in GROUP}
        )
        destinations = irregular_dispatch()
        self.assertEqual(tuple(map(len, destinations.values())), (2, 1, 1, 0))
        self.assertEqual(destinations[3], ())
        with self.assertRaises(ValueError):
            all_to_all({s: {d: () for d in GROUP} for s in GROUP[:-1]}, GROUP)
        self.assertEqual(irregular_round_trip(), {(0, "A"): 5, (1, "B"): 3})
        selected = routes()
        outputs = [(r, r.expert * r.value) for r in selected]
        self.assertEqual(combine(outputs[::-1], selected), irregular_round_trip())
        wrong = list(outputs)
        wrong[1] = (outputs[1][0], outputs[2][1])
        wrong[2] = (outputs[2][0], outputs[1][1])
        # Valid tags/counts cannot detect payloads attached to the wrong tags.
        self.assertEqual(combine(wrong, selected), {(0, "A"): 2, (1, "B"): 5})
        self.assertNotEqual(combine(wrong, selected), irregular_round_trip())
        bad_identity = [(replace(r, ordinal=0), value) for r, value in outputs]
        with self.assertRaises(ValueError):
            combine(bad_identity, selected)

    def test_completion_order_participants_counts_and_buffer_lifetimes(self) -> None:
        first = CollectiveCall("A", (0, 1), "all-gather", 2)
        second = CollectiveCall("B", (0, 1), "reduce-scatter", 4)
        valid = {0: (first, second), 1: (first, second)}
        validate_collective_order(valid, (0, 1))
        for bad in (
            {0: (first, second), 1: (second, first)},
            {0: (first, second)},
            {0: (first, second), 1: (replace(first, input_count=3), second)},
            {0: (first, second), 1: (replace(first, group=(0, 2)), second)},
            {0: (first, second), 1: (replace(first, dtype="bf16"), second)},
        ):
            with self.assertRaises(ValueError):
                validate_collective_order(bad, (0, 1))
        validate_buffer_trace(buffer_trace())
        for key, value in (("consumer_start", 2), ("reuse", 5), ("submitted", 0)):
            with self.assertRaises(ValueError):
                validate_buffer_trace({**buffer_trace(), key: value})

    def test_cost_and_result_residency_do_not_claim_peak_or_link_traffic(self) -> None:
        self.assertEqual(transfer_microseconds(4000), F(16, 5))
        self.assertEqual(transfer_microseconds(4_000_000), 203)
        reduced = reduce_scatter(four_rank_partials(), GROUP)
        replicated = all_reduce(four_rank_partials(), GROUP)
        self.assertEqual(tuple(len(v) * 2 for v in reduced.values()), (4,) * 4)
        self.assertEqual(sum(len(v) * 2 for v in replicated.values()), 64)
        self.assertEqual(sum(len(v) * 2 for v in reduced.values()), 16)


if __name__ == "__main__":
    unittest.main()
