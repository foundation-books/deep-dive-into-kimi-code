"""Independent fixture outcomes and adversarial boundary/event checks."""

import itertools
import unittest
from dataclasses import replace
from . import reference as r
from .recovery import Controller, effect_retry_allowed
from .accounting import AllocationEvent as Event, occupancy_trace


class InfrastructureTests(unittest.TestCase):
    def test_phase_ledger_and_transfer_overlap(self):
        # w_r=512, w_t=1024, o=2048, h=256, d=4096, all synthetic bytes.
        events = (
            Event("GPU", "rollout", 512),
            Event("GPU", "prefix", 160),
            Event("host", "fixed", 256),
            Event("host", "prefix", 160),
            Event("NVMe", "old", 2048),
            Event("durable", "G/E8", 4096),
            Event("GPU", "prefix", None),
            Event("host", "prefix", None),
            Event("GPU", "rollout", None),
            Event("GPU", "training", 1024),
            Event("GPU", "slots", 128),
            Event("host", "training", 2048),
            Event("NVMe", "new", 2048),
            Event("NVMe", "old", None),
            Event("host", "training", None),
            Event("GPU", "training", None),
            Event("GPU", "slots", None),
            Event("GPU", "rollout", 512),
            Event("GPU", "prefix", 160),
            Event("GPU", "scratch", 64),
            Event("GPU", "scratch", None),
            Event("GPU", "growth", 64),
        )
        trace = occupancy_trace(events)
        self.assertEqual(
            trace[5], {"GPU": 672, "host": 416, "NVMe": 2048, "durable": 4096}
        )
        self.assertEqual(trace[12]["NVMe"], 4096)
        peaks = {tier: max(row.get(tier, 0) for row in trace) for tier in trace[-1]}
        self.assertEqual(
            peaks, {"GPU": 1152, "host": 2304, "NVMe": 4096, "durable": 4096}
        )
        self.assertEqual(trace[-1]["GPU"], 736)
        self.assertEqual(trace[19]["GPU"], 736)
        for suffix in (Event("GPU", "prefix", 160), Event("GPU", "missing", None)):
            with self.assertRaises(ValueError):
                occupancy_trace(events + (suffix,))

    def test_group_gate_and_provenance(self):
        groups = {"H": (True,) * 4, "G": (True, True, True, False)}
        self.assertEqual(sum(map(sum, groups.values())), 7)
        self.assertEqual(r.ready_groups(groups), {"H"})
        old = r.Record()
        new = r.continue_record(old)
        self.assertEqual(new.mask, (0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1))
        self.assertEqual(new.versions, (None, None, 7, 7, None, None, 7, 7, 8, 8, 8, 8))
        self.assertEqual(new.log_probs[:8], old.log_probs)
        groups["G"] = (True,) * 4
        self.assertEqual(r.ready_groups(groups), {"G", "H"})

    def test_invalid_records(self):
        self.assertTrue(r.valid_record(r.Record()))
        for change in (
            {"environment": "E0"},
            {"in_flight": True},
            {"sampler": ""},
            {"observation_end": 8},
            {"versions": (7,) * 8},
            {"mask": (1,)},
        ):
            with self.subTest(change=change):
                self.assertFalse(r.valid_record(replace(r.Record(), **change)))
        probabilities = list(r.Record().log_probs)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            probabilities[2] = invalid
            self.assertFalse(
                r.valid_record(replace(r.Record(), log_probs=tuple(probabilities)))
            )

    def test_cache_state(self):
        self.assertEqual((r.cache_bytes(8), r.cache_bytes(12)), (160, 224))
        self.assertEqual(r.cache_bytes(8) * 2, 320)
        self.assertEqual([r.scalar_state(a, (1, 1)) for a in (1, 2)], [2, 3])
        prefix = r.Record().tokens
        cache = r.Cache(7, prefix, 8)
        self.assertTrue(r.compatible(cache, 7, prefix))
        for bad in (
            replace(cache, version=8),
            replace(cache, boundary=4),
            replace(cache, groups=frozenset({"MLA"})),
            replace(cache, verified=False),
            replace(cache, prefix=("wrong",) * 8),
        ):
            self.assertFalse(r.compatible(bad, 7, prefix))

    def test_unique_bytes_and_growth(self):
        self.assertEqual(
            r.physical_bytes((("slot0", 64), ("slot0", 64), ("slot1", 64))), 128
        )
        self.assertEqual(
            r.physical_bytes((("fixed", 100), ("source", 160), ("staging", 80))), 340
        )
        self.assertEqual(
            r.physical_bytes((("fixed", 100), ("source", 80), ("staging", 80))), 260
        )
        self.assertTrue(r.can_admit((60,), (25,), 100, 10))
        self.assertTrue(r.can_admit((60,), (15,), 100, 10))
        self.assertFalse(r.can_admit((60,), (25, 15), 100, 10))
        # Enumerate all occupancies under small admitted reservation vectors.
        for reservations in itertools.product(range(5), repeat=3):
            if r.can_admit((), reservations, 9, 1):
                for occupancy in itertools.product(
                    *(range(x + 1) for x in reservations)
                ):
                    self.assertLessEqual(sum(occupancy) + 1, 9)

    def test_slot_hazards(self):
        ordinary = r.slot_schedule()
        self.assertTrue(r.safe_accesses(ordinary))
        self.assertEqual(ordinary[-1], r.Access(0, 8, 11, False))
        late = ordinary + (r.Access(0, 5, 7, False),)
        self.assertFalse(r.safe_accesses(late))
        repaired = r.slot_schedule(7)
        self.assertTrue(r.safe_accesses(repaired))
        self.assertEqual(
            repaired[-2:], (r.Access(0, 7, 9, True), r.Access(0, 9, 12, False))
        )
        # Independent tick oracle for pairwise conflict semantics.
        for a, b in itertools.product(range(4), repeat=2):
            accesses = (r.Access(0, a, a + 2, True), r.Access(0, b, b + 1, False))
            overlap = bool(set(range(a, a + 2)) & {b})
            self.assertEqual(r.safe_accesses(accesses), not overlap)

    def test_snapshot_ancestry_and_fork(self):
        base = {i: "base" for i in range(10)}
        d1 = r.Delta("D1", "base", ((1, "first"), (3, "old"), (7, "first")))
        d2 = r.Delta("D2", "D1", ((3, "new"), (9, "new")))
        restored = r.restore(base, (d1, d2))
        self.assertEqual(
            [restored[i] for i in (1, 3, 7, 9)], ["first", "new", "first", "new"]
        )
        self.assertEqual(base[3], "base")
        self.assertEqual([len(d.pages) * 4 for d in (d1, d2)], [12, 8])
        for chain in ((d2,), (d2, d1), (d1, d1)):
            with self.assertRaises(ValueError):
                r.restore(base, chain)
        fork = restored.copy()
        for page in (0, 2, 4):
            fork[page] = "private"
        self.assertEqual(sum(fork[i] != restored[i] for i in fork) * 4, 12)

    def test_crash_at_each_transition(self):
        events = ("prepare", "validate", "commit", "admit")
        for crash_after in range(5):
            controller = Controller()
            failed = controller.takeover()
            for event in events[:crash_after]:
                self.assertTrue(self._step(controller, failed, event))
            frontier = controller.record.tokens
            current = controller.takeover()
            self.assertFalse(controller.may_execute(failed))
            self.assertEqual(controller.reserved, 0)
            for event in events:
                self.assertFalse(self._step(controller, failed, event))
                self.assertTrue(self._step(controller, current, event))
                self.assertTrue(self._step(controller, current, event))
            self.assertTrue(controller.may_execute(current))
            self.assertEqual(controller.reserved, 224)
            self.assertEqual(controller.record.tokens, frontier)

    @staticmethod
    def _step(controller, generation, event):
        if event == "admit":
            cache = r.Cache(8, controller.record.tokens, 8)
            return controller.admit(generation, cache, 234, margin=10)
        return controller.transition(generation, event)

    def test_admission_enforces_all_gates(self):
        controller = Controller()
        generation = controller.takeover()
        cache = r.Cache(8, controller.record.tokens, 8)
        self.assertFalse(controller.admit(generation, cache, 1000))
        for event in ("prepare", "validate", "commit"):
            self.assertTrue(controller.transition(generation, event))
        self.assertFalse(controller.transition(generation, "admit"))
        self.assertFalse(controller.admit(generation, replace(cache, version=7), 1000))
        incomplete = replace(cache, groups=frozenset({"MLA"}))
        self.assertFalse(controller.admit(generation, incomplete, 1000))
        self.assertFalse(controller.admit(generation, cache, 233, margin=10))
        self.assertFalse(controller.admit(generation, cache, 234, (1,), margin=10))
        self.assertEqual((controller.stage, controller.reserved), ("committed", 0))
        self.assertFalse(controller.may_execute(generation))
        self.assertTrue(controller.admit(generation, cache, 244, (10,), margin=10))
        self.assertTrue(controller.admit(generation, cache, 244, (10,), margin=10))
        self.assertEqual(controller.reserved, 224)
        controller.adopted_version = 9
        self.assertFalse(controller.may_execute(generation))
        self.assertEqual(controller.reserved, 224)
        controller.takeover()
        self.assertFalse(controller.admit(generation, cache, 1000))
        self.assertFalse(controller.may_execute(generation))
        self.assertEqual(controller.reserved, 0)

    def test_invalid_transitions_and_external_effect(self):
        controller = Controller()
        generation = controller.takeover()
        self.assertFalse(controller.transition(generation, "commit"))
        self.assertFalse(controller.transition(generation, "admit"))
        bad = replace(r.Record(), generation=generation, environment="E0")
        self.assertTrue(controller.transition(generation, "prepare", bad))
        self.assertFalse(controller.transition(generation, "validate"))
        self.assertFalse(controller.transition(generation, "commit"))
        self.assertEqual(controller.record.environment, "E8")
        generation = controller.takeover()
        advanced = replace(r.continue_record(controller.record), generation=generation)
        self.assertTrue(controller.transition(generation, "prepare", advanced))
        self.assertFalse(controller.transition(generation, "validate"))
        self.assertEqual(len(controller.record.tokens), 8)
        self.assertFalse(effect_retry_allowed("unknown", False))
        self.assertTrue(effect_retry_allowed("unknown", True))
        self.assertTrue(effect_retry_allowed("confirmed_absent", False))
        self.assertFalse(effect_retry_allowed("confirmed_done", False))


if __name__ == "__main__":
    unittest.main()
