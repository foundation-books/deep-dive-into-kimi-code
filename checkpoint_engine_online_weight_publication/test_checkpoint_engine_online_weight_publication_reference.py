"""Independent expected columns and adversarial publication traces."""

import unittest
from . import reference as r


class PublicationTests(unittest.TestCase):
    def test_reshard_and_counts(self):
        source = r.snapshot(8)
        shards = [
            {c: v for c, v in source.items() if r.source_owner(c) == rank}
            for rank in range(2)
        ]
        self.assertEqual([len(s) for s in shards], [14, 14])
        self.assertEqual(4 * len(source), 112)
        left = r.reconstruct(shards, 0, source)
        right = r.reconstruct(shards, 1, source)
        self.assertEqual(
            [left[("W", i, j)] for i in range(4) for j in (0, 1)],
            [800, 801, 804, 805, 808, 809, 812, 813],
        )
        self.assertEqual(
            [right[("W", i, j)] for i in range(4) for j in (2, 3)],
            [802, 803, 806, 807, 810, 811, 814, 815],
        )
        self.assertEqual((len(left) + len(right)) * 4, 128)

    def test_missing_conflicting_and_mixed(self):
        source = r.snapshot(8)
        with self.assertRaises(ValueError):
            r.reconstruct([{}], 0, source)
        with self.assertRaises(ValueError):
            r.reconstruct([source, {("W", 0, 0): 700}], 0, source)
        mixed = dict(source)
        mixed[("W", 0, 0)] = 700
        with self.assertRaises(ValueError):
            r.reconstruct([mixed], 0, source)

    def test_pipeline_and_lifetime(self):
        for m in range(1, 6):
            for a in range(1, 5):
                for b in range(1, 5):
                    self.assertEqual(
                        r.pipeline_time(m, a, b), a + b + (m - 1) * max(a, b)
                    )
        self.assertEqual(r.pipeline_time(4, 2, 3), 14)
        self.assertEqual(
            r.peak_bytes(
                {"old": (128, 0, 10), "new": (128, 3, 20), "scratch": (32, 3, 8)}
            ),
            288,
        )
        self.assertEqual(
            r.peak_bytes({"model": (128, 0, 20), "scratch": (32, 3, 8)}), 160
        )

    def test_request_survives_activation_and_lost_ack(self):
        replica = r.Replica()
        old = replica.acquire()
        token = (3, 8, 1)
        replica.begin(*token)
        with self.assertRaises(ValueError):
            replica.activate(token)
        replica.chunk(token, r.snapshot(8))
        replica.chunk(token, r.snapshot(8))
        with self.assertRaises(ValueError):
            replica.install(token, False)
        replica.install(token, True)
        self.assertEqual(replica.activate(token), 8)
        self.assertEqual(replica.activate(token), 8)
        new = replica.acquire()
        self.assertEqual(replica.read(old, ("W", 0, 0)), 700)
        self.assertEqual(replica.read(new, ("W", 0, 0)), 800)
        with self.assertRaises(ValueError):
            replica.retire(7, True)
        replica.release(old)
        with self.assertRaises(ValueError):
            replica.retire(7, False)
        replica.retire(7, True)

    def test_restart_fences_and_rebuilds(self):
        replica = r.Replica()
        token = (3, 8, 1)
        replica.begin(*token)
        replica.chunk(token, {("W", 0, 0): 800})
        replica.restart(4)
        with self.assertRaises(ValueError):
            replica.acquire()
        with self.assertRaises(ValueError):
            replica.chunk(token, r.snapshot(8))
        token = (4, 8, 1)
        replica.begin(*token)
        replica.chunk(token, r.snapshot(8))
        replica.install(token, True)
        replica.activate(token)
        self.assertEqual(replica.read(replica.acquire(), ("b", 3, 0)), 823)

    def test_crash_after_activation_requires_rebuild(self):
        replica = r.Replica()
        token = (3, 8, 1)
        replica.begin(*token)
        replica.chunk(token, r.snapshot(8))
        replica.install(token, True)
        replica.activate(token)
        replica.restart(4)
        self.assertEqual(replica.decision, token)
        with self.assertRaises(ValueError):
            replica.activate(token)
        with self.assertRaises(ValueError):
            replica.acquire()
        target = (4, replica.decision[1], 1)
        replica.begin(*target)
        replica.chunk(target, r.snapshot(8))
        replica.install(target, True)
        replica.activate(target)
        self.assertEqual(replica.read(replica.acquire(), ("E1", 1, 1)), 837)

    def test_chunk_validation_and_snapshot_isolation(self):
        replica = r.Replica()
        token = (3, 8, 1)
        replica.begin(*token)
        payload = r.snapshot(8)
        replica.chunk(token, payload)
        payload[("W", 0, 0)] = -1
        with self.assertRaises(ValueError):
            replica.chunk(token, payload)
        replica.install(token, True)
        replica.activate(token)
        self.assertEqual(replica.read(replica.acquire(), ("W", 0, 0)), 800)

    def test_complete_two_replica_packet_trace(self):
        source = r.snapshot(8)
        packets = []
        for worker in range(2):
            cols = range(2 * worker, 2 * worker + 2)
            groups = [
                [("W", i, j) for i in range(2) for j in cols],
                [("W", i, j) for i in range(2, 4) for j in cols],
                [("b", i, 0) for i in range(2)],
                [("b", i, 0) for i in range(2, 4)],
                [(f"E{worker}", i, j) for i in range(2) for j in range(2)],
            ]
            chunks = [{c: source[c] for c in group} for group in groups]
            self.assertEqual([4 * len(c) for c in chunks], [16, 16, 8, 8, 16])
            self.assertEqual(
                [{r.source_owner(c) for c in chunk} for chunk in chunks],
                [{0}, {1}, {0}, {1}, {worker}],
            )
            coordinates = [c for group in groups for c in group]
            self.assertEqual(len(coordinates), len(set(coordinates)))
            self.assertEqual(
                set(coordinates), {c for c in source if r.destination_owns(worker, c)}
            )
            # A single receive slot is held until installation completes.
            complete = 0
            completions = []
            for chunk in chunks:
                complete += 2 * (4 * len(chunk) / 16)
                completions.append(complete)
            self.assertEqual(completions, [2, 4, 5, 6, 8])
            packets.append(chunks)

        a, b = r.Replica(), r.Replica()
        old = a.acquire()  # Before time 0; released after device completion at 20.
        token = (3, 8, 1)
        a.begin(*token)
        b.begin(*token)  # Time 2.
        for chunks in packets:
            b.chunk(token, chunks[0])  # Time 4: first packet installed per worker.
        b.restart(4)  # Quiescence and teardown are external prerequisites.
        with self.assertRaises(ValueError):
            b.chunk(token, packets[0][1])
        with self.assertRaises(ValueError):
            b.acquire()
        for chunks in packets:
            for chunk in chunks:
                a.chunk(token, chunk)
        a.install(token, True)
        a.activate(token)  # Time 8.
        new = a.acquire()
        self.assertEqual(a.read(new, ("W", 3, 3)), 815)
        self.assertEqual(a.read(old, ("W", 3, 3)), 715)
        a.release(new)  # Time 9.
        retry = (4, 8, 1)
        b.begin(*retry)  # Time 10.
        for chunks in packets:
            for chunk in chunks:
                b.chunk(retry, chunk)
        b.install(retry, True)
        b.activate(retry)  # Time 18; acknowledgment lost.
        self.assertEqual(b.activate(retry), 8)  # Time 19.
        with self.assertRaises(ValueError):
            a.retire(7, True)
        a.release(old)  # Time 20, including its final device read.
        a.retire(7, True)
        self.assertEqual(
            r.peak_bytes(
                {"old": (128, 0, 20), "new": (128, 0, 21), "scratch": (32, 0, 8)}
            ),
            288,
        )
        self.assertEqual(
            r.peak_bytes(
                {
                    "old": (128, 0, 4),
                    "shadow": (128, 2, 4),
                    "scratch": (32, 2, 4),
                    "rebuilt": (128, 10, 21),
                    "retry_scratch": (32, 10, 18),
                }
            ),
            288,
        )


if __name__ == "__main__":
    unittest.main()
