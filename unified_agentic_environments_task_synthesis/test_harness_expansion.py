"""Equivalence, permissions, parser, isolation and joint-coverage checks."""

from __future__ import annotations

from .harness_expansion import PROGRAMS, World, encode_call, joint_cells


ROWS = {"Atlas": (90, 81, 10, 1), "Boreal": (90, 72, 10, 8)}


def test_equivalent_operations_and_full_task() -> None:
    observations = []
    for h in ["H0", "H1", "H2"]:
        w = World(ROWS, "repair")
        observations.append(w.call(h, encode_call(h, "read")))
        assert not w.complete()
        assert w.call(
            h, encode_call(h, "write", resource="analysis.py", body=PROGRAMS["mean"])
        ) == {"status": "ok"}
        assert w.call(h, encode_call(h, "test"))["public_pass"]
        assert w.complete()
    assert observations[0] == observations[1] == observations[2]


def test_no_hidden_repair_permissions_or_shared_mutation() -> None:
    a, b = World(ROWS, "answer"), World(ROWS, "repair")
    assert a.call("H0", encode_call("H1", "read"))["status"] == "invalid call"
    assert (
        a.call(
            "H0",
            encode_call("H0", "write", resource="analysis.py", body=PROGRAMS["mean"]),
        )["status"]
        == "invalid call"
    )
    b.call(
        "H0", encode_call("H0", "write", resource="analysis.py", body=PROGRAMS["mean"])
    )
    assert a.files["analysis.py"] == PROGRAMS["pooled"]
    assert not a.complete("Boreal")


def test_equal_marginals_do_not_fill_joint_cells() -> None:
    p, c = joint_cells(True), joint_cells(False)
    assert sum(p.values()) == sum(c.values()) == 60
    assert sum(v == 0 for v in p.values()) == 2
    assert all(v == 15 for v in c.values())
