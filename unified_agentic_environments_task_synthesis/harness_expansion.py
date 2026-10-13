"""Strict equivalent interfaces for the bounded answer/repair comparison.

Only the three displayed, repository-authored program candidates are offered;
the checker recognizes the exact general mean program without executing Python.
"""

from __future__ import annotations

import json
from typing import Any

from .task_oracle import independent_scores


PROGRAMS = {
    "mean": "def score(rows):\n    return sum(c/n for _,n,c in rows)/len(rows)\n",
    "pooled": "def score(rows):\n    return sum(c for _,n,c in rows)/sum(n for _,n,c in rows)\n",
    "constant": "def score(rows):\n    return 0.8\n",
}


def encode_call(
    harness: str, operation: str, *, resource: str = "counts.csv", body: str = ""
) -> str:
    """Render a canonical operation in one exact interface schema."""
    if harness == "H0":
        name = {"read": "read_file", "write": "write_file", "test": "run_tests"}[
            operation
        ]
        args = {} if operation == "test" else {"path": resource}
        if operation == "write":
            args["text"] = body
    elif harness == "H1":
        name = {"read": "files.read", "write": "files.write", "test": "tests.run"}[
            operation
        ]
        args = {} if operation == "test" else {"path": resource}
        if operation == "write":
            args["text"] = body
    elif harness == "H2":
        name = "check" if operation == "test" else "document"
        args = (
            {"suite": "public"}
            if operation == "test"
            else {"operation": operation, "resource": resource}
        )
        if operation == "write":
            args["body"] = body
    else:
        raise ValueError("unknown interface")
    return json.dumps({"name": name, "arguments": args}, sort_keys=True)


def decode_call(harness: str, generated: str) -> tuple[str, str, str]:
    """Exact name/key validation: never repair another interface's output."""
    item = json.loads(generated)
    if set(item) != {"name", "arguments"} or not isinstance(item["arguments"], dict):
        raise ValueError("malformed call")
    args = item["arguments"]
    for operation in ["read", "write", "test"]:
        resource = str(
            args.get("resource" if harness == "H2" else "path", "counts.csv")
        )
        body = str(args.get("body" if harness == "H2" else "text", ""))
        if encode_call(harness, operation, resource=resource, body=body) == json.dumps(
            item, sort_keys=True
        ):
            return operation, resource, body
    raise ValueError("unavailable spelling or argument structure")


class World:
    """One fresh world with family-specific permissions and independent truth."""

    def __init__(self, rows: dict[str, tuple[int, int, int, int]], family: str) -> None:
        if family not in ("answer", "repair"):
            raise ValueError("unknown task family")
        self.rows = dict(rows)
        self.family = family
        self.files = {"counts.csv": self._csv(), "analysis.py": PROGRAMS["pooled"]}
        self.reads = 0
        self.invalid = 0
        self.calls = 0

    def _csv(self) -> str:
        text = "classifier,class,support,correct\n"
        for name, (n0, c0, n1, c1) in self.rows.items():
            text += f"{name},0,{n0},{c0}\n{name},1,{n1},{c1}\n"
        return text

    def call(self, harness: str, generated: str) -> dict[str, Any]:
        self.calls += 1
        try:
            operation, resource, body = decode_call(harness, generated)
            if operation == "read" and resource in self.files:
                self.reads += 1
                return {"status": "ok", "text": self.files[resource]}
            if (
                operation == "write"
                and self.family == "repair"
                and resource == "analysis.py"
            ):
                self.files[resource] = body
                return {"status": "ok"}
            if operation == "test" and self.family == "repair":
                return {
                    "status": "ok",
                    "public_pass": self.files["analysis.py"] == PROGRAMS["mean"],
                }
        except (ValueError, KeyError, TypeError):
            pass
        self.invalid += 1
        return {"status": "invalid call"}

    def complete(self, answer: str | None = None) -> bool:
        if self.invalid or self.reads == 0:
            return False
        if self.family == "answer":
            truth = max(
                self.rows,
                key=lambda k: independent_scores(
                    [
                        ("0", self.rows[k][0], self.rows[k][1]),
                        ("1", self.rows[k][2], self.rows[k][3]),
                    ]
                )[1],
            )
            return answer == truth
        # This study offers only declared program strings. The exact general
        # mean candidate is accepted; pooled and constant alternatives fail.
        return self.files["analysis.py"] == PROGRAMS["mean"]


def joint_cells(paired_only: bool) -> dict[str, int]:
    """Count the fixed paired-only or crossed sixty-episode worksheet."""
    return {
        "answer/H0": 30 if paired_only else 15,
        "answer/H1": 0 if paired_only else 15,
        "repair/H0": 0 if paired_only else 15,
        "repair/H1": 30 if paired_only else 15,
    }
