"""Reproduce Chapter 24's two tables and two vector diagrams from checked cases.

Run from ``code/`` with an explicit output directory. ``--check`` is read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .onboarding import DAY, MANAGER_AVAILABLE, scripted_attempt
from .repair_oracle import TicketService, investigation_record


def clock_time(minutes: int) -> str:
    """Format an integer simulated minute as day and clock time."""

    day, minute = divmod(minutes, DAY)
    hour, minute = divmod(minute, 60)
    return f"D{day + 1} {hour:02d}:{minute:02d}"


def _object_label(objects: list[dict[str, Any]]) -> str:
    if not objects:
        return "none"
    if len(objects) != 1:
        return f"{len(objects)} objects"
    obj = objects[0]
    return f"D{obj['day']}, {obj['manager']}"


def obligation_rows() -> str:
    """Render the checked onboarding snapshots as TeX table rows."""

    world = scripted_attempt()
    obligations = {
        "initial commitments": "pending request retained",
        "date observed": "revise date; request still D4",
        "date reconciled": "await old request",
        "manager reconciled": "await old request",
        "old result available": "repair D4 grant",
        "grant reconciled": "one D6 grant",
    }
    rows = []
    for item in world.milestones:
        request = item["requests"][0]
        grant = item["grants"]
        effect = f"request D{request['day']} ({request['status']})"
        if grant:
            effect += f"; grant D{grant[0]['day']}"
        rows.append(
            f"{clock_time(int(item['time']))} ({item['label']}) & "
            f"{_object_label(item['calendar'])} / {_object_label(item['tasks'])} & "
            f"{effect} & {obligations[str(item['label'])]} \\\\"
        )
    return "\n".join(rows) + "\n\\bottomrule\n"


def investigation_rows() -> str:
    """Render candidate revisions and cumulative oracle calls."""

    record = investigation_record()
    trace = record["trace"]
    groups = (
        ("normal-only", trace[2], "create, assign, close: success"),
        ("duplicate revised", trace[4], "second create: existing T2"),
        ("closure revised", trace[6], "unassigned close: assignment required"),
    )
    return (
        "\n".join(
            f"{candidate} & {evidence} & {item['cumulative']}/8 \\\\"
            for candidate, item, evidence in groups
        )
        + "\n\\bottomrule\n"
    )


def _probe_predictions() -> tuple[str, str, str, str]:
    always_new = TicketService(reuse_identical=False)
    always_new.create("B", "camera")
    duplicate_a = str(always_new.create("B", "camera")["status"])
    reused = TicketService()
    reused.create("B", "camera")
    duplicate_b = str(reused.create("B", "camera")["status"])
    permissive = TicketService(require_assignment=False)
    close_a = str(
        permissive.close(str(permissive.create("C", "lens")["ticket"]))["status"]
    )
    guarded = TicketService()
    close_b = str(guarded.close(str(guarded.create("C", "lens")["ticket"]))["status"])
    return duplicate_a, duplicate_b, close_a, close_b


def query_figure() -> str:
    """Draw predictions and observed branches from the checked oracle."""

    duplicate_a, duplicate_b, close_a, close_b = _probe_predictions()
    actual = investigation_record()
    if actual["trace"][4]["output"]["status"] != duplicate_b:
        raise AssertionError("duplicate figure differs from checked oracle")
    if actual["unassigned_status"] != close_b:
        raise AssertionError("closure figure differs from checked oracle")
    close_b = close_b.replace("_", r"\_")
    return rf"""\begin{{tikzpicture}}[x=1cm,y=1cm,>=Latex,
  box/.style={{draw,rounded corners,align=center,inner sep=4pt,font=\normalsize}},
  answer/.style={{box,fill=blue!9}}, probe/.style={{box,fill=orange!14}}]
\node[box] (ordinary) at (0,3.4) {{Normal path: both fit}};
\node[probe] (dup) at (0,2.3) {{Repeat same reference and device}};
\node[box] (da) at (-3,1.1) {{Always new\\\texttt{{{duplicate_a}}}}};
\node[answer] (db) at (3,1.1) {{Reuse identical\\\texttt{{{duplicate_b}}} observed}};
\draw[->] (ordinary) -- (dup);
\draw[->] (dup) -- (da); \draw[->,very thick] (dup) -- (db);
\node[box] (normal) at (0,-.4) {{Close rule still uncertain}};
\node[probe] (close) at (0,-1.5) {{Close without assignment}};
\node[box] (ca) at (-3,-2.7) {{Permissive\\\texttt{{{close_a}}}}};
\node[answer] (cb) at (3,-2.7) {{Guarded\\\texttt{{{close_b}}} observed}};
\draw[->] (normal) -- (close);
\draw[->] (close) -- (ca); \draw[->,very thick] (close) -- (cb);
\end{{tikzpicture}}
"""


def timeline_figure() -> str:
    """Draw the four application lanes from a successful world trace."""

    world = scripted_attempt()
    initial = world.milestones[0]
    final = world.milestones[-1]
    before_grant_repair = world.milestones[-2]
    first_day = initial["calendar"][0]["day"]
    current_day = final["calendar"][0]["day"]
    current_manager = final["calendar"][0]["manager"]
    old_grant_day = before_grant_repair["grants"][0]["day"]
    event_times = {
        row["event"]: clock_time(row["time"]) for row in world.history if "event" in row
    }
    # The three decision labels are derived from the actual checked action log.
    date_action = next(
        row
        for row in world.history
        if row.get("action") == "revise_calendar" and row["time"] < MANAGER_AVAILABLE
    )
    manager_action = next(
        row
        for row in world.history
        if row.get("action") == "revise_calendar" and row["time"] > MANAGER_AVAILABLE
    )
    grant_action = next(
        row for row in world.history if row.get("action") == "reconcile_grant"
    )
    date_index = date_action["index"]
    manager_index = manager_action["index"]
    grant_index = grant_action["index"]
    date_time = clock_time(date_action["time"])
    manager_time = clock_time(manager_action["time"])
    grant_time = clock_time(grant_action["time"])
    return rf"""\begin{{tikzpicture}}[x=1cm,y=.9cm,>=Latex,
  event/.style={{draw,rounded corners,fill=blue!9,align=center,inner sep=3pt,font=\normalsize}},
  state/.style={{draw,rounded corners,fill=white,align=center,inner sep=3pt,font=\normalsize}},
  action/.style={{draw,rounded corners,fill=orange!16,align=center,inner sep=3pt,font=\normalsize}}]
\foreach \yy/\name in {{5/External,4/Messages,3/Calendar,2/Tasks,1/Access,0/Policy}} {{
  \node[anchor=east,font=\normalsize] at (-.25,\yy) {{\name}};
  \draw[gray!55] (0,\yy) -- (13.2,\yy);
}}
\node[state] at (1.2,3) {{D{first_day}, Mira}};
\node[state] at (1.2,2) {{D{first_day} request\\pending}};
\node[event] (date) at (3.7,5) {{{event_times["date_occurs"]}\\date D{current_day}}};
\node[state] (date-vis) at (3.7,4) {{{event_times["date_available"].split()[1]} visible}};
\node[state] at (3.7,3) {{D{current_day}, Mira}};
\node[state] at (3.7,2) {{record D{current_day}\\request D{first_day}}};
\node[action] at (3.7,0) {{a{date_index} {date_time.split()[1]}\\revise date}};
\node[event] (mgr) at (6.7,5) {{{event_times["manager_occurs"]}\\{current_manager}}};
\node[state] (mgr-vis) at (6.7,4) {{{event_times["manager_available"].split()[1]} visible}};
\node[state] at (6.7,3) {{D{current_day}, {current_manager}}};
\node[state] at (6.7,2) {{{current_manager}\\request D{first_day}}};
\node[action] at (6.7,0) {{a{manager_index} {manager_time.split()[1]}\\revise manager}};
\node[event] (complete) at (9.6,2) {{{event_times["request_completes"].split()[1]} request\\done}};
\node[state] (result) at (12.1,2) {{{event_times["grant_available"].split()[1]} result\\seen}};
\node[state] at (9.8,1) {{D{old_grant_day} grant}};
\node[state] at (12.1,1) {{D{current_day} grant}};
\node[action] at (11,0) {{a{grant_index} {grant_time.split()[1]}\\reconcile grant}};
\draw[->,dashed] (date) -- (date-vis);
\draw[->,dashed] (mgr) -- (mgr-vis);
\draw[->,dashed] (complete) -- (result);
\end{{tikzpicture}}
"""


def assets() -> dict[str, str]:
    """Return every deterministic table and vector figure source."""

    return {
        "onboarding_obligations_rows.tex": obligation_rows(),
        "repair_investigation_rows.tex": investigation_rows(),
        "onboarding_timeline.tex": timeline_figure(),
        "repair_query_tree.tex": query_figure(),
    }


def main() -> None:
    """Write or verify generated chapter assets in an explicit directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in assets().items():
        destination = args.output_dir / name
        if args.check:
            if not destination.is_file() or destination.read_text() != content:
                raise SystemExit(f"out of date: {destination}")
        else:
            destination.write_text(content)
        print(destination)


if __name__ == "__main__":
    main()
