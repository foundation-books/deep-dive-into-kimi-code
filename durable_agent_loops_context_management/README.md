# Historical Chapter 29 reference implementation

The historical identifiers in these finite checks predate the current print
numbering. The print chapter contains the central worked trace; this package
lets readers execute the bounded state and accounting examples.

This dependency-free teaching oracle makes the chapter's finite durable-loop
contracts inspectable. It includes:

- plan-frontier and admissible-action intersections;
- evidence-linked progress and monotone budget accounting;
- mandatory context selection under capacity;
- semantic-retention checks for compaction;
- provenance closure and checkpoint completeness;
- ordered resume admission, version-drift classification, and open-effect
  reconciliation;
- checkpoint-count, explicit write-cost, and lost-work accounting;
- explicit drift classifications and dispositions; and
- terminal-certificate and oscillation checks.

Run the focused tests from the companion-code root (`code/` in the book source):

    python3 -B _maintenance/chapter_test_runner.py \
      durable_agent_loops_context_management/test_durable_agent_loops_context_management_reference.py

The module is a chapter-created finite correctness oracle. It invokes no model,
external tool, or network service and establishes no production compatibility,
policy optimality, performance, or empirical compaction quality. Those
measurements remain pending under the registered Chapter 29 protocols.
