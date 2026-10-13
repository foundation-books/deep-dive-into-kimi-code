# Chapter 28 reference implementation

The historical identifiers in these finite checks predate the current print
numbering. The print chapter contains the central worked trace; this package
lets readers execute its bounded state transitions.

This dependency-free teaching oracle makes the chapter's finite execution
contract inspectable. It includes:

- immutable runtime events and causal-parent validation;
- deterministic folding into the six-component execution state;
- complete K3-shaped assistant-record and call/result closure checks;
- versioned registry, canonical call, admission, and attempt lifecycle rules;
- a receipt-bearing crash window with explicit unknown outcome and recovered
  observation;
- finite idempotency and recovery-decision examples;
- raw-result projection with status, truncation, references, and digests;
- declared-resource conflict graphs and provider-order result projection;
- invalid-transition fixtures; and
- the complete `studio-execution-001` handoff.

Run the focused tests from the repository root:

    (cd code && python3 -B -m unittest reasoning_state_tool_execution.test_reasoning_state_tool_execution_reference)

The module is a chapter-created correctness oracle. It does not invoke Kimi K3,
Kimi Code, a network service, or an external tool; it does not establish
production compatibility, performance, or exactly-once effects. Source-level
claims remain pinned in Chapter 28 metadata.
