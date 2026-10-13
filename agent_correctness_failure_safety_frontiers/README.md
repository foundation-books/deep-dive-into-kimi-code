# Agent Correctness, Failure, Safety, and Frontiers

This historical package retains finite checks used by the revised print
chapter. Existing D31.*, I31.* and CH31-* identifiers name its original
fixtures, not current print numbering.


These finite, dependency-free models accompany Chapter 32. They make ten
mathematical checks executable: layered correctness, evidence verdicts, pure
exchange replay, atomic authorization, paired confidentiality, sufficient
compaction, effect reconciliation, worker interleavings, probability
arithmetic and bounded-claim admission.

Run from the companion root:

~~~sh
python3 -B -m unittest agent_correctness_failure_safety_frontiers.test_agent_correctness_failure_safety_frontiers_reference
~~~

The exchange protocol deliberately rejects duplicate terminal delivery.
The service models atomic serialization against revocation; it is not a
concurrent implementation. Idempotency covers the job mapping for the same
key and payload, while physical attempts can increase. Different keys
remain distinct operations. The separate swarm model supplies the global
one-allocation reservation rule.

The grant fixture assumes authentic issuance and lineage. Its child-grant
check verifies explicit permission for a different principal and limits
further delegation; it is not a cryptographic capability implementation.

The confidentiality domain has two deterministic terminating runs and one
public output. Claim admission checks a small evidence contract; it does not
automatically prove arbitrary properties. All numbers are mathematical
fixtures. No model, benchmark, network service or empirical defense campaign
is executed.
