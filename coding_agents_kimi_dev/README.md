# Chapter 25 semantic companion

This historical package retains finite checks used by the revised print
chapter. Existing D24.*, I24.*, CH24-* and J24 identifiers name its original
fixtures, not current print numbering.


The reference module implements eight dependency-free checks for the chapter's
created software-engineering fixtures: instance and patch identity, the crop
mapping bug, typed test transitions, role rewards, patch--test self-play,
pass-at-k, adaptation-curve arithmetic, and the final evidence handoff.

Run the focused tests from the `code/` directory:

    python3 -B -m unittest coding_agents_kimi_dev.test_coding_agents_kimi_dev_reference

The checks reproduce finite arithmetic and schema claims only. They do not run a
model, train a policy, execute a benchmark, estimate transfer, or reconstruct a
private Kimi system.
