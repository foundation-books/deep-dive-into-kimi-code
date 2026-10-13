# Chapter 27 reference implementation

This dependency-free module makes the chapter's small mathematical fixtures
executable. It includes:

- a deliberately limited schema-profile checker and instance validator;
- diagnostic-preserving, key-sorted schema projection;
- a finite byte-language recognizer and legal-token renormalization;
- explicit accepted, dead-prefix, premature-end, and interrupted outcomes;
- a typed XTML-oriented event tree, zone checker, and scope evaluator;
- opaque call-ID normalization and exact result matching, including open and
  ambiguous runs; and
- the complete `studio-contract-001` validity trace and ten corruptions.

The running fixture is centralized as `STUDIO_RECORD` in `fixture.py`; its
schema, effective options, calls, out-of-order arrivals, response, and evidence
ledger drive the product-state check rather than a table of expected labels.

Run its tests from the repository root:

    make -C code chapter-tests

The module is a teaching oracle, not Moonshot AI's Walle implementation, Kimi
tokenizer, model, server, or runtime. Walle validates schema definitions; the
small instance validator here exists only to make Chapter 27's denotational
examples testable. Source-level claims remain pinned in the chapter metadata.
