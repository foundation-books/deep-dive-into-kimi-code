# Chapter 25 semantic companion

The original dependency-free reference checks remain in reference.py.
The visual-loop investigations add deterministic image fixtures and twelve
bounded cases (T01--T12): geometry, image delivery and ordering, source-image
choice, assistant/observation ownership, budget termination, diagnostic replay,
freshness, and leakage. They do not run K3, veRL, a GUI VM, or model training.

Run from the code/ directory:

    python3 -B _maintenance/chapter_test_runner.py \
      visual_multimodal_agent_learning/test_visual_multimodal_agent_learning_reference.py
    python3 -B _maintenance/chapter_test_runner.py \
      visual_multimodal_agent_learning/test_visual_loop_investigations.py

The second suite uses only the Python standard library. Its optional pinned
processor test is skipped unless KIMI_K3_PROCESSOR_FILE points to a local
copy of the exact c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721
kimi_k3_processor.py. It verifies the registered SHA-256 before exercising
that source's processor call and extractor with stub tokenizer and vision
processor. This establishes path-only, automatic user/tool extraction, and
explicit-media routing through the source method. Stub pixel/grid fields are
not real K3 tensors. Full pixel_values and grid_thws correspondence needs
the configured K3 tokenizer and vision processor and remains unverified here.
Do not load K3 model weights for this input-interface question.

The optional veRL source check uses the registered KIMI_VERL_CHECKOUT at
ea53291385ce764019a2b40733605f21d8317583. It checks that the tool loop
emits the response mask, the agent-loop manager places it in the batch, and
the selected trainer consumes it for advantage estimation before passing the
batch to the actor update. This is a pinned source-path check, not a trainer
run or a gradient test for K3's visual components.

crop_source_checks.py deterministically generates the six PNGs in fixtures/.
The studio source and crop match the printed 160-by-160 root, 60-by-120 crop,
and 300-by-600 viewport. The separate chart detail fixture compares a
retained source crop with an enlarged fourfold preview. Both views derive
from one image; their pixel difference does not establish any model-accuracy
difference. The checked source revisions and claim limits are listed in
source_manifest.json; the book's citations state the source basis.

No model-level learning effect is measured by these finite fixtures.

## October 2026 finite inspection task

`inspection_task.py` defines protected numerical-panel truth, source crop
rectangles, disjoint label families and merged-grid accounting without image or
model dependencies. Its tests are included by the registered chapter test.
They distinguish legal useless crops from correct answers and keep protected
values out of the allowed execution text. Actual rendered-pixel masking and
media interventions are reviewed separately in the complete experiment archive.

The book's optional Qwen2.5-VL-3B study updates only a six-action inspection
readout beside a frozen visual model. Its final full policy is worse than cold;
the common-answerer projection and local pixel-sensitivity comparison have
separate meanings. The ordinary companion never downloads weights or launches
that study. The actual episode and worked gradient are printed in Chapter 25.
