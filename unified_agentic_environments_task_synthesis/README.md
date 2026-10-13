# Chapter 24 semantic companion

The print chapter contains the task and handoff explanation. The original
`D22.*` and `I22.*` identifiers used by these checks predate its current
theorem counters.

`reference.py` implements eight dependency-free checks for the chapter's
pedagogical fixtures: trace probability, harness pairing, joint
task-by-harness coverage, concept-DAG admission, synthesis gates, split
contamination, persistent autonomous execution, and the final evidence
handoff. `studio_world.py` is the shared, book-authored stateful task fixture:
it starts with an incorrect report, exposes one learner-visible observation at
each action, retains changes during an episode, resets on a fresh attempt, and
checks report state, grounded image evidence, closure, and submission. The
flat and namespaced configurations change the inspection spelling while using
the same six canonical actions and terminal evaluator.

The current studio inspection is one batch action:
`inspect_boxes(ids=["red", "blue"])` and
`assets.inspect_bboxes({"shape_ids": ["red", "blue"]})` both adapt to
`InspectBoxes(("red", "blue"))`. The returned boxes use
`(x, y, width, height)`, with image `y` increasing downward. The focused tests
check both adapters, missing-argument rejection, unchanged source geometry, the
six-action inventory, and the three-task split chain. Additional tests exercise
two successful action orders, state retention and reset, missing capabilities,
lost observation context, fabricated evidence, words-only completion, and
event visibility. The fixture does not certify a trained policy's execution.

The studio task does not copy AppWorld task data or implement private K3
machinery. Its finite state and outcome rules are defined in the print chapter.

Run the tests from the companion-code root (`code/`):

    python3 -B _maintenance/chapter_test_runner.py \
      unified_agentic_environments_task_synthesis/test_unified_agentic_environments_task_synthesis_reference.py

These checks establish only the stated finite fixtures. They do not reproduce
Kimi K3's private environment generator, run a model, or measure agent quality.
The separate author-side Chapter 24 Search-R1 component adds pinned response
processing, returned-observation insertion, and actor-mask execution on
scripted searches. Its records live under
`experiments/unified_agentic_environments_task_synthesis/` in the complete book
source and are outside this companion export; the volume states their scope.

## Graph-guided analysis packet

`task_factory.py` uses a declared ancestor-qualified local query and source
roles to build three independent teaching tasks from the fictional
Atlas/Boreal count packet in `fixtures/`. `sources.json` identifies the pinned
public definitions and the exact local fixture bytes. Each task receives a
fresh folder. The reference routes use the learner-visible definitions, note,
and counts; `task_oracle.py` computes expected values separately with exact
rational arithmetic. The candidate runner accepts only the named, repository-
defined repair variants. Its process limits make this a finite illustration,
not an execution service for arbitrary submitted programs.

From `code/`, run:

    python3 -B -m unified_agentic_environments_task_synthesis.task_factory_cli \
      construct --family question
    python3 -B -m unified_agentic_environments_task_synthesis.task_factory_cli \
      reference --family repair
    python3 -B -m unified_agentic_environments_task_synthesis.task_factory_cli \
      candidate --candidate hardcoded
    python3 -B -m unified_agentic_environments_task_synthesis.task_factory_cli \
      numbers

Use `question`, `repair`, or `report` for construction and reference checks.
Named repair candidates are `reference`, `alternative`, `buggy`, `hardcoded`,
`swapped`, and `tamper-counts`. The focused test command above also checks
source identity, information boundaries, fresh resets, protected packet
splits, and unsupported-evidence diagnostics without network or model access.

## Persistent workflows and autonomous investigation

`onboarding.py` is a separate, book-created four-application world with
ordered simulated events, two successful response routes, delayed
provisioning, event-local deadlines, protected collateral state, and
independent author checks. `repair_oracle.py` is a fixed three-operation
ticket service with bounded policy-facing oracle calls, public diagnostics,
and one final submission. `learning_fixture.py` checks the one-parameter
reconciliation update and annotates assistant actions versus tool
observations. These cases do not change `studio-ops-001` or the analysis
packet above.

From `code/`, run the focused cases:

    python3 -B -m unittest \
      unified_agentic_environments_task_synthesis.test_persistent_autonomous -q

To inspect TeX renderings of the same authored fixtures, write them to a local
output directory:

    python3 -B -m unified_agentic_environments_task_synthesis.render_book_assets \
      --output-dir build/task-fixtures

This creates two tables and two vector diagrams without manuscript files.
Repeat with `--check` to verify the generated files. The finite fixtures show
their declared mechanisms, not a trained model's completion rate or a K3-private
implementation.

## October 2026 interface and summary additions

`harness_expansion.py` supplies strict H0/H1/H2 read, write and test adapters,
fresh answer/repair worlds, family permissions and the 60-episode joint-coverage
worksheet. The repair checker recognizes only the three declared candidate
program strings; it does not execute arbitrary Python. The registered chapter
test checks operation equivalence, rejected foreign spellings, write permissions,
state isolation and empty joint cells. All normal checks remain dependency-free.

`fixtures/search_learning_summary.json` is generated from the unchanged accepted
Search-R1 result, keeping its 5/24 versus 6/24 final outcomes and limitations.
The new optional model experiments live in the complete repository's experiment
campaign. Their readout updates and negative final outcomes are separate from
adapter correctness. The frozen Ada summary probe has no update and stays
unfinished under both notes; P2 and P3 are deferred.
