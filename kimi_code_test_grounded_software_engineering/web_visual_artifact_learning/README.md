# Controlled dashboard fixture

This is the original six-row teaching application for Chapter 29. It does not require a model service, external data, npm packages, or network access. The candidate files D0--D4 are authored snapshots; they are not K3 outputs. The reference implementation and independent oracle are separate from the candidate code. From the companion-code root (`code/` in the book source), run:

```sh
node kimi_code_test_grounded_software_engineering/web_visual_artifact_learning/runner/browser_checks.mjs dashboard-check
node kimi_code_test_grounded_software_engineering/web_visual_artifact_learning/training_example/step_coefficients.mjs
node kimi_code_test_grounded_software_engineering/web_visual_artifact_learning/probes/feedback_boundary.mjs
```

The captured evidence used Node v24.7.0 and Google Chrome 153.0.8010.53. The
runner accepts Chrome or Chromium with browser build 153.0.8010.53. It searches
`PATH` for `google-chrome`, `google-chrome-stable`, `chromium`,
`chromium-browser`, `chrome`, `chrome.exe`, or `chromium.exe`. Set
`K3_DASHBOARD_CHROME` to an executable path or command name to select a browser;
this takes priority over automatic discovery. On macOS or Windows, installations
outside `PATH` require this override. Paths containing spaces are supported.
The runner checks the browser build through the DevTools protocol across
platforms and records the reported browser product in new results.

It writes `results.json` and browser PNGs to its output directory. It creates
disposable local candidate copies, parses actual CSV downloads, checks at
960×900, 390×844 and 430×932 CSS pixels, and rebuilds the selected source for a
separate final case. The default application in `app/` is D3; opening
`app/index.html` directly shows the working candidate. The runner never
rewrites its visual reference. `source_versions.json` records the external
readings; no upstream code was copied. The archived `runnerSha256` identifies
the runner used for the original capture, before the portable discovery change.

Development selection uses 960×900 desktop and 390×844 narrow captures. The
430×932 width is held out until the selected candidate is rebuilt for the
final B [1,3] check. Every navigation waits for the HTML root's
`data-ready=true` marker and rejects page exceptions; it does not wait for a
correct answer. Fresh navigation resets application state before each
sequence. The result records source and runner SHA-256 hashes, four visual
criterion bits, the W1–W8 gate, selection, final data/export, and normalized
chart-title box coordinates. Chrome hides scrollbars for reproducible
captures; font rendering may still vary slightly across hosts. The runner
uses local `file:` pages and no model credentials or external host access.

The recording probe uses matched D2 and D3 desktop captures to check that
different image bytes reach a visual request and only the corresponding
scripted text feedback reaches a later coding request. It also confirms that
changing the viewport changes captured pixels with the same D3 source and
filter. This is an original request-boundary demonstration, not an upstream
model call or a test of visual understanding.

The bounded source probe below requires three exact public files at the
commits in `source_versions.json`: WebGen-Bench
`src/ui_test_bolt/compute_acc.py`, Design2Code
`Design2Code/metrics/visual_score.py`, and WebGen-Agent
`WebGen-RL/verl/trainer/ppo/core_algos.py`. Keep them outside this module;
the script checks their SHA-256 values before executing any source function.
From the companion-code root, with your own paths to those files:

```sh
python3 -B kimi_code_test_grounded_software_engineering/web_visual_artifact_learning/probes/source_bound_metrics.py \
  --webgen-bench-source /path/to/compute_acc.py \
  --design2code-source /path/to/visual_score.py \
  --webgen-core-source /path/to/core_algos.py
```

This executes the pinned Bolt.diy aggregation script on four invented
operations and isolated pinned distance/step-estimator function bodies with
small inputs. It does not run the Design2Code OCR/CLIP pipeline, the
WebGen-Agent trainer, or a K3 model. The book-authored arithmetic example
remains in `training_example/step_coefficients.mjs` and is a separate check.
The exact final outputs of the two probes are retained in
`evidence/feedback_boundary.json` and `evidence/source_bound_metrics.json`.
