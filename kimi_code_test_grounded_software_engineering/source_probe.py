"""Read-only checks against the exact public source revisions used in Chapter 29."""

from __future__ import annotations

import ast
from collections.abc import Callable
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

PINS = {
    "KIMI_DEV_CHECKOUT": (
        "5f60757f5ef54f66455c209aca169c8f850b96bd",
        "https://github.com/MoonshotAI/Kimi-Dev",
    ),
    "KIMI_CODE_CH29_CHECKOUT": (
        "eba23edb93ec00aa11ffa288e2b89ebbf3d77a70",
        "https://github.com/MoonshotAI/kimi-code",
    ),
    "KIMI_KERNELBENCH_CHECKOUT": (
        "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
        "https://github.com/ScalingIntelligence/KernelBench",
    ),
}
FILE_HASHES = {
    "bugfixer": (
        "KIMI_DEV_CHECKOUT",
        "kimidev/examples/rollout_messages_bugfixer.py",
        "6865cd2f7b2291c40367a07b6d654110670202c099834fe5b67b1aa85da96f10",
    ),
    "testwriter": (
        "KIMI_DEV_CHECKOUT",
        "kimidev/examples/rollout_messages_testwriter.py",
        "edb02d1f0f3869508db8527b8ffde7fece04117ca428b38d73b5a307e0e266e2",
    ),
    "edit_tool": (
        "KIMI_CODE_CH29_CHECKOUT",
        "packages/agent-core-v2/src/agent/tools/edit/editTool.ts",
        "321865050bc70689dbf88a8a2bd99f4ced9feb5f5de8b1752b755b9bd1362594",
    ),
    "file_edit": (
        "KIMI_CODE_CH29_CHECKOUT",
        "packages/agent-core-v2/src/app/edit/fileEditService.ts",
        "a6adc6ca4648c106de91c05ae4255066d4385a048df940d15fc7080cca343f02",
    ),
    "edit_apply": (
        "KIMI_CODE_CH29_CHECKOUT",
        "packages/agent-core-v2/src/app/edit/editService.ts",
        "c586e7686a0227e16c9d6879f3a8d196db11bfe91be199b4e04c7837d91531e8",
    ),
    "kernel_eval": (
        "KIMI_KERNELBENCH_CHECKOUT",
        "src/kernelbench/eval.py",
        "b42ce1268ec64ee7274db307ab98945c1e868792b54689554014b28c41787ee6",
    ),
}


BUGFIXER_FIXTURE_FILES = {
    "canvas.py": "root_point = viewport_to_root(u, v, crop, viewport)",
    "transforms.py": "y = y0 + (wc / Wv) * v",
}


def _fixture_repo_files(_structure: object, names: list[str]) -> dict[str, str]:
    return {name: BUGFIXER_FIXTURE_FILES[name] for name in names}


class _PromptRecorder:
    """Collect prompts emitted by the pinned prompt function."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, _model: object, prompt: str) -> tuple[list[str], str]:
        self.prompts.append(prompt)
        return [], "recorded"


def checkout_roots() -> dict[str, Path]:
    """Resolve pinned clean source checkouts from their declared environment."""

    roots: dict[str, Path] = {}
    for env_name, (commit, remote) in PINS.items():
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"{env_name} is unset")
        root = Path(value).resolve(strict=True)
        actual_commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        actual_remote = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
        if actual_commit != commit or actual_remote.removesuffix(".git") != remote:
            raise RuntimeError(
                f"{env_name} does not match its recorded commit and remote"
            )
        if dirty:
            raise RuntimeError(f"{env_name} has local changes")
        roots[env_name] = root
    return roots


def pinned_sources(roots: dict[str, Path]) -> dict[str, str]:
    """Load and hash-check the chapter's pinned upstream source files."""

    sources: dict[str, str] = {}
    for name, (env_name, relative, expected_hash) in FILE_HASHES.items():
        data = (roots[env_name] / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise RuntimeError(f"{name} content differs from the pinned source record")
        sources[name] = data.decode("utf-8")
    return sources


def bugfixer_prompt_probe(source: str) -> None:
    """Execute the actual pinned prompt function with a recording model stub."""
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "repair_prompt_response"
    )
    recorder = _PromptRecorder()

    namespace: dict[str, object] = {
        "get_repo_files": _fixture_repo_files,
        "llm_chat": recorder,
    }
    # Execute only this function from a commit- and hash-checked official source file.
    exec(  # noqa: S102
        compile(
            ast.Module(body=[function], type_ignores=[]), "<pinned-bugfixer>", "exec"
        ),
        namespace,
    )
    repair = namespace["repair_prompt_response"]
    if not callable(repair):
        raise RuntimeError("pinned repair function is not callable")
    repair(None, "wrong root point", {}, ["canvas.py"])
    repair(None, "wrong root point", {}, ["transforms.py"])
    if "root_point =" not in recorder.prompts[0] or "y = y0 +" in recorder.prompts[0]:
        raise RuntimeError("caller-only prompt has unexpected contents")
    if "y = y0 +" not in recorder.prompts[1] or "root_point =" in recorder.prompts[1]:
        raise RuntimeError("transform-only prompt has unexpected contents")
    if (
        "SEARCH/REPLACE" not in recorder.prompts[0]
        or "SEARCH/REPLACE" not in recorder.prompts[1]
    ):
        raise RuntimeError("pinned edit request format changed")


def testwriter_control_probe(source: str) -> None:
    """Run the pinned TestWriter controller with local file and model stubs."""
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve_instance_data"
    )
    chosen_paths: list[list[str]] = []
    localization_calls = [0]
    generated_patches: list[str] = []

    def localize(
        _model: object, _issue: object, _structure: object
    ) -> tuple[list[object], list[str]]:
        localization_calls[0] += 1
        return [], [
            "src/canvas.py",
            "tests/test_first.py",
            "tests/test_second.py",
        ]

    def edit_prompt(
        _model: object, _issue: object, _structure: object, names: list[str]
    ) -> tuple[list[object], str, dict[str, str]]:
        chosen_paths.append(names)
        return [], "replacement", {}

    def convert_patch(**_kwargs: object) -> str:
        return generated_patches.pop(0)

    namespace: dict[str, object] = {
        "os": os,
        "json": json,
        "search_instance_id_and_extract_structure": lambda *_args: {},
        "parse_patch": lambda _patch: {},
        "get_relevant_files": lambda _instance: ["tests/test_oracle.py"],
        "generate_found_edit_locs": lambda *_args: ([], []),
        "relevant_file_prompt_response": localize,
        "repair_prompt_response": edit_prompt,
        "generate_model_patch_difflib_testwritter": convert_patch,
    }
    # Execute only this function from a commit- and hash-checked official source file.
    exec(  # noqa: S102
        compile(
            ast.Module(body=[function], type_ignores=[]), "<pinned-testwriter>", "exec"
        ),
        namespace,
    )
    solve = namespace["solve_instance_data"]
    if not callable(solve):
        raise RuntimeError("pinned TestWriter controller is not callable")
    with tempfile.TemporaryDirectory(prefix="kimi-ch29-testwriter-") as temporary:
        for enable_gt, expected in (
            (False, ["tests/test_first.py"]),
            (True, ["tests/test_oracle.py"]),
        ):
            _assert_testwriter_case(
                solve,
                temporary,
                enable_gt,
                expected,
                chosen_paths,
                localization_calls,
                generated_patches,
            )


def _assert_testwriter_case(
    solve: Callable[..., object],
    temporary: str,
    enable_gt: bool,
    expected: list[str],
    chosen_paths: list[list[str]],
    localization_calls: list[int],
    generated_patches: list[str],
) -> None:
    chosen_paths.clear()
    localization_calls[0] = 0
    generated_patches[:] = ["", "@@ converted patch"]
    instance_id = f"probe-{int(enable_gt)}"
    result = solve(
        {"instance_id": instance_id, "problem_statement": "issue"},
        "unused",
        temporary,
        set(),
        [],
        SimpleNamespace(temperature=0.0),
        enable_gt=enable_gt,
        retry_times=2,
    )
    saved = json.loads(
        (Path(temporary) / f"{instance_id}.jsonl").read_text(encoding="utf-8")
    )
    if result != 1 or chosen_paths != [expected, expected]:
        raise RuntimeError("TestWriter file-selection branch changed")
    if localization_calls[0] != 2 or saved["pred_found_files"] != [
        "tests/test_first.py"
    ]:
        raise RuntimeError("TestWriter effective one-file or retry path changed")
    if saved["retry_times"] != 2 or saved["model_patch"] != "@@ converted patch":
        raise RuntimeError("TestWriter patch-shape retry outcome changed")


def _kernelbench_branch_probe(kernel: str) -> None:
    ast.parse(kernel)
    if "if kernel_exec_result and kernel_exec_result.correctness:" not in kernel:
        raise RuntimeError("KernelBench main timing condition changed")
    if "if measure_performance and check_for_excessive_speedup:" not in kernel:
        raise RuntimeError("KernelBench separate excessive-speedup branch changed")


def _kimi_code_edit_branch_probe(tool: str, file_edit: str, edit_apply: str) -> None:
    for required in (
        "ToolAccesses.readWriteFile(path)",
        "lease.runtime.identity.generation !== inspected.identity.generation",
        "this.editor.edit(",
    ):
        if required not in tool:
            raise RuntimeError(f"Kimi Code edit tool branch changed: {required}")
    if (
        "fs.readText(input.path" not in file_edit
        or "fs.writeText(input.path" not in file_edit
    ):
        raise RuntimeError("Kimi Code current-file read/write path changed")
    if "if (count === 0)" not in edit_apply or "if (count > 1)" not in edit_apply:
        raise RuntimeError("Kimi Code missing/ambiguous match path changed")


def source_branch_probe(sources: dict[str, str]) -> None:
    """Check source-bound branches without claiming native end-to-end execution."""
    bugfixer_prompt_probe(sources["bugfixer"])
    testwriter_control_probe(sources["testwriter"])
    bf = sources["bugfixer"]
    tw = sources["testwriter"]
    if "if enable_gt == True:" not in bf or 'gt_results["found_files"]' not in bf:
        raise RuntimeError("BugFixer oracle-file branch changed")
    if "model_patch == \"\" or '@@' not in model_patch" not in bf:
        raise RuntimeError("BugFixer patch-shape retry changed")
    if "base_name.startswith('test')" not in tw or "pred_found_files[:1]" not in tw:
        raise RuntimeError("TestWriter effective predicted-file selection changed")
    if (
        "if(len(pred_found_files) > 0):" not in tw
        or 'gt_results["found_files"]' not in tw
    ):
        raise RuntimeError("TestWriter reference-file condition changed")
    if "model_patch == \"\" or '@@' not in model_patch" not in tw:
        raise RuntimeError("TestWriter patch-shape retry changed")

    _kimi_code_edit_branch_probe(
        sources["edit_tool"], sources["file_edit"], sources["edit_apply"]
    )
    _kernelbench_branch_probe(sources["kernel_eval"])


def main() -> int:
    """Run source-pinned probes and report their scope limits."""

    source_branch_probe(pinned_sources(checkout_roots()))
    print(
        "PASS: pinned sources, BugFixer prompt capture, TestWriter control flow, "
        "and inspected edit/timing branches"
    )
    print("LIMIT: TypeScript and GPU branches were inspected, not executed natively")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
