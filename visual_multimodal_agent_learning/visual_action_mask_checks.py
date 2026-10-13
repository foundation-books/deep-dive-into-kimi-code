"""Book-created visual action ownership and optional veRL source guard."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import subprocess

from .visual_observation_trace import (
    Event,
    action_mask,
    can_continue_after_tool,
    eligible_for_policy_batch,
)


def studio_mask_check() -> dict[str, object]:
    """Report assistant-span ownership for the studio teaching trace."""

    events = [
        Event("i0", "environment", "image", 0),
        Event("crop-call", "assistant", "call", 4),
        Event("i1", "executor", "image", 3),
        Event("answer", "assistant", "answer", 2),
    ]
    mask = action_mask(events)
    diagnostic = events + [Event("oracle-replay", "assistant", "answer", 2, True)]
    image_index = next(
        index
        for index, event in enumerate(events)
        if event.kind == "image" and event.author == "executor"
    )
    later_answer = any(
        event.kind == "answer" and event.author == "assistant"
        for event in events[image_index + 1 :]
    )
    return {
        "response_mask": mask,
        "sampled_positions": sum(mask),
        "image_conditions_later_answer": later_answer,
        "ordinary_batch_eligible": eligible_for_policy_batch(events),
        "diagnostic_batch_eligible": eligible_for_policy_batch(diagnostic),
        "full_budget_blocks_next_decision": not can_continue_after_tool(
            len(mask), len(mask)
        ),
    }


def probe_pinned_verl() -> dict[str, str]:
    """Guard rollout-mask construction and selected trainer consumption."""

    checkout = os.environ.get("KIMI_VERL_CHECKOUT")
    if not checkout:
        raise RuntimeError("KIMI_VERL_CHECKOUT is not configured")
    root = Path(checkout)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != "ea53291385ce764019a2b40733605f21d8317583":
        raise ValueError("veRL checkout differs from registered revision")
    remote = (
        subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .rstrip("/")
    )
    if remote != "https://github.com/verl-project/verl":
        raise ValueError("veRL checkout differs from registered remote")
    paths = {
        "tool_agent_loop": (
            "verl/experimental/agent_loop/tool_agent_loop.py",
            (
                "Continuous Token is the only tokenization path",
                "ct_merge_non_assistant_msg(",
                "response_mask=agent_data.response_mask[: self.response_length]",
            ),
        ),
        "agent_loop": (
            "verl/experimental/agent_loop/agent_loop.py",
            (
                'output["response_mask"] = torch.tensor(',
                '"response_mask": response_mask,',
            ),
        ),
        "ray_trainer": (
            "verl/trainer/ppo/ray_trainer.py",
            (
                'if "response_mask" not in batch.batch.keys():',
                'grpo_calculation_mask = data.batch["response_mask"]',
                "response_mask=grpo_calculation_mask,",
                "actor_output = self._update_actor(batch)",
            ),
        ),
    }
    subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"]
        + [relative for relative, _ in paths.values()],
        check=True,
    )
    digests = {"revision": revision, "remote": remote}
    for key, (relative, markers) in paths.items():
        source = (root / relative).read_bytes()
        source_text = source.decode("utf-8")
        if any(marker not in source_text for marker in markers):
            raise AssertionError(f"pinned veRL {key} mask handoff changed")
        digests[key] = sha256(source).hexdigest()
    return digests
