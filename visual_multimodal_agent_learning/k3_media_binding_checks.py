"""Bounded K3 caller-path checks, with an optional pinned-source probe."""

from __future__ import annotations

import ast
from hashlib import sha256
import os
from pathlib import Path
from typing import Any

from .visual_observation_trace import Media, validate_binding

PINNED_PROCESSOR_SHA256 = (
    "ec9f7e86d2ab0eee07a8e7e7c037046e77ac3c25a710ad1298ec13be3b585b54"
)


def automatic_media_from_messages(messages: list[dict[str, Any]]) -> list[Media]:
    """Book-side adapter for the pinned processor's user-role extraction rule."""

    result: list[Media] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                image = part.get(part["type"])
                if not isinstance(image, Media):
                    raise ValueError("image content lacks a media object")
                result.append(image)
    return result


def explicit_media_binding(
    text: str,
    medias: list[Media],
    expected: list[Media],
    *,
    prepared_grid_count: int,
) -> dict[str, object]:
    """Check the declared explicit text/media path before a K3 processor call."""

    if text.count("<|kimi_image_placeholder|>") != len(medias):
        raise ValueError("placeholder/media count mismatch")
    validate_binding(expected, medias, prepared_grids=prepared_grid_count)
    return {
        "path": "explicit-text-and-medias",
        "image_ids": [media.event_id for media in medias],
        "dimensions": [[media.width, media.height] for media in medias],
        "grid_count": prepared_grid_count,
    }


class _FakeBatchFeature:
    def __init__(self, data: dict[str, object]) -> None:
        self.data = data


class _FakeProcessorMixin:
    def __init__(self, image_processor: Any, tokenizer: Any, **_: Any) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer


class _FakeImage:
    def __init__(self, size: tuple[int, int], name: str) -> None:
        self.size = size
        self.name = name


class _FakeMediaProcessor:
    _transparent_bg_config = None
    _transparent_bg_fill_stage = None

    def make_image_prompt(self, width: int, height: int) -> str:
        return f"<image:{width}x{height}>"

    def preprocess(self, medias: list[dict], **_: Any) -> _FakeBatchFeature:
        return _FakeBatchFeature(
            {
                "pixel_values": [media["image"].name for media in medias],
                "grid_thws": [media["image"].size for media in medias],
            }
        )


class _FakeTokenizer:
    def __call__(self, text: str, **_: Any) -> dict[str, object]:
        return {"input_ids": [len(text)], "attention_mask": [1]}

    def apply_chat_template(self, messages: list[dict], **_: Any) -> dict[str, object]:
        return {"input_ids": [len(messages)], "attention_mask": [1]}


def probe_pinned_extractor() -> dict[str, object]:
    """Exercise exact pinned processor routing with stub media preparation.

    The source hash is checked first. Stub media preparation makes this a
    caller-path test, not a test of real K3 image tensors or visual token grids.
    """

    processor_file = os.environ.get("KIMI_K3_PROCESSOR_FILE")
    if not processor_file:
        raise RuntimeError("KIMI_K3_PROCESSOR_FILE is not configured")
    source_path = Path(processor_file)
    source = source_path.read_bytes()
    digest = sha256(source).hexdigest()
    if digest != PINNED_PROCESSOR_SHA256:
        raise ValueError("configured processor differs from registered pin")
    tree = ast.parse(source, filename=str(source_path))
    processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiK3Processor"
    )

    namespace: dict[str, Any] = {
        "ProcessorMixin": _FakeProcessorMixin,
        "BatchFeature": _FakeBatchFeature,
        "ensure_media_type": lambda media, **_: media,
    }
    module = ast.fix_missing_locations(ast.Module(body=[processor], type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    instance = namespace["KimiK3Processor"](
        image_processor=_FakeMediaProcessor(), tokenizer=_FakeTokenizer()
    )
    root = _FakeImage((160, 160), "root")
    crop = _FakeImage((300, 600), "crop")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": root}]},
        {"role": "tool", "content": [{"type": "image", "image": crop}]},
    ]
    automatic = instance(messages=messages).data
    explicit = instance(
        medias=[{"type": "image", "image": crop}],
        text="<|kimi_image_placeholder|>",
    ).data
    path_only = instance(
        messages=[{"role": "user", "content": [{"type": "text", "text": "crop.png"}]}]
    ).data
    if automatic["pixel_values"] != ["root"] or automatic["grid_thws"] != [(160, 160)]:
        raise AssertionError("pinned extractor behavior changed")
    if explicit["pixel_values"] != ["crop"] or explicit["grid_thws"] != [(300, 600)]:
        raise AssertionError("pinned explicit media path behavior changed")
    if path_only["pixel_values"]:
        raise AssertionError("path-only message unexpectedly supplied media")
    return {
        "processor_sha256": digest,
        "user_image_extracted": True,
        "tool_image_automatic_extraction": False,
        "explicit_media_path": "crop reaches stub pixel/grid fields",
        "path_only_media_count": 0,
        "real_image_preprocessing_executed": False,
    }
