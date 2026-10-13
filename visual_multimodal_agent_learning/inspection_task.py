"""Protected truth and source geometry for the finite inspection study.

This module needs no image or model package. Rendering and optional model
execution live in the experiment runner; truth never enters its tool reply.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


RECTANGLES = (
    (0, 0, 336, 224),
    (336, 0, 672, 224),
    (0, 224, 336, 448),
    (336, 224, 672, 448),
)


@dataclass(frozen=True)
class PanelTask:
    """Author-side source specification, separate from the allowed view."""

    query: str
    panels: tuple[str, ...]
    values: tuple[int, ...]

    @property
    def answer(self) -> int:
        return self.values[self.panels.index(self.query)]

    def crop(self, action: int) -> tuple[int, int, int, int]:
        if not 0 <= action < len(RECTANGLES):
            raise ValueError("unavailable crop")
        return RECTANGLES[action]

    def allowed_text(self) -> str:
        return f"Read the numeric value in panel {self.query}. Crop executed."


def source_task(index: int, *, version_two: bool = True) -> PanelTask:
    """Reproduce the exact accepted source families, including their RNG."""
    if index < 0:
        raise ValueError("negative source origin")
    if version_two and not 44 <= index < 88:
        raise ValueError("origin outside the frozen version-two split")
    alphabet = "ABCD"
    if version_two and index >= 68:
        alphabet = "IJKL" if index < 72 else "MNOP"
    rng = random.Random(2700 + index)
    panels = list(alphabet)
    rng.shuffle(panels)
    values = tuple(rng.choice((2, 7)) for _ in panels)
    return PanelTask(alphabet[index % 4], tuple(panels), values)


def visual_positions(grids: tuple[tuple[int, int, int], ...]) -> int:
    """Four spatial patches merge into one visual position in this adapter."""
    if any(min(grid) <= 0 or grid[1] % 2 or grid[2] % 2 for grid in grids):
        raise ValueError("invalid merged image grid")
    return sum(t * h * w // 4 for t, h, w in grids)
