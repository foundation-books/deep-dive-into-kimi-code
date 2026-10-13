"""Exact source-image conventions and a deterministic chart-detail fixture."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import struct
import zlib


def xywh_to_edges(rectangle: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert a positive-size x/y/width/height crop to edge coordinates."""

    x, y, width, height = rectangle
    if width <= 0 or height <= 0:
        raise ValueError("crop dimensions must be positive")
    return x, y, x + width, y + height


def viewport_edges_to_root(
    rectangle: tuple[int, int, int, int],
    *,
    source_crop: tuple[int, int, int, int],
    viewport: tuple[int, int],
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Map a current-viewport edge rectangle back to its root-image edges."""

    x0, y0, source_width, source_height = source_crop
    viewport_width, viewport_height = viewport
    left, top, right, bottom = rectangle
    if not (
        0 <= left < right <= viewport_width and 0 <= top < bottom <= viewport_height
    ):
        raise ValueError("rectangle outside current viewport")
    return (
        Fraction(x0) + Fraction(source_width * left, viewport_width),
        Fraction(y0) + Fraction(source_height * top, viewport_height),
        Fraction(x0) + Fraction(source_width * right, viewport_width),
        Fraction(y0) + Fraction(source_height * bottom, viewport_height),
    )


def chart_source() -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """A 320x160 chart with one-pixel ticks that an 80x40 preview loses."""

    rows: list[tuple[tuple[int, int, int], ...]] = []
    for y in range(160):
        row: list[tuple[int, int, int]] = []
        for x in range(320):
            color = (250, 250, 250)
            if x == 30 or y == 130:
                color = (25, 25, 25)
            if 60 <= x < 105 and 78 <= y < 130:
                color = (44, 104, 170)
            if 150 <= x < 195 and 58 <= y < 130:
                color = (196, 85, 58)
            if 225 <= x < 270 and 42 <= y < 130:
                color = (48, 137, 91)
            if 210 <= x < 290 and 20 <= y < 37 and x % 4 == 1:
                color = (15, 15, 15)
            row.append(color)
        rows.append(tuple(row))
    return tuple(rows)


def source_and_preview_details() -> tuple[bytes, bytes, int, int]:
    """Return same-size RGB details from source and downsampled preview."""

    source = chart_source()
    preview = tuple(tuple(source[y * 4][x * 4] for x in range(80)) for y in range(40))
    # Source rectangle (208, 16, 288, 56): 80x40 pixels. The preview
    # rectangle (52, 4, 72, 14) is enlarged back by nearest neighbour.
    source_detail = bytes(
        channel
        for y in range(16, 56)
        for x in range(208, 288)
        for channel in source[y][x]
    )
    preview_detail = bytes(
        channel
        for y in range(4, 14)
        for _ in range(4)
        for x in range(52, 72)
        for _ in range(4)
        for channel in preview[y][x]
    )
    return source_detail, preview_detail, 80, 40


def _png(rgb: bytes, width: int, height: int) -> bytes:
    if len(rgb) != width * height * 3:
        raise ValueError("RGB dimensions disagree")
    raw = b"".join(
        b"\x00" + rgb[row * width * 3 : (row + 1) * width * 3] for row in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def write_chart_fixture(directory: Path) -> None:
    """Write only the three declared deterministic Chapter 25 PNGs."""

    directory.mkdir(parents=True, exist_ok=True)
    source = chart_source()
    source_rgb = bytes(channel for row in source for pixel in row for channel in pixel)
    high, preview, width, height = source_and_preview_details()
    (directory / "chart-source.png").write_bytes(_png(source_rgb, 320, 160))
    (directory / "chart-source-detail.png").write_bytes(_png(high, width, height))
    (directory / "chart-preview-enlarged.png").write_bytes(_png(preview, width, height))


def write_studio_fixture(directory: Path) -> None:
    """Render only the declared studio geometry, with a distinct same-shape crop."""

    directory.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[tuple[int, int, int], ...]] = []
    for y in range(160):
        row: list[tuple[int, int, int]] = []
        for x in range(160):
            color = (247, 247, 247)
            if (x - 40) ** 2 + (y - 40) ** 2 <= 20**2:
                color = (176, 65, 62)
            if 20 <= x < 60 and 80 <= y < 120:
                color = (42, 98, 157)
            row.append(color)
        rows.append(tuple(row))
    root = tuple(rows)
    crop = bytes(
        channel
        for y in range(20, 140)
        for _ in range(5)
        for x in range(10, 70)
        for _ in range(5)
        for channel in root[y][x]
    )
    distractor = crop.replace(bytes((42, 98, 157)), bytes((49, 130, 73)))
    root_rgb = bytes(channel for row in root for pixel in row for channel in pixel)
    (directory / "studio-root.png").write_bytes(_png(root_rgb, 160, 160))
    (directory / "studio-crop.png").write_bytes(_png(crop, 300, 600))
    (directory / "studio-distractor-crop.png").write_bytes(_png(distractor, 300, 600))


if __name__ == "__main__":
    fixture_directory = Path(__file__).with_name("fixtures")
    write_chart_fixture(fixture_directory)
    write_studio_fixture(fixture_directory)
