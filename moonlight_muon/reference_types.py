"""Shared matrix aliases and constants for the Chapter 12 companion."""

from __future__ import annotations

Matrix = tuple[tuple[float, ...], ...]
FlatTensor = tuple[float, ...]
NS5_COEFFICIENTS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

__all__ = ["FlatTensor", "Matrix", "NS5_COEFFICIENTS"]
