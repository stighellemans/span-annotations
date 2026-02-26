"""Utilities for span annotation parsing, alignment, and evaluation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .alignment import align_texts, make_aligner

__all__ = ["align_texts", "make_aligner"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in {"align_texts", "make_aligner"}:
        return getattr(import_module(".alignment", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
