"""Utilities for span annotation parsing, alignment, and evaluation."""

from .alignment import align_texts, make_aligner

__all__ = [
    "align_texts",
    "make_aligner",
]
__version__ = "0.1.0"
