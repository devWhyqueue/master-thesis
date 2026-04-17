"""Scoring and duplicate logic for BibTeX screening."""

from __future__ import annotations

from .core import detect_duplicates, score_entry, screening_updates

__all__ = ["detect_duplicates", "score_entry", "screening_updates"]
