"""Resolver and enrichment pipeline for BibTeX entries."""

from __future__ import annotations

from .core import resolve_entry, score_candidate

__all__ = ["resolve_entry", "score_candidate"]
