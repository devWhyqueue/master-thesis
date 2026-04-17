"""Metadata resolution, provider access, screening, and reporting."""

from __future__ import annotations

from .cache import ResponseCache, resolve_cache_paths
from .providers import CrossrefProvider, OpenAlexProvider
from .reporting import (
    render_duplicates,
    render_enrichment_preview,
    render_enrichment_summary,
    render_pdf_sync_preview,
    render_pdf_sync_summary,
    render_screening_summary,
)
from .resolve import resolve_entry, score_candidate
from .screening import detect_duplicates, score_entry, screening_updates
from .text import normalize_title

__all__ = [
    "CrossrefProvider",
    "OpenAlexProvider",
    "ResponseCache",
    "detect_duplicates",
    "normalize_title",
    "render_duplicates",
    "render_enrichment_preview",
    "render_enrichment_summary",
    "render_pdf_sync_preview",
    "render_pdf_sync_summary",
    "render_screening_summary",
    "resolve_cache_paths",
    "resolve_entry",
    "score_candidate",
    "score_entry",
    "screening_updates",
]
