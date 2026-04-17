"""Small text normalization helpers."""

from __future__ import annotations

import re


def normalize_title(title: str) -> str:
    """Normalize a title for matching and dedupe."""

    cleaned = re.sub(r"[^a-z0-9]+", " ", title.casefold())
    return " ".join(cleaned.split())
