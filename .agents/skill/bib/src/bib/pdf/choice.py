"""PDF filename and extracted-text selection heuristics."""

from __future__ import annotations

from ..metadata.text import normalize_title
from .text import _clean_text


def _choose_title(filename_title: str | None, text_title: str | None) -> str | None:
    filename_title = _clean_text(filename_title)
    text_title = _clean_text(text_title)
    if not filename_title:
        return text_title
    if not text_title:
        return filename_title
    filename_tokens = set(normalize_title(filename_title).split())
    text_tokens = set(normalize_title(text_title).split())
    if not filename_tokens:
        return text_title
    overlap = len(filename_tokens & text_tokens) / max(len(filename_tokens), 1)
    domain_tokens = {
        "computational",
        "digital",
        "pathology",
        "foundation",
        "foundational",
        "models",
        "model",
        "datasets",
        "adaptation",
        "evaluation",
        "strategies",
        "artificial",
        "intelligence",
        "progress",
        "future",
        "directions",
        "survey",
    }
    domain_gain = len((text_tokens & domain_tokens) - (filename_tokens & domain_tokens))
    if overlap >= 0.6 and (
        len(text_tokens) - len(filename_tokens) >= 2
        or (":" in text_title and ":" not in filename_title)
        or domain_gain >= 2
        or (len(filename_tokens) <= 5 and len(text_tokens) > len(filename_tokens))
    ):
        return text_title
    return filename_title


def _choose_authors(
    filename_authors: list[str],
    text_authors: list[str],
    chosen_title: str | None,
    filename_title: str | None,
) -> list[str]:
    cleaned_filename_authors: list[str] = []
    for author in filename_authors:
        cleaned = _clean_text(author)
        if cleaned:
            cleaned_filename_authors.append(cleaned)
    cleaned_text_authors: list[str] = []
    for author in text_authors:
        cleaned = _clean_text(author)
        if cleaned:
            cleaned_text_authors.append(cleaned)
    filename_authors = cleaned_filename_authors
    text_authors = cleaned_text_authors
    if not filename_authors:
        return text_authors
    if not text_authors:
        return filename_authors
    text_has_full_names = any(len(author.split()) >= 2 for author in text_authors)
    filename_weak = (
        len(filename_authors) == 1
        and (
            "et al" in filename_authors[0].casefold()
            or len(filename_authors[0].split()) == 1
        )
    ) or all(len(author.split()) == 1 for author in filename_authors)
    if filename_weak and text_has_full_names:
        return text_authors
    upgraded_title = normalize_title(chosen_title or "") != normalize_title(
        filename_title or ""
    )
    return (
        text_authors
        if upgraded_title
        and len(text_authors) > len(filename_authors)
        and text_has_full_names
        else filename_authors
    )
