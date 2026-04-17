"""PDF entry matching helpers."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..config import BibConfig
from ..metadata.text import normalize_title
from ..models import BibEntry, ExtractedPdfMetadata


def match_existing_entry(
    pdf: ExtractedPdfMetadata, entries: list[BibEntry], config: BibConfig
) -> tuple[BibEntry | None, float, list[str]]:
    """Find the best existing BibTeX entry for a PDF."""
    best_entry, best_score, best_reasons = _best_match(pdf, entries)
    if best_score >= config.pdf_sync.match_min_confidence:
        return best_entry, best_score, best_reasons
    return None, best_score, best_reasons


def _best_match(
    pdf: ExtractedPdfMetadata, entries: list[BibEntry]
) -> tuple[BibEntry | None, float, list[str]]:
    best_entry: BibEntry | None = None
    best_score = 0.0
    best_reasons: list[str] = []
    for entry in entries:
        exact = _exact_match(pdf, entry)
        if exact is not None:
            return exact
        score, reasons = _scored_match(pdf, entry)
        if score > best_score:
            best_entry, best_score, best_reasons = entry, score, reasons
    return best_entry, best_score, best_reasons


def _exact_match(
    pdf: ExtractedPdfMetadata, entry: BibEntry
) -> tuple[BibEntry, float, list[str]] | None:
    if entry.fields.get("file") == pdf.relative_pdf_path:
        return entry, 1.0, ["existing file field match"]
    if pdf.doi and entry.doi and pdf.doi.casefold() == entry.doi.casefold():
        return entry, 1.0, ["exact DOI match"]
    return None


def _scored_match(
    pdf: ExtractedPdfMetadata, entry: BibEntry
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    score += _title_score(pdf, entry, reasons)
    score += _year_score(pdf, entry, reasons)
    score += _author_score(pdf, entry, reasons)
    score += _venue_score(pdf, entry, reasons)
    return score, reasons


def _title_score(
    pdf: ExtractedPdfMetadata, entry: BibEntry, reasons: list[str]
) -> float:
    pdf_title = normalize_title(pdf.title or "")
    entry_title = normalize_title(entry.title)
    if not (pdf_title and entry_title):
        return 0.0
    title_score = _similarity(pdf_title, entry_title)
    reasons.append(f"title={title_score:.2f}")
    return 0.7 * title_score


def _year_score(
    pdf: ExtractedPdfMetadata, entry: BibEntry, reasons: list[str]
) -> float:
    if pdf.year is None or entry.year is None:
        return 0.0
    if pdf.year == entry.year:
        reasons.append("year exact")
        return 0.15
    if abs(pdf.year - entry.year) == 1:
        reasons.append("year near")
        return 0.05
    return 0.0


def _author_score(
    pdf: ExtractedPdfMetadata, entry: BibEntry, reasons: list[str]
) -> float:
    first_author = _first_author_token(pdf.authors)
    entry_author = _bibtex_first_author_token(entry.fields.get("author", ""))
    if first_author and entry_author and first_author == entry_author:
        reasons.append("author exact")
        return 0.1
    return 0.0


def _venue_score(
    pdf: ExtractedPdfMetadata, entry: BibEntry, reasons: list[str]
) -> float:
    if not (pdf.venue and entry.venue):
        return 0.0
    venue_score = _similarity(normalize_title(pdf.venue), normalize_title(entry.venue))
    if venue_score:
        reasons.append(f"venue={venue_score:.2f}")
        return 0.05 * venue_score
    return 0.0


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def _first_author_token(authors: list[str]) -> str | None:
    if not authors:
        return None
    name = authors[0].strip()
    if not name:
        return None
    token = name.split()[-1].casefold()
    return "".join(char for char in token if char.isalnum())


def _bibtex_first_author_token(author_field: str) -> str | None:
    if not author_field.strip():
        return None
    first_author = author_field.split(" and ", 1)[0].strip()
    if "," in first_author:
        surname = first_author.split(",", 1)[0].strip()
        if surname:
            return "".join(char for char in surname.casefold() if char.isalnum())
    parts = [part for part in first_author.split() if part]
    if not parts:
        return None
    return "".join(char for char in parts[-1].casefold() if char.isalnum())
