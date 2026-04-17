"""PDF sync entry creation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import BibConfig
from ..models import BibEntry, ExtractedPdfMetadata, ResolvedMetadata


@dataclass(slots=True)
class PdfSyncResult:
    """Result for one PDF processed by pdf-sync."""

    pdf: ExtractedPdfMetadata
    matched_key: str | None
    match_confidence: float
    created_key: str | None
    updates: dict[str, str | None]
    new_entry: dict[str, object] | None
    reasons: list[str]


@dataclass(slots=True)
class MergedPdfMetadata:
    """Merged PDF-local and resolved metadata."""

    title: str | None
    authors: list[str]
    year: int | None
    journal: str | None
    booktitle: str | None
    publisher: str | None
    volume: str | None
    number: str | None
    pages: str | None
    doi: str | None
    url: str | None
    file_field: str | None
    publication_type: str | None


def build_partial_entry(pdf: ExtractedPdfMetadata) -> BibEntry:
    """Build a synthetic BibEntry from extracted PDF metadata."""
    fields = _partial_entry_fields(pdf)
    return BibEntry(
        entry_type=pdf.entry_type or "article",
        key="__pdfsync__",
        fields=fields,
        field_order=list(fields),
        start=0,
        end=0,
        raw_text="",
        header=f"@{pdf.entry_type or 'article'}{{__pdfsync__,",
        field_segments=[],
    )


def create_new_entry(
    pdf: ExtractedPdfMetadata,
    resolved: ResolvedMetadata,
    existing_keys: set[str],
    config: BibConfig,
) -> dict[str, object] | None:
    """Create a new BibTeX entry from PDF and resolved metadata."""

    merged = _merge_pdf_and_resolved(pdf, resolved)
    if (
        creation_confidence(pdf, resolved) < config.pdf_sync.create_min_confidence
        or not merged.title
    ):
        return None
    key = generate_key(merged, existing_keys)
    entry_type = _entry_type_from_metadata(merged)
    fields = _render_new_entry_fields(merged, pdf.relative_pdf_path)
    return {"entry_type": entry_type, "key": key, "fields": fields}


def _partial_entry_fields(pdf: ExtractedPdfMetadata) -> dict[str, str]:
    fields: dict[str, str] = {}
    if pdf.title:
        fields["title"] = pdf.title
    if pdf.authors:
        fields["author"] = " and ".join(pdf.authors)
    if pdf.year is not None:
        fields["year"] = str(pdf.year)
    if pdf.doi:
        fields["doi"] = pdf.doi
    if pdf.url:
        fields["url"] = pdf.url
    if pdf.venue:
        fields["booktitle" if pdf.entry_type == "inproceedings" else "journal"] = (
            pdf.venue
        )
    fields["file"] = pdf.file_field or pdf.relative_pdf_path
    return fields


def _render_new_entry_fields(
    merged: MergedPdfMetadata, relative_pdf_path: str
) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if merged.authors:
        fields.append(("author", " and ".join(merged.authors)))
    if merged.title is not None:
        fields.append(("title", merged.title))
    for name in (
        "journal",
        "booktitle",
        "publisher",
        "volume",
        "number",
        "pages",
        "doi",
        "url",
    ):
        value = getattr(merged, name)
        if value:
            fields.append((name, value))
    if merged.year is not None:
        fields.append(("year", str(merged.year)))
    fields.append(("file", merged.file_field or relative_pdf_path))
    return fields


def creation_confidence(pdf: ExtractedPdfMetadata, resolved: ResolvedMetadata) -> float:
    """Return creation confidence for a PDF-derived entry."""

    return _creation_confidence(_merge_pdf_and_resolved(pdf, resolved))


def generate_key(metadata: MergedPdfMetadata, existing_keys: set[str]) -> str:
    """Generate an AuthorYearSlug key with collision suffixes."""

    author = _first_author_token(metadata.authors) or "unknown"
    year = str(metadata.year) if metadata.year is not None else "nodate"
    title_slug = _slugify_title(metadata.title or "untitled")
    base = f"{author}{year}{title_slug}"
    key = base
    counter = 2
    while key in existing_keys:
        key = f"{base}-{counter}"
        counter += 1
    return key


def _merge_pdf_and_resolved(
    pdf: ExtractedPdfMetadata, resolved: ResolvedMetadata
) -> MergedPdfMetadata:
    venue = resolved.venue or pdf.venue
    entry_type = _entry_type_from_type(resolved.publication_type, venue, pdf.entry_type)
    journal = venue if entry_type == "article" else None
    booktitle = venue if entry_type == "inproceedings" else None
    return MergedPdfMetadata(
        title=resolved.title or pdf.title,
        authors=pdf.authors,
        year=resolved.year or pdf.year,
        journal=resolved.journal or journal,
        booktitle=resolved.booktitle or booktitle,
        publisher=resolved.publisher,
        volume=resolved.volume,
        number=resolved.number,
        pages=resolved.pages,
        doi=resolved.doi or pdf.doi,
        url=resolved.url or pdf.url,
        file_field=pdf.file_field,
        publication_type=resolved.publication_type or pdf.entry_type,
    )


def _creation_confidence(merged: MergedPdfMetadata) -> float:
    score = 0.0
    if merged.title:
        score += 0.5
    if merged.authors:
        score += 0.15
    if merged.year is not None:
        score += 0.15
    if merged.doi:
        score += 0.2
    return min(score, 1.0)


def _entry_type_from_metadata(metadata: MergedPdfMetadata) -> str:
    return _entry_type_from_type(
        metadata.publication_type, metadata.booktitle or metadata.journal, None
    )


def _entry_type_from_type(
    publication_type: str | None, venue: str | None, fallback: str | None
) -> str:
    value = (publication_type or "").casefold()
    if "proceedings" in value or fallback == "inproceedings":
        return "inproceedings"
    if venue and any(
        token in venue.casefold()
        for token in ("conference", "iclr", "neurips", "icml", "cvpr")
    ):
        return "inproceedings"
    return "article"


def _first_author_token(authors: list[str]) -> str | None:
    if not authors:
        return None
    name = authors[0].strip()
    if not name:
        return None
    token = name.split()[-1].casefold()
    return re.sub(r"[^a-z0-9]+", "", token)


def _bibtex_first_author_token(author_field: str) -> str | None:
    if not author_field.strip():
        return None
    first_author = author_field.split(" and ", 1)[0].strip()
    if "," in first_author:
        surname = first_author.split(",", 1)[0].strip()
        if surname:
            return re.sub(r"[^a-z0-9]+", "", surname.casefold())
    parts = [part for part in first_author.split() if part]
    if not parts:
        return None
    return re.sub(r"[^a-z0-9]+", "", parts[-1].casefold())


def _slugify_title(title: str) -> str:
    words = [word for word in re.findall(r"[A-Za-z0-9]+", title) if word]
    if not words:
        return "untitled"
    first = words[0].casefold()
    second = words[1].capitalize() if len(words) > 1 else ""
    return f"{first}{second}"
