"""PDF metadata extraction helpers."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..config import BibConfig
from ..models import ExtractedPdfMetadata
from .choice import _choose_authors, _choose_title
from .text import (
    _clean_text,
    _extract_authors,
    _extract_doi,
    _extract_title,
    _extract_venue,
    _extract_year,
    _infer_entry_type,
    _parse_author_line,
    _parse_filename,
)


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Return all PDFs under the directory in stable order."""

    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def extract_pdf_metadata(
    pdf_path: Path, bib_path: Path, config: BibConfig
) -> ExtractedPdfMetadata:
    """Extract lightweight metadata from a local PDF."""
    relative_pdf_path = pdf_path.relative_to(bib_path.parent).as_posix()
    filename_title, filename_authors = _parse_filename(pdf_path, config)
    doi, venue, year, text_title, text_authors = _extract_reader_metadata(
        pdf_path, config
    )
    title = _choose_title(filename_title, text_title)
    authors = _choose_authors(filename_authors, text_authors, title, filename_title)
    return ExtractedPdfMetadata(
        pdf_path=pdf_path,
        relative_pdf_path=relative_pdf_path,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        url=f"https://doi.org/{doi}" if doi else None,
        entry_type=_infer_entry_type(venue),
        file_field=relative_pdf_path,
        notes=[pdf_path.parent.name],
    )


def _extract_reader_metadata(
    pdf_path: Path, config: BibConfig
) -> tuple[str | None, str | None, int | None, str | None, list[str]]:
    doi = venue = text_title = None
    year: int | None = None
    text_authors: list[str] = []
    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError, ValueError):
        return doi, venue, year, text_title, text_authors
    metadata = reader.metadata
    if metadata is not None:
        text_title = _clean_text(getattr(metadata, "title", None))
        if getattr(metadata, "author", None):
            text_authors = _parse_author_line(str(metadata.author))
    extracted_pages = _extract_pages_text(reader, config)
    doi = _extract_doi(extracted_pages)
    venue = _extract_venue(extracted_pages)
    year = _extract_year(extracted_pages)
    page_title = _extract_title(extracted_pages)
    if page_title:
        text_title = page_title
    page_authors = _extract_authors(extracted_pages)
    if page_authors:
        text_authors = page_authors
    return doi, venue, year, text_title, text_authors


def _extract_pages_text(reader: PdfReader, config: BibConfig) -> str:
    page_count = min(len(reader.pages), max(config.pdf_sync.max_pages, 1))
    return "\n".join(
        (reader.pages[index].extract_text() or "") for index in range(page_count)
    )
