"""Shared models for BibTeX processing and screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ScreeningBucket(str, Enum):
    """Supported screening buckets."""

    KEEP = "keep"
    REVIEW = "review"
    EXCLUDE = "exclude"


@dataclass(slots=True)
class DuplicateInfo:
    """Duplicate information discovered during a run."""

    doi_keys: dict[str, list[str]] = field(default_factory=dict)
    title_keys: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedField:
    """A parsed BibTeX field segment."""

    name: str
    raw_text: str
    value: str


@dataclass(slots=True)
class BibEntry:
    """A BibTeX entry with raw formatting preserved."""

    entry_type: str
    key: str
    fields: dict[str, str]
    field_order: list[str]
    start: int
    end: int
    raw_text: str
    header: str
    field_segments: list[ParsedField]

    def __post_init__(self) -> None:
        self.field_order = list(self.field_order)

    @property
    def doi(self) -> str | None:
        """Return a normalized DOI when present."""

        value = self.fields.get("doi")
        if not value:
            return None
        return value.strip().lower()

    @property
    def title(self) -> str:
        """Return the entry title or an empty string."""

        return self.fields.get("title", "")

    @property
    def year(self) -> int | None:
        """Return the entry year if it parses as an integer."""

        value = self.fields.get("year", "").strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @property
    def venue(self) -> str | None:
        """Return the venue-like field if present."""

        return self.fields.get("journal") or self.fields.get("booktitle")

    @property
    def url(self) -> str | None:
        """Return the URL field when present."""

        return self.fields.get("url")


@dataclass(slots=True)
class EnrichmentData:
    """Normalized metadata returned by a provider."""

    provider: str
    data: dict[str, Any]
    source_url: str | None = None


@dataclass(slots=True)
class ProviderResult:
    """Provider lookup or search result."""

    provider: str
    items: list[EnrichmentData] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class CitationStats:
    """Citation-related signals used for scoring."""

    citation_count: int | None = None
    expected_citations: float | None = None


@dataclass(slots=True)
class ResolvedMetadata:
    """Canonical metadata resolved for a BibTeX entry."""

    doi: str | None = None
    title: str | None = None
    year: int | None = None
    journal: str | None = None
    booktitle: str | None = None
    publisher: str | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    url: str | None = None
    publication_type: str | None = None
    citation_stats: CitationStats = field(default_factory=CitationStats)
    is_retracted: bool = False
    has_serious_update: bool = False
    is_review_or_benchmark: bool = False
    provider_notes: list[str] = field(default_factory=list)

    @property
    def venue(self) -> str | None:
        """Return the canonical venue field."""

        return self.journal or self.booktitle


@dataclass(slots=True)
class MatchCandidate:
    """A possible canonical record for an entry."""

    provider: str
    confidence: float
    metadata: ResolvedMetadata
    reasons: list[str]


@dataclass(slots=True)
class ResolutionResult:
    """Result of identifying and enriching a BibTeX entry."""

    matched: bool
    confidence: float
    metadata: ResolvedMetadata
    reasons: list[str] = field(default_factory=list)
    ambiguous: bool = False
    network_failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScoreBreakdown:
    """Detailed score components for an entry."""

    total_score: float
    bucket: ScreeningBucket
    reasons: list[str]
    details: dict[str, Any]


@dataclass(slots=True)
class ScreenedEntry:
    """A screened entry with resolved metadata and score data."""

    entry: BibEntry
    resolution: ResolutionResult
    score: ScoreBreakdown


@dataclass(slots=True)
class EnrichmentPreview:
    """A preview of canonical field changes for one entry."""

    key: str
    confidence: float
    updates: dict[str, str]
    reasons: list[str]


@dataclass(slots=True)
class ExtractedPdfMetadata:
    """Normalized metadata extracted from a local PDF."""

    pdf_path: Path
    relative_pdf_path: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    entry_type: str | None = None
    file_field: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CachePaths:
    """Resolved cache locations."""

    base_dir: Path
    providers_dir: Path
