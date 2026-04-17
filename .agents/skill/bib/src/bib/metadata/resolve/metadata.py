"""Resolver metadata conversion helpers."""

from __future__ import annotations

from urllib.parse import urlparse

from ...models import BibEntry, CitationStats, ResolvedMetadata
from ...metadata.text import normalize_title


def _metadata_from_payload(
    provider: str, payload: dict[str, object], entry: BibEntry
) -> ResolvedMetadata:
    title = _as_str(payload.get("title")) or entry.title
    venue = _as_str(payload.get("container-title"))
    doi = _as_str(payload.get("doi"))
    publication_type = _as_str(payload.get("type")) or entry.entry_type.casefold()
    is_review = (
        "review" in (title or "").casefold() or "benchmark" in (title or "").casefold()
    )
    if publication_type:
        is_review = is_review or any(
            token in publication_type.casefold()
            for token in ("review", "meta-analysis", "benchmark")
        )
    citation_count = _as_int(payload.get("cited_by_count"))
    raw_percentile = payload.get("citation_normalized_percentile")
    expected_citations = (
        _as_float(raw_percentile.get("value"))
        if isinstance(raw_percentile, dict)
        else None
    )
    journal = (
        venue
        if entry.entry_type.casefold() == "article" or entry.fields.get("journal")
        else None
    )
    booktitle = venue if journal is None else None
    if (
        provider == "crossref"
        and publication_type
        and "proceedings" in publication_type.casefold()
    ):
        journal = None
        booktitle = venue
    if provider == "openalex" and entry.fields.get("booktitle"):
        journal = None
        booktitle = venue
    return ResolvedMetadata(
        doi=doi.lower() if doi else None,
        title=title,
        year=_as_int(payload.get("year")),
        journal=journal,
        booktitle=booktitle,
        publisher=_as_str(payload.get("publisher")),
        volume=_as_str(payload.get("volume")),
        number=_as_str(payload.get("number")),
        pages=_as_str(payload.get("pages")),
        url=_as_str(payload.get("url")),
        publication_type=publication_type,
        citation_stats=CitationStats(
            citation_count=citation_count,
            expected_citations=expected_citations,
        ),
        is_retracted=bool(payload.get("is_retracted")),
        has_serious_update=bool(payload.get("relation"))
        or bool(payload.get("update-to")),
        is_review_or_benchmark=is_review,
    )


def _merge_provider_payloads(
    entry: BibEntry, payloads: dict[str, list[dict[str, object]]]
) -> ResolvedMetadata:
    crossref = payloads.get("crossref", [{}])[0] if payloads.get("crossref") else {}
    openalex = payloads.get("openalex", [{}])[0] if payloads.get("openalex") else {}
    metadata = (
        _metadata_from_payload("crossref", crossref, entry)
        if crossref
        else _metadata_from_payload("openalex", openalex, entry)
    )
    metadata.doi = metadata.doi or (_as_str(openalex.get("doi")) or None)
    metadata.url = metadata.url or _as_str(openalex.get("url")) or entry.url
    metadata.publication_type = (
        metadata.publication_type
        or _as_str(openalex.get("type"))
        or entry.entry_type.casefold()
    )
    metadata.year = metadata.year or _as_int(openalex.get("year")) or entry.year
    metadata.title = metadata.title or _as_str(openalex.get("title")) or entry.title
    metadata.journal = metadata.journal or (
        _as_str(openalex.get("container-title"))
        if entry.fields.get("journal")
        else None
    )
    if not metadata.journal and not metadata.booktitle:
        metadata.booktitle = (
            _as_str(openalex.get("container-title"))
            if entry.fields.get("booktitle") or entry.entry_type == "inproceedings"
            else None
        )
    metadata.publisher = metadata.publisher or _as_str(openalex.get("publisher"))
    metadata.volume = metadata.volume or _as_str(openalex.get("volume"))
    metadata.number = metadata.number or _as_str(openalex.get("number"))
    metadata.pages = metadata.pages or _as_str(openalex.get("pages"))
    metadata.is_retracted = metadata.is_retracted or bool(openalex.get("is_retracted"))
    metadata.citation_stats = CitationStats(
        citation_count=_as_int(openalex.get("cited_by_count")),
        expected_citations=None,
    )
    return metadata


def _metadata_from_entry(entry: BibEntry) -> ResolvedMetadata:
    return ResolvedMetadata(
        doi=entry.doi,
        title=entry.title or None,
        year=entry.year,
        journal=entry.fields.get("journal"),
        booktitle=entry.fields.get("booktitle"),
        publisher=entry.fields.get("publisher"),
        volume=entry.fields.get("volume"),
        number=entry.fields.get("number"),
        pages=entry.fields.get("pages"),
        url=entry.url,
        publication_type=entry.entry_type.casefold(),
    )


def _venue_overlap(left: str, right: str) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    shared = left_tokens & right_tokens
    return len(shared) / max(len(left_tokens), len(right_tokens))


def _url_hint_match(
    entry_url: str | None, candidate_url: str | None, candidate_doi: str | None
) -> bool:
    if entry_url and candidate_url and _domain(entry_url) == _domain(candidate_url):
        return True
    return bool(entry_url and candidate_doi and candidate_doi in entry_url)


def _domain(url: str) -> str:
    return urlparse(url).netloc.casefold()


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, str)):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, str)):
            return float(value)
        return None
    except (TypeError, ValueError):
        return None
