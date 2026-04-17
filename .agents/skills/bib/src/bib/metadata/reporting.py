"""Console reporting helpers for BibTeX workflows."""

from __future__ import annotations

from collections import Counter

from ..models import DuplicateInfo, EnrichmentPreview, ScreenedEntry
from ..pdf.creation import PdfSyncResult


def render_screening_summary(screened_entries: list[ScreenedEntry]) -> str:
    """Render a compact screening summary."""

    counter = Counter(item.score.bucket.value for item in screened_entries)
    total = len(screened_entries)
    return "processed={} keep={} review={} exclude={}".format(
        total,
        counter.get("keep", 0),
        counter.get("review", 0),
        counter.get("exclude", 0),
    )


def render_enrichment_summary(
    previews: list[EnrichmentPreview], changed_count: int
) -> str:
    """Render a compact enrichment summary."""

    return f"processed={len(previews)} changed={changed_count}"


def render_enrichment_preview(previews: list[EnrichmentPreview]) -> str:
    """Render human-readable dry-run output for enrichment."""

    lines: list[str] = []
    for preview in previews:
        if preview.updates:
            changes = ", ".join(
                f"{field}={value}" for field, value in sorted(preview.updates.items())
            )
            lines.append(
                f"{preview.key}: confidence={preview.confidence:.2f} {changes}"
            )
    return "\n".join(lines)


def render_duplicates(duplicates: DuplicateInfo) -> str:
    """Render duplicate candidates."""

    lines: list[str] = []
    for doi, keys in sorted(duplicates.doi_keys.items()):
        lines.append(f"doi {doi}: {', '.join(keys)}")
    for title, keys in sorted(duplicates.title_keys.items()):
        lines.append(f"title {title}: {', '.join(keys)}")
    return "\n".join(lines) if lines else "no duplicates found"


def render_pdf_sync_summary(results: list[PdfSyncResult]) -> str:
    """Render a compact pdf-sync summary."""

    matched = sum(1 for result in results if result.matched_key)
    created = sum(1 for result in results if result.created_key)
    return f"processed={len(results)} matched={matched} created={created} needs_review={len(results) - matched - created}"


def render_pdf_sync_preview(results: list[PdfSyncResult]) -> str:
    """Render dry-run details for pdf-sync."""

    lines: list[str] = []
    for result in results:
        if result.matched_key:
            lines.append(_matched_line(result))
        elif result.created_key:
            lines.append(_created_line(result))
        else:
            lines.append(_review_line(result))
    return "\n".join(lines)


def _matched_line(result: PdfSyncResult) -> str:
    changes = ", ".join(
        f"{name}={value}"
        for name, value in sorted(result.updates.items())
        if value is not None
    )
    suffix = f" {changes}" if changes else ""
    return f"{result.pdf.relative_pdf_path}: matched={result.matched_key} confidence={result.match_confidence:.2f}{suffix}"


def _created_line(result: PdfSyncResult) -> str:
    return f"{result.pdf.relative_pdf_path}: create={result.created_key} confidence={result.match_confidence:.2f}"


def _review_line(result: PdfSyncResult) -> str:
    payload = _review_payload(result)
    reason = "; ".join(result.reasons) if result.reasons else "no confident match"
    return f"{result.pdf.relative_pdf_path}: needs_review {payload} reason={reason}"


def _review_payload(result: PdfSyncResult) -> str:
    payload: list[str] = []
    if result.pdf.title:
        payload.append(f"title={result.pdf.title}")
    if result.pdf.authors:
        payload.append(f"authors={' | '.join(result.pdf.authors[:4])}")
    if result.pdf.year is not None:
        payload.append(f"year={result.pdf.year}")
    if result.pdf.doi:
        payload.append(f"doi={result.pdf.doi}")
    if result.pdf.venue:
        payload.append(f"venue={result.pdf.venue}")
    return " ".join(payload).strip()
