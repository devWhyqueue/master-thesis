"""Scoring and duplicate logic for BibTeX screening."""

from __future__ import annotations

from ...config import BibConfig
from ...models import BibEntry, DuplicateInfo, ResolutionResult, ScoreBreakdown
from ..text import normalize_title
from .rules import (
    _assign_bucket,
    _citation_signal,
    _duplicate_keys,
    _format_details,
    _matches_review_like,
    _matches_top_tier_venue,
    _missing_metadata_count,
    _missing_reason,
)


def detect_duplicates(entries: list[BibEntry]) -> DuplicateInfo:
    """Detect duplicates by DOI and normalized title."""

    doi_keys: dict[str, list[str]] = {}
    title_keys: dict[str, list[str]] = {}
    for entry in entries:
        if entry.doi:
            doi_keys.setdefault(entry.doi, []).append(entry.key)
        title = normalize_title(entry.title)
        if title:
            title_keys.setdefault(title, []).append(entry.key)
    return DuplicateInfo(
        doi_keys={k: v for k, v in doi_keys.items() if len(v) > 1},
        title_keys={k: v for k, v in title_keys.items() if len(v) > 1},
    )


def score_entry(
    entry: BibEntry,
    resolution: ResolutionResult,
    duplicates: DuplicateInfo,
    config: BibConfig,
) -> ScoreBreakdown:
    """Score one entry and assign a screening bucket."""

    metadata = resolution.metadata
    details = _base_details(resolution)
    reasons: list[str] = []
    publication_type = (metadata.publication_type or "").casefold()
    missing_metadata_count = _missing_metadata_count(entry, resolution)
    details["missing_metadata_count"] = missing_metadata_count
    score = _aggregate_score(
        entry,
        resolution,
        duplicates,
        config,
        publication_type,
        missing_metadata_count,
        details,
        reasons,
    )
    bucket = _assign_bucket(score, reasons, missing_metadata_count, config)
    if not reasons:
        _append_default_reason(reasons)
    details["bucket_rule"] = bucket.value
    return ScoreBreakdown(
        total_score=score, bucket=bucket, reasons=reasons[:4], details=details
    )


def screening_updates(
    score: ScoreBreakdown, include_details: bool, details_format: str
) -> dict[str, str | None]:
    """Render screening fields as BibTeX field updates."""

    updates = {
        "x_screening_bucket": score.bucket.value,
        "x_screening_score": f"{score.total_score:.2f}",
        "x_screening_reason": "; ".join(score.reasons),
        "x_screening_details": None,
    }
    if include_details:
        updates["x_screening_details"] = _format_details(score.details, details_format)
    return updates


def _append_default_reason(reasons: list[str]) -> None:
    reasons.append(
        "review suggested: limited evidence, no strong positive or negative signals"
    )


def _base_details(resolution: ResolutionResult) -> dict[str, object]:
    metadata = resolution.metadata
    return {
        "doi": metadata.doi,
        "venue": metadata.venue,
        "publication_type": metadata.publication_type,
        "citation_count": metadata.citation_stats.citation_count,
        "expected_citations": metadata.citation_stats.expected_citations,
        "retracted": metadata.is_retracted,
        "serious_update": metadata.has_serious_update,
        "provider_notes": metadata.provider_notes,
        "resolution_confidence": resolution.confidence,
        "matched": resolution.matched,
    }


def _aggregate_score(
    entry: BibEntry,
    resolution: ResolutionResult,
    duplicates: DuplicateInfo,
    config: BibConfig,
    publication_type: str,
    missing_metadata_count: int,
    details: dict[str, object],
    reasons: list[str],
) -> float:
    screening = config.screening
    metadata = resolution.metadata
    score = _risk_signals(metadata, publication_type, screening, reasons)
    venue_is_top_tier = _matches_top_tier_venue(metadata.venue, config)
    details["top_tier_venue"] = venue_is_top_tier
    score += _venue_signal(metadata.venue, venue_is_top_tier, screening, reasons)
    citation_score, citation_reason, citation_ratio = _citation_signal(
        entry, resolution, config
    )
    score += citation_score
    details["citation_ratio"] = citation_ratio
    if citation_reason:
        reasons.append(citation_reason)
    if _is_review_signal(entry, resolution, publication_type, config):
        score += screening.weights.review_or_benchmark
        reasons.append("positive signal: review or benchmark style article")
    duplicate_keys = _duplicate_keys(entry, duplicates)
    if duplicate_keys:
        score += screening.weights.duplicate_penalty
        reasons.append(
            f"review duplicate candidate with {', '.join(sorted(duplicate_keys))}"
        )
        details["duplicate_keys"] = sorted(duplicate_keys)
    if missing_metadata_count >= screening.thresholds.missing_metadata_threshold:
        score += screening.weights.missing_metadata
        reasons.append(_missing_reason(resolution))
    return score


def _risk_signals(
    metadata, publication_type: str, screening, reasons: list[str]
) -> float:
    score = 0.0
    if metadata.is_retracted:
        score += screening.weights.retracted
        reasons.append("suggest exclusion: retraction flag detected")
    if metadata.has_serious_update:
        score += screening.weights.serious_update
        reasons.append("review carefully: update or relation flag detected")
    blocked_types = {
        item.casefold() for item in screening.thresholds.non_target_publication_types
    }
    if publication_type and publication_type in blocked_types:
        score += screening.weights.non_target_publication_type
        reasons.append(
            f"suggest exclusion: publication type '{publication_type}' is likely off-target"
        )
    return score


def _venue_signal(
    venue: str | None, top_tier: bool, screening, reasons: list[str]
) -> float:
    if top_tier:
        reasons.append("positive signal: strong venue standing")
        return screening.weights.strong_venue
    if venue:
        reasons.append("limited venue signal: not in configured top-tier list")
        return screening.weights.poor_venue / 4
    return 0.0


def _is_review_signal(
    entry: BibEntry,
    resolution: ResolutionResult,
    publication_type: str,
    config: BibConfig,
) -> bool:
    metadata = resolution.metadata
    return metadata.is_review_or_benchmark or _matches_review_like(
        entry, publication_type, config
    )
