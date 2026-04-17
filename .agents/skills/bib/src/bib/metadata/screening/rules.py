"""Scoring helper rules for BibTeX screening."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from ...config import BibConfig
from ...models import BibEntry, DuplicateInfo, ResolutionResult, ScreeningBucket
from ..text import normalize_title


def _missing_reason(resolution: ResolutionResult) -> str:
    if resolution.network_failures:
        return "review suggested: provider lookup failed, metadata stayed incomplete"
    if resolution.matched:
        return "review suggested: external match found but metadata remains incomplete"
    return "review suggested: no resolvable external match found"


def _missing_metadata_count(entry: BibEntry, resolution: ResolutionResult) -> int:
    metadata = resolution.metadata
    missing = 0
    if not metadata.doi:
        missing += 1
    if not metadata.publication_type:
        missing += 1
    if metadata.citation_stats.citation_count is None:
        missing += 1
    if not metadata.venue:
        missing += 1
    if entry.title and not metadata.title:
        missing += 1
    return missing


def _citation_signal(
    entry: BibEntry, resolution: ResolutionResult, config: BibConfig
) -> tuple[float, str | None, float | None]:
    stats = resolution.metadata.citation_stats
    if stats.citation_count is None:
        return 0.0, None, None
    expected = stats.expected_citations or _expected_citations(entry.year)
    age = _paper_age(entry.year)
    if age >= config.screening.thresholds.old_paper_years:
        expected = max(expected, config.screening.thresholds.min_expected_citations_old)
    ratio = stats.citation_count / expected if expected else None
    if ratio is not None and ratio >= config.screening.thresholds.strong_citation_ratio:
        return (
            config.screening.weights.good_citation_impact,
            "positive signal: citation impact is strong for paper age",
            ratio,
        )
    if (
        ratio is not None
        and age >= config.screening.thresholds.old_paper_years
        and ratio <= config.screening.thresholds.weak_citation_ratio
    ):
        return (
            config.screening.weights.weak_citation_impact_old,
            "suggest exclusion: older paper shows weak citation traction",
            ratio,
        )
    return 0.0, None, ratio


def _expected_citations(year: int | None) -> float:
    age = _paper_age(year)
    return max(2.0, 2.5 * math.sqrt(max(age, 1)) * age)


def _paper_age(year: int | None) -> int:
    if year is None:
        return 0
    return max(0, datetime.now(UTC).year - year)


def _matches_review_like(
    entry: BibEntry, publication_type: str, config: BibConfig
) -> bool:
    review_types = {
        item.casefold() for item in config.screening.thresholds.review_like_types
    }
    if any(token in publication_type for token in review_types):
        return True
    title = entry.title.casefold()
    return any(token in title for token in review_types)


def _matches_top_tier_venue(venue: str | None, config: BibConfig) -> bool:
    if not venue:
        return False
    normalized = venue.casefold()
    for item in config.screening.thresholds.top_tier_venues:
        candidate = item.casefold()
        if (
            candidate == normalized
            or candidate in normalized
            or normalized in candidate
        ):
            return True
    return False


def _duplicate_keys(entry: BibEntry, duplicates: DuplicateInfo) -> set[str]:
    related: set[str] = set()
    if entry.doi and entry.doi in duplicates.doi_keys:
        related.update(
            key for key in duplicates.doi_keys[entry.doi] if key != entry.key
        )
    normalized = normalize_title(entry.title)
    if normalized and normalized in duplicates.title_keys:
        related.update(
            key for key in duplicates.title_keys[normalized] if key != entry.key
        )
    return related


def _assign_bucket(
    score: float, reasons: list[str], missing_metadata_count: int, config: BibConfig
) -> ScreeningBucket:
    rules = config.screening.bucket_rules
    thresholds = config.screening.thresholds
    if any("retraction" in reason for reason in reasons):
        return ScreeningBucket.EXCLUDE
    if score >= rules.keep_min_score:
        return ScreeningBucket.KEEP
    if score <= rules.exclude_max_score:
        if (
            not rules.exclude_requires_metadata
            or missing_metadata_count < thresholds.missing_metadata_threshold
        ):
            return ScreeningBucket.EXCLUDE
    return ScreeningBucket.REVIEW


def _format_details(details: dict[str, object], details_format: str) -> str:
    parts = [f"{key}={details[key]}" for key in sorted(details)]
    return "; ".join(parts)
