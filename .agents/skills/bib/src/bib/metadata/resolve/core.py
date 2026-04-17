"""Resolver and enrichment pipeline for BibTeX entries."""

from __future__ import annotations

import difflib

from ...config import BibConfig
from ...metadata.cache import ResponseCache
from ...models import (
    BibEntry,
    EnrichmentData,
    MatchCandidate,
    ProviderResult,
    ResolutionResult,
)
from ..providers import CrossrefProvider, OpenAlexProvider
from ..text import normalize_title
from .metadata import (
    _metadata_from_entry,
    _metadata_from_payload,
    _merge_provider_payloads,
    _url_hint_match,
    _venue_overlap,
)


def resolve_entry(
    entry: BibEntry,
    providers: list[CrossrefProvider | OpenAlexProvider],
    cache: ResponseCache,
    config: BibConfig,
) -> ResolutionResult:
    """Resolve and enrich metadata for one BibTeX entry."""

    network_failures: list[str] = []
    by_doi = _resolve_by_doi(entry, providers, cache, network_failures)
    if by_doi is not None:
        return by_doi
    candidates = _collect_candidates(entry, providers, cache, config, network_failures)
    if not candidates:
        return _no_match_result(entry, network_failures)
    best, ambiguous = _choose_best(candidates, config)
    if best.confidence < config.resolution.min_confidence or ambiguous:
        return _low_confidence_result(entry, best, ambiguous, network_failures)
    metadata = _finalize_metadata(entry, best, providers, cache, network_failures)
    metadata.provider_notes.append(f"resolved by {best.provider} search")
    return ResolutionResult(
        True, best.confidence, metadata, best.reasons, False, network_failures
    )


def score_candidate(entry: BibEntry, item: EnrichmentData) -> MatchCandidate | None:
    """Score one provider candidate against a BibTeX entry."""

    metadata = _metadata_from_payload(item.provider, item.data, entry)
    if not metadata.title:
        return None
    entry_title = normalize_title(entry.title)
    candidate_title = normalize_title(metadata.title)
    title_ratio = difflib.SequenceMatcher(None, entry_title, candidate_title).ratio()
    score, reasons = _candidate_score(entry, metadata, title_ratio)
    return MatchCandidate(item.provider, min(1.0, score), metadata, reasons)


def _resolve_by_doi(
    entry: BibEntry,
    providers: list[CrossrefProvider | OpenAlexProvider],
    cache: ResponseCache,
    network_failures: list[str],
) -> ResolutionResult | None:
    if not entry.doi:
        return None
    provider_payloads: dict[str, list[dict[str, object]]] = {}
    for provider in providers:
        result = _fetch_by_doi(provider, entry.doi, cache)
        if result.error:
            network_failures.append(f"{provider.name}: {result.error}")
        if result.items:
            provider_payloads[provider.name] = [dict(result.items[0].data)]
    if not provider_payloads:
        return None
    metadata = _merge_provider_payloads(entry, provider_payloads)
    metadata.provider_notes.append("resolved by DOI")
    return ResolutionResult(
        True, 1.0, metadata, ["exact DOI lookup"], False, network_failures
    )


def _collect_candidates(
    entry: BibEntry,
    providers: list[CrossrefProvider | OpenAlexProvider],
    cache: ResponseCache,
    config: BibConfig,
    network_failures: list[str],
) -> list[MatchCandidate]:
    candidates: list[MatchCandidate] = []
    for provider in providers:
        result = _search_provider(provider, entry, cache, config)
        if result.error:
            network_failures.append(f"{provider.name}: {result.error}")
        for item in result.items:
            candidate = score_candidate(entry, item)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _no_match_result(entry: BibEntry, network_failures: list[str]) -> ResolutionResult:
    metadata = _metadata_from_entry(entry)
    metadata.provider_notes.append(
        "provider lookup failed"
        if network_failures
        else "no resolvable external match found"
    )
    return ResolutionResult(
        False,
        0.0,
        metadata,
        ["no resolvable external match found"],
        False,
        network_failures,
    )


def _choose_best(
    candidates: list[MatchCandidate], config: BibConfig
) -> tuple[MatchCandidate, bool]:
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    best = candidates[0]
    ambiguous = (
        len(candidates) > 1
        and best.confidence - candidates[1].confidence
        < config.resolution.ambiguity_delta
    )
    return best, ambiguous


def _low_confidence_result(
    entry: BibEntry, best: MatchCandidate, ambiguous: bool, network_failures: list[str]
) -> ResolutionResult:
    metadata = _metadata_from_entry(entry)
    metadata.provider_notes.append("match candidate below confidence threshold")
    return ResolutionResult(
        False, best.confidence, metadata, best.reasons, ambiguous, network_failures
    )


def _finalize_metadata(
    entry: BibEntry,
    best: MatchCandidate,
    providers: list[CrossrefProvider | OpenAlexProvider],
    cache: ResponseCache,
    network_failures: list[str],
):
    if not best.metadata.doi:
        return best.metadata
    follow_up_payloads: dict[str, list[dict[str, object]]] = {}
    for provider in providers:
        result = _fetch_by_doi(provider, best.metadata.doi, cache)
        if result.error:
            network_failures.append(f"{provider.name}: {result.error}")
        if result.items:
            follow_up_payloads[provider.name] = [dict(result.items[0].data)]
    return (
        _merge_provider_payloads(entry, follow_up_payloads)
        if follow_up_payloads
        else best.metadata
    )


def _candidate_score(
    entry: BibEntry, metadata, title_ratio: float
) -> tuple[float, list[str]]:
    score = 0.70 * title_ratio
    reasons = [f"title similarity={title_ratio:.2f}"]
    if entry.year is not None and metadata.year is not None:
        if entry.year == metadata.year:
            score += 0.15
            reasons.append("year exact")
        elif abs(entry.year - metadata.year) == 1:
            score += 0.05
            reasons.append("year near")
    venue_score = _venue_overlap(entry.venue or "", metadata.venue or "")
    if venue_score:
        score += 0.10 * venue_score
        reasons.append(f"venue overlap={venue_score:.2f}")
    if _url_hint_match(entry.url, metadata.url, metadata.doi):
        score += 0.10
        reasons.append("url hint")
    if metadata.doi:
        score += 0.05
        reasons.append("doi present")
    return score, reasons


def _fetch_by_doi(
    provider: CrossrefProvider | OpenAlexProvider, doi: str, cache: ResponseCache
) -> ProviderResult:
    cached = cache.get(provider.name, "doi", doi)
    if cached is not None:
        return ProviderResult(provider=provider.name, items=[cached])
    result = provider.fetch_by_doi(doi)
    if result.items:
        cache.put(provider.name, "doi", doi, result.items[0])
    return result


def _search_provider(
    provider: CrossrefProvider | OpenAlexProvider,
    entry: BibEntry,
    cache: ResponseCache,
    config: BibConfig,
) -> ProviderResult:
    query_key = "|".join(
        [
            normalize_title(entry.title),
            str(entry.year or ""),
            normalize_title(entry.venue or ""),
            (entry.url or "").strip().casefold(),
        ]
    )
    cached = cache.get(provider.name, "search", query_key)
    if cached is not None:
        items = [
            EnrichmentData(provider=provider.name, data=item)
            for item in cached.data.get("items", [])
        ]
        return ProviderResult(provider=provider.name, items=items)
    result = provider.search(entry, config.resolution.search_rows)
    if result.items:
        cache.put(
            provider.name,
            "search",
            query_key,
            EnrichmentData(
                provider=provider.name,
                data={"items": [item.data for item in result.items]},
            ),
        )
    return result
