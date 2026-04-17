"""Metadata providers for bib enrichment."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

import httpx

from ..config import ProviderConfig
from ..models import BibEntry, EnrichmentData, ProviderResult


class CrossrefProvider:
    """Fetch and search publication metadata from Crossref."""

    name = "crossref"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    def fetch_by_doi(self, doi: str) -> ProviderResult:
        """Fetch DOI metadata or return a graceful provider result."""

        if not self._config.enabled:
            return ProviderResult(provider=self.name)
        url = f"https://api.crossref.org/works/{doi}"
        try:
            payload = self._get_json(url)
        except httpx.HTTPError as error:
            return ProviderResult(provider=self.name, error=str(error))
        return ProviderResult(
            provider=self.name,
            items=[
                EnrichmentData(
                    provider=self.name,
                    data=_normalize_message(payload.get("message", {})),
                    source_url=url,
                )
            ],
        )

    def search(self, entry: BibEntry, rows: int) -> ProviderResult:
        """Search Crossref using bibliographic metadata."""

        if not self._config.enabled:
            return ProviderResult(provider=self.name)
        query = " ".join(
            part
            for part in [entry.title, entry.venue or "", str(entry.year or "")]
            if part
        ).strip()
        url = f"https://api.crossref.org/works?{urlencode({'query.bibliographic': query, 'rows': rows})}"
        try:
            payload = self._get_json(url)
        except httpx.HTTPError as error:
            return ProviderResult(provider=self.name, error=str(error))
        items = payload.get("message", {}).get("items", [])
        return ProviderResult(
            provider=self.name,
            items=[
                EnrichmentData(
                    provider=self.name, data=_normalize_message(item), source_url=url
                )
                for item in items
            ],
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        headers = {"User-Agent": self._config.user_agent}
        with httpx.Client(
            timeout=self._config.timeout_seconds, headers=headers, follow_redirects=True
        ) as client:
            response = client.get(url)
            response.raise_for_status()
        return response.json()


class OpenAlexProvider:
    """Fetch citation and venue metadata from OpenAlex."""

    name = "openalex"

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config

    def fetch_by_doi(self, doi: str) -> ProviderResult:
        """Fetch a canonical record by DOI."""

        if not self._config.enabled:
            return ProviderResult(provider=self.name)
        encoded = quote(f"https://doi.org/{doi}", safe="")
        url = f"https://api.openalex.org/works/{encoded}"
        try:
            payload = self._get_json(url)
        except httpx.HTTPError as error:
            return ProviderResult(provider=self.name, error=str(error))
        return ProviderResult(
            provider=self.name,
            items=[
                EnrichmentData(
                    provider=self.name, data=_normalize_work(payload), source_url=url
                )
            ],
        )

    def search(self, entry: BibEntry, rows: int) -> ProviderResult:
        """Search OpenAlex using title-oriented matching."""

        if not self._config.enabled:
            return ProviderResult(provider=self.name)
        url = f"https://api.openalex.org/works?{urlencode({'search': entry.title, 'per-page': rows, 'mailto': 'bib-skill@example.invalid'})}"
        try:
            payload = self._get_json(url)
        except httpx.HTTPError as error:
            return ProviderResult(provider=self.name, error=str(error))
        results = payload.get("results", [])
        return ProviderResult(
            provider=self.name,
            items=[
                EnrichmentData(
                    provider=self.name, data=_normalize_work(item), source_url=url
                )
                for item in results
            ],
        )

    def _get_json(self, url: str) -> dict[str, Any]:
        headers = {"User-Agent": self._config.user_agent}
        with httpx.Client(
            timeout=self._config.timeout_seconds, headers=headers, follow_redirects=True
        ) as client:
            response = client.get(url)
            response.raise_for_status()
        return response.json()


def _normalize_message(payload: dict[str, Any]) -> dict[str, Any]:
    published_parts = (
        payload.get("published-print", {}).get("date-parts")
        or payload.get("published-online", {}).get("date-parts")
        or payload.get("published", {}).get("date-parts")
        or []
    )
    year = published_parts[0][0] if published_parts and published_parts[0] else None
    return {
        "doi": payload.get("DOI"),
        "title": (payload.get("title") or [None])[0],
        "container-title": (payload.get("container-title") or [None])[0],
        "type": payload.get("type"),
        "publisher": payload.get("publisher"),
        "volume": payload.get("volume"),
        "number": payload.get("issue"),
        "pages": payload.get("page"),
        "url": payload.get("URL"),
        "year": year,
        "relation": payload.get("relation", {}),
        "update-to": payload.get("update-to", []),
        "subtype": payload.get("subtype"),
    }


def _normalize_work(payload: dict[str, Any]) -> dict[str, Any]:
    primary_location = payload.get("primary_location") or {}
    source = primary_location.get("source") or {}
    ids = payload.get("ids") or {}
    doi = ids.get("doi")
    if isinstance(doi, str) and doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/")
    return {
        "doi": doi,
        "title": payload.get("display_name"),
        "container-title": source.get("display_name"),
        "type": payload.get("type"),
        "publisher": source.get("host_organization_name"),
        "volume": payload.get("biblio", {}).get("volume"),
        "number": payload.get("biblio", {}).get("issue"),
        "pages": _pages(payload.get("biblio", {})),
        "url": primary_location.get("landing_page_url")
        or primary_location.get("pdf_url"),
        "year": payload.get("publication_year"),
        "is_retracted": payload.get("is_retracted", False),
        "cited_by_count": payload.get("cited_by_count"),
        "citation_normalized_percentile": payload.get("citation_normalized_percentile"),
        "primary_location": primary_location,
        "host_venue": source,
        "ids": ids,
    }


def _pages(biblio: dict[str, Any]) -> str | None:
    first = biblio.get("first_page")
    last = biblio.get("last_page")
    if first and last:
        return f"{first}--{last}"
    return first or last
