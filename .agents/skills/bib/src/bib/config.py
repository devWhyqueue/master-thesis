"""Configuration loading for BibTeX processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProviderConfig:
    enabled: bool = True
    timeout_seconds: float = 15.0
    user_agent: str = "master-thesis-bib/0.1"


@dataclass(slots=True)
class CacheConfig:
    dir: str = ".agents/skills/bib/.cache"
    ttl_days: int = 30
    enabled: bool = True


@dataclass(slots=True)
class ResolutionConfig:
    min_confidence: float = 0.74
    ambiguity_delta: float = 0.06
    search_rows: int = 5


@dataclass(slots=True)
class EnrichmentConfig:
    overwrite_fields: list[str] = field(
        default_factory=lambda: [
            "doi",
            "url",
            "journal",
            "booktitle",
            "publisher",
            "volume",
            "number",
            "pages",
            "year",
            "title",
        ]
    )
    canonicalize_title: bool = True


@dataclass(slots=True)
class PdfSyncConfig:
    pdf_dir: str = "papers"
    max_pages: int = 2
    filename_parsing: bool = True
    match_min_confidence: float = 0.82
    create_min_confidence: float = 0.72


@dataclass(slots=True)
class ScoringWeights:
    strong_venue: float = 1.5
    good_citation_impact: float = 1.2
    review_or_benchmark: float = 0.8
    weak_citation_impact_old: float = -1.3
    poor_venue: float = -1.2
    non_target_publication_type: float = -1.4
    missing_metadata: float = -0.7
    duplicate_penalty: float = -0.4
    serious_update: float = -2.0
    retracted: float = -4.0


@dataclass(slots=True)
class ThresholdConfig:
    top_tier_venues: list[str] = field(
        default_factory=lambda: [
            "nature medicine",
            "nature reviews bioengineering",
            "nature",
            "science",
            "the lancet",
            "new england journal of medicine",
            "iclr",
            "neurips",
            "icml",
            "cvpr",
            "nature machine intelligence",
        ]
    )
    non_target_publication_types: list[str] = field(
        default_factory=lambda: [
            "editorial",
            "letter",
            "news",
            "correction",
            "addendum",
            "commentary",
        ]
    )
    review_like_types: list[str] = field(
        default_factory=lambda: [
            "review",
            "systematic-review",
            "meta-analysis",
            "benchmark",
        ]
    )
    old_paper_years: int = 5
    min_expected_citations_old: float = 8.0
    strong_citation_ratio: float = 1.5
    weak_citation_ratio: float = 0.4
    missing_metadata_threshold: int = 2


@dataclass(slots=True)
class BucketRuleConfig:
    keep_min_score: float = 1.25
    exclude_max_score: float = -1.75
    exclude_requires_metadata: bool = True


@dataclass(slots=True)
class OutputConfig:
    include_details: bool = True
    details_format: str = "compact"


@dataclass(slots=True)
class ScreeningConfig:
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    bucket_rules: BucketRuleConfig = field(default_factory=BucketRuleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass(slots=True)
class BibConfig:
    crossref: ProviderConfig = field(default_factory=ProviderConfig)
    openalex: ProviderConfig = field(default_factory=ProviderConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    pdf_sync: PdfSyncConfig = field(default_factory=PdfSyncConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dict(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _build_config(data: dict[str, Any]) -> BibConfig:
    providers = data.get("providers", {})
    screening = data.get("screening", {})
    config = BibConfig(
        crossref=ProviderConfig(**providers.get("crossref", {})),
        openalex=ProviderConfig(**providers.get("openalex", {})),
        cache=CacheConfig(**data.get("cache", {})),
        resolution=ResolutionConfig(**data.get("resolution", {})),
        enrichment=EnrichmentConfig(**data.get("enrichment", {})),
        pdf_sync=PdfSyncConfig(**data.get("pdf_sync", {})),
        screening=ScreeningConfig(
            weights=ScoringWeights(**screening.get("weights", {})),
            thresholds=ThresholdConfig(**screening.get("thresholds", {})),
            bucket_rules=BucketRuleConfig(**screening.get("bucket_rules", {})),
            output=OutputConfig(**screening.get("output", {})),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: BibConfig) -> None:
    if config.cache.ttl_days < 0:
        raise ValueError("cache.ttl_days must be non-negative")
    if not 0 < config.resolution.min_confidence <= 1:
        raise ValueError("resolution.min_confidence must be within (0, 1]")
    if not 0 < config.pdf_sync.match_min_confidence <= 1:
        raise ValueError("pdf_sync.match_min_confidence must be within (0, 1]")
    if not 0 < config.pdf_sync.create_min_confidence <= 1:
        raise ValueError("pdf_sync.create_min_confidence must be within (0, 1]")
    if config.resolution.ambiguity_delta < 0:
        raise ValueError("resolution.ambiguity_delta must be non-negative")
    if (
        config.screening.bucket_rules.keep_min_score
        <= config.screening.bucket_rules.exclude_max_score
    ):
        raise ValueError("screening keep threshold must exceed exclude threshold")
    if config.screening.output.details_format not in {"json", "compact"}:
        raise ValueError("screening.output.details_format must be 'json' or 'compact'")


def default_config() -> BibConfig:
    """Return the default configuration."""

    return BibConfig()


def default_config_dict() -> dict[str, Any]:
    """Return the default configuration as a plain dictionary."""

    config = default_config()
    return {
        "providers": {
            "crossref": asdict(config.crossref),
            "openalex": asdict(config.openalex),
        },
        "cache": asdict(config.cache),
        "resolution": asdict(config.resolution),
        "enrichment": asdict(config.enrichment),
        "pdf_sync": asdict(config.pdf_sync),
        "screening": {
            "weights": asdict(config.screening.weights),
            "thresholds": asdict(config.screening.thresholds),
            "bucket_rules": asdict(config.screening.bucket_rules),
            "output": asdict(config.screening.output),
        },
    }


def load_config(path: Path | None) -> BibConfig:
    """Load a configuration file or fall back to defaults."""

    if path is None:
        return default_config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration file must contain a mapping")
    return _build_config(_merge_dict(default_config_dict(), raw))
