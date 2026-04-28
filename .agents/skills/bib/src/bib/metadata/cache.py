"""Local cache for provider responses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import CacheConfig
from ..models import CachePaths, EnrichmentData


def resolve_cache_paths(repo_root: Path, config: CacheConfig) -> CachePaths:
    """Resolve the cache directories for the current repo."""

    base = (repo_root / config.dir).resolve()
    providers = base / "providers"
    providers.mkdir(parents=True, exist_ok=True)
    return CachePaths(base_dir=base, providers_dir=providers)


class ResponseCache:
    """Simple JSON file cache with TTL support."""

    def __init__(self, cache_paths: CachePaths, config: CacheConfig) -> None:
        self._paths = cache_paths
        self._config = config

    # Most filesystems cap filenames at 255 bytes.  We stay well under that
    # limit (200 bytes) so that the prefix "{provider}-{namespace}-" and the
    # ".json" suffix leave enough room.  When the safe key alone would push the
    # filename over that budget we fall back to a short SHA-256 digest instead.
    _MAX_SAFE_FILENAME_BYTES: int = 200
    _HASH_PREFIX_LENGTH: int = 16

    def _cache_path(self, provider: str, namespace: str, key: str) -> Path:
        safe_key = (
            key.replace("/", "__")
            .replace(":", "_")
            .replace("?", "_")
            .replace("&", "_")
            .replace("=", "_")
        )
        filename = f"{provider}-{namespace}-{safe_key}.json"
        if len(filename.encode()) > self._MAX_SAFE_FILENAME_BYTES:
            digest = hashlib.sha256(key.encode()).hexdigest()[: self._HASH_PREFIX_LENGTH]
            filename = f"{provider}-{namespace}-{digest}.json"
        return self._paths.providers_dir / filename

    def get(self, provider: str, namespace: str, key: str) -> EnrichmentData | None:
        """Return a cached response when fresh enough."""

        if not self._config.enabled:
            return None
        path = self._cache_path(provider, namespace, key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if self._config.ttl_days and datetime.now(UTC) - fetched_at > timedelta(
            days=self._config.ttl_days
        ):
            return None
        return EnrichmentData(
            provider=payload["provider"],
            data=payload["data"],
            source_url=payload.get("source_url"),
        )

    def put(
        self, provider: str, namespace: str, key: str, item: EnrichmentData
    ) -> None:
        """Persist an enrichment response."""

        if not self._config.enabled:
            return
        path = self._cache_path(provider, namespace, key)
        payload: dict[str, Any] = asdict(item)
        payload["fetched_at"] = datetime.now(UTC).isoformat()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
