"""Audit-bound completion markers shared by the CPU and CUDA fitters.

A run is complete only when its standard record and its sidecar exist and the sidecar carries the
current inventory and configuration fingerprints. The sidecar is written last, so a killed run is
refit instead of skipped.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import RUN_RECORD_NAME, output_root

from lora_ablation import config_digest, init_seed
from lora_ablation.preflight import INVENTORY_NAME

__all__ = [
    "AUDIT_SIDECAR",
    "require_inventory",
    "expected_sidecar",
    "is_complete",
    "write_sidecar",
    "measure",
]

AUDIT_SIDECAR = "audit.json"
_FINGERPRINTS = ("audit_fingerprint", "config_sha256")


def require_inventory(config: dict[str, Any]) -> str:
    """The audited inventory's fingerprint; fitting needs an inventory of this exact configuration."""
    path = output_root(config) / INVENTORY_NAME
    if not path.is_file():
        raise ValueError(f"inventory missing at {path}; run preflight first")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["config_sha256"] != config_digest(config):
        raise ValueError(
            "inventory differs from the current configuration; rerun preflight"
        )
    return str(stored["fingerprint"])


def expected_sidecar(
    config: dict[str, Any], fingerprint: str, split_idx: int, draw_idx: int
) -> dict[str, Any]:
    """The identity fields a reusable run's sidecar must carry."""
    base = int(config["protocol"]["init_seed_base"])
    return {
        "audit_fingerprint": fingerprint,
        "config_sha256": config_digest(config),
        "init_seed": init_seed(split_idx, draw_idx, base),
    }


def is_complete(out_dir: Path, expected: dict[str, Any]) -> bool:
    """Reusable only with record, sidecar, and identical fingerprints; stale input refits, stale audit raises."""
    sidecar = out_dir / AUDIT_SIDECAR
    if not (sidecar.is_file() and (out_dir / RUN_RECORD_NAME).is_file()):
        return False
    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    if any(stored.get(k) != expected[k] for k in _FINGERPRINTS):
        raise ValueError(
            f"fingerprint of completed run differs from the audit: {out_dir}"
        )
    return all(stored.get(k) == v for k, v in expected.items())


def write_sidecar(
    out_dir: Path, expected: dict[str, Any], extra: dict[str, Any]
) -> None:
    """Mark a finished run: fingerprints plus execution settings and measured resources."""
    (out_dir / AUDIT_SIDECAR).write_text(
        json.dumps({**expected, **extra}, indent=2, default=str), encoding="utf-8"
    )


def _peak_rss_mb() -> float | None:
    """Peak host memory (MB) on POSIX; ``None`` where the ``resource`` module does not exist."""
    if importlib.util.find_spec("resource") is None:
        return None
    resource = importlib.import_module("resource")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def measure() -> Iterator[dict[str, Any]]:
    """Wall time and peak host memory of the enclosed block, filled in on exit."""
    result: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        result["peak_rss_mb"] = _peak_rss_mb()
