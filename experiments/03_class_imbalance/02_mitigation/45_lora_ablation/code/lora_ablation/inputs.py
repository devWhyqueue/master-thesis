"""Input gates of the analysis: audited inventory, unchanged adapted records, complete frozen fits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    compute_sha256,
    ensure_dirs,
    output_root,
    split_paths,
)

from analyze.paths import run_dir as shard_run_dir
from analyze.paths import shard_keys

from centre import N_SPLITS

from mitigation.grid import stage1_jobs, stage2_jobs

from lora_ablation import ARM, config_digest, source_root
from lora_ablation.fit import HEAD_NAME
from lora_ablation.preflight import INVENTORY_NAME
from lora_ablation.resume import AUDIT_SIDECAR

__all__ = ["frozen_paths", "adapted_paths", "check_inventory", "summarize_resources"]


def frozen_paths(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Per-split result roots of the new (frozen) runs."""
    return {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}


def adapted_paths(config: dict[str, Any]) -> dict[int, dict[str, Path]]:
    """Per-split result roots of the adapted source runs (read-only, never created)."""
    return {
        s: {"results": source_root(config) / f"split={s}" / "results"}
        for s in range(N_SPLITS)
    }


def check_inventory(config: dict[str, Any]) -> dict[str, Any]:
    """Reject analysis unless every adapted record and frozen candidate matches the audit."""
    path = output_root(config) / INVENTORY_NAME
    if not path.is_file():
        raise ValueError(f"inventory missing at {path}")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if inventory["config_sha256"] != config_digest(config):
        raise ValueError("inventory differs from the current configuration")
    for record in inventory["source_records"]:
        source = source_root(config) / record["path"]
        if not source.is_file() or compute_sha256(source) != record["sha256"]:
            raise ValueError(f"inventory source record changed or missing: {source}")
    _check_frozen(config, inventory["fingerprint"])
    return inventory


def _ce_head_sha(paths: dict[str, Path], draw: int) -> str:
    """Hash of one draw's stored CE head; missing weights invalidate every post-hoc record."""
    head = shard_run_dir(paths, ARM, "ce", None, draw) / HEAD_NAME
    if not head.is_file():
        raise ValueError(f"CE head weights missing: {head}")
    return hashlib.sha256(head.read_bytes()).hexdigest()


def _check_frozen(config: dict[str, Any], fingerprint: str) -> None:
    paths = frozen_paths(config)
    jobs = [*stage1_jobs(config), *stage2_jobs(config)]
    for s, d in shard_keys():
        head_sha = _ce_head_sha(paths[s], d)
        for arm, method, param in jobs:
            out = shard_run_dir(paths[s], arm, method, param, d)
            if (
                not (out / RUN_RECORD_NAME).is_file()
                or not (out / AUDIT_SIDECAR).is_file()
            ):
                raise ValueError(f"missing frozen candidate: {out}")
            stored = json.loads((out / AUDIT_SIDECAR).read_text(encoding="utf-8"))
            if stored["audit_fingerprint"] != fingerprint:
                raise ValueError(
                    f"frozen candidate does not match the inventory: {out}"
                )
            if method == "posthoc_la" and stored.get("ce_head_sha256") != head_sha:
                raise ValueError(
                    f"post-hoc record does not match its CE head weights: {out}"
                )


def summarize_resources(
    config: dict[str, Any],
) -> dict[str, dict[str, float | int | None]]:
    """Per method: fits, total/mean seconds, and the largest host/GPU peak memory (MB)."""
    paths = frozen_paths(config)
    gathered: dict[str, list[dict[str, Any]]] = {}
    for _, method, param in (*stage1_jobs(config), *stage2_jobs(config)):
        for s, d in shard_keys():
            sidecar = shard_run_dir(paths[s], ARM, method, param, d) / AUDIT_SIDECAR
            gathered.setdefault(method, []).append(
                json.loads(sidecar.read_text())["resources"]
            )
    return {m: _summary(rows) for m, rows in gathered.items()}


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    seconds = sum(r["seconds"] for r in rows)

    def peak(key: str) -> float | None:
        """Largest recorded value of one memory field, if any fit recorded it."""
        values = [r[key] for r in rows if r.get(key) is not None]
        return max(values) if values else None

    return {
        "fits": len(rows),
        "seconds_total": seconds,
        "seconds_mean": seconds / len(rows),
        "peak_rss_mb": peak("peak_rss_mb"),
        "peak_gpu_mb": peak("peak_gpu_mb"),
    }
