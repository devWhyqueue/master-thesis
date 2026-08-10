"""PANDA materialization orchestration and public dataset contracts.

The cluster-scale pipeline is a 4-stage chain, each stage independently
resumable through signed partials (see the ``hydra-cluster`` skill for how
the stages are submitted as a SLURM dependency chain):

1. ``audit_shard``  - audit ``slides[i::audit_shard_count]`` and sha256 their
   raw TIFFs; write one signed partial pair per shard index.
2. ``combine``      - concatenate every audit partial, gate on locked cohort
   counts, assign the 48 publish shards, write the signed audit inventory.
3. ``pack_shard_stage`` - copy one publish shard's tiles and pack its SqFS.
4. ``publish``      - verify all 48 packed shards and write the signed
   canonical inventory and materialization sidecar.

``materialize()`` runs the same four stages in-process (``audit_shard_count``
forced to 1) for local runs, smoke tests, and the end-to-end test.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import compute_sha256
from imbalance_benchmark.datasets.data.panda.audit_pool import AuditJob, run_audit_jobs
from imbalance_benchmark.datasets.data.panda.partials import (
    audit_partial_done,
    read_audit_partial,
    read_combined_inventory,
    read_raw_hashes,
    write_audit_partial,
    write_combined_inventory,
)
from imbalance_benchmark.datasets.data.panda.publish import (
    LOCKED_SLIDES,
    assert_locked_counts,
    balanced_shards,
    materialize_config,
    pack_one_shard,
    publish_inventory,
)
from imbalance_benchmark.datasets.data.panda.slide_audit import audit_slide, canary_rows
from imbalance_benchmark.datasets.features.panda_inventory import (
    load_materialized_inventory,
    reduce_feature_inventory,
    verify_feature_inventory,
)
from imbalance_benchmark.datasets.panda import load_slide_frame

logger = logging.getLogger(__name__)

__all__ = [
    "audit_canary",
    "audit_shard",
    "audit_slide",
    "combine",
    "load_materialized_inventory",
    "materialize",
    "pack_shard_stage",
    "publish",
    "reduce_feature_inventory",
    "select_physical_shard",
    "verify_feature_inventory",
]


def materialize(config: dict[str, Any]) -> dict[str, Any]:
    """Run all 4 stages in-process: one audit shard, then combine/pack/publish."""
    cfg = materialize_config(config)
    audit_shard(config, 0, shard_count=1)
    combine(config, shard_count=1)
    for index in range(int(cfg["shard_count"])):
        pack_shard_stage(config, index)
    return publish(config)


def audit_shard(
    config: dict[str, Any], shard_index: int, shard_count: int | None = None
) -> None:
    """Audit one shard of slides and sha256 their raw sources into a signed partial."""
    cfg = materialize_config(config)
    count = shard_count if shard_count is not None else int(cfg["audit_shard_count"])
    if audit_partial_done(cfg, shard_index):
        logger.info("PANDA audit shard %d already verified; skipping", shard_index)
        return
    official = _load_official(cfg)
    slides = official.sort_values("slide_id").reset_index(drop=True)
    shard = slides.iloc[shard_index::count]
    audited, crashed = _audit_slides(shard, cfg)
    frame = pd.concat(audited, ignore_index=True) if audited else pd.DataFrame()
    raw_hashes = {
        str(row.slide_id): {
            "image": compute_sha256(Path(str(row.image_path))),
            "mask": compute_sha256(Path(str(row.mask_path))) if row.has_mask else None,
        }
        for _, row in shard.iterrows()
    }
    write_audit_partial(cfg, shard_index, frame, raw_hashes, crashed)


def combine(config: dict[str, Any], shard_count: int | None = None) -> None:
    """Concatenate every audit partial, gate on locked counts, assign publish shards."""
    cfg = materialize_config(config)
    count = shard_count if shard_count is not None else int(cfg["audit_shard_count"])
    official = _load_official(cfg)
    frames: list[pd.DataFrame] = []
    raw_hashes: dict[str, dict[str, str | None]] = {}
    crashed: list[str] = []
    for index in range(count):
        frame, raw, shard_crashed = read_audit_partial(cfg, index)
        frames.append(frame)
        raw_hashes.update(raw)
        crashed.extend(shard_crashed)
    if crashed:
        raise ValueError(
            f"PANDA audit worker crashed on {len(crashed)} slide(s), excluded from "
            f"their shard's partial: {sorted(crashed)}"
        )
    inventory = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    assert_locked_counts(official, inventory)
    inventory = inventory.copy()
    inventory["shard_index"] = balanced_shards(inventory, int(cfg["shard_count"]))
    write_combined_inventory(cfg, inventory, raw_hashes)


def pack_shard_stage(config: dict[str, Any], shard_index: int) -> None:
    """Copy one publish shard's verified tiles from the audit inventory and pack it."""
    cfg = materialize_config(config)
    inventory = read_combined_inventory(cfg)
    shard = cast(
        pd.DataFrame,
        inventory[inventory.shard_index == shard_index].reset_index(drop=True),
    )
    pack_one_shard(shard, shard_index, Path(cfg["scratch_root"]), cfg)


def publish(config: dict[str, Any]) -> dict[str, Any]:
    """Verify all 48 packed shards and publish the signed canonical inventory."""
    cfg = materialize_config(config)
    official = _load_official(cfg)
    inventory = read_combined_inventory(cfg)
    assert_locked_counts(official, inventory)
    raw_hashes = read_raw_hashes(cfg)
    return publish_inventory(official, inventory, cfg, raw_hashes)


def _load_official(cfg: dict[str, Any]) -> pd.DataFrame:
    official = load_slide_frame(Path(cfg["raw_root"]))
    if len(official) != LOCKED_SLIDES:
        raise ValueError(f"PANDA official cohort changed: {len(official)} slides")
    return official


def _audit_slides(
    official: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[list[pd.DataFrame], list[str]]:
    slides = official.sort_values("slide_id").reset_index(drop=True)
    if slides.empty:
        return [], []
    workers = min(_worker_count(cfg), len(slides))
    jobs: list[AuditJob] = [
        (position, len(slides), row, cfg)
        for position, (_, row) in enumerate(slides.iterrows(), start=1)
    ]
    return run_audit_jobs(jobs, workers, _audit_slide_job)


def _worker_count(cfg: dict[str, Any] | None = None) -> int:
    """Use the CPUs allocated to the materialization job, unless overridden."""
    configured = cfg.get("audit_workers") if cfg else None
    if configured:
        return max(1, int(configured))
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, int(cpus)) if cpus else os.cpu_count() or 1


def _audit_slide_job(job: AuditJob) -> pd.DataFrame:
    position, total, row, cfg = job
    logger.info("PANDA auditing slide %d/%d: %s", position, total, row.slide_id)
    return audit_slide(
        row,
        Path(cfg["legacy_tiles_dir"]) / str(row["slide_id"]),
        float(cfg["jpeg_mae_max"]),
        Path(cfg["legacy_manifest_dir"]) / f"{row['slide_id']}.csv",
        int(cfg.get("audit_io_workers", 1)),
        int(cfg.get("audit_band_rows", 512)),
    )


def audit_canary(config: dict[str, Any]) -> None:
    """Audit providers, grades, mask states and tile-count extremes without publishing."""
    cfg = materialize_config(config)
    rows = canary_rows(
        load_slide_frame(Path(cfg["raw_root"])), Path(cfg["legacy_tiles_dir"])
    )
    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        started = time.monotonic()
        audit_slide(
            row,
            Path(cfg["legacy_tiles_dir"]) / str(row["slide_id"]),
            float(cfg["jpeg_mae_max"]),
            Path(cfg["legacy_manifest_dir"]) / f"{row['slide_id']}.csv",
            int(cfg.get("audit_io_workers", 1)),
            int(cfg.get("audit_band_rows", 512)),
        )
        logger.info(
            "PANDA canary audited slide %d/%d: %s (%.1fs)",
            position,
            len(rows),
            row.slide_id,
            time.monotonic() - started,
        )


def select_physical_shard(
    frame: pd.DataFrame, dataset: dict[str, object], index: int, count: int
) -> pd.DataFrame:
    """Choose PANDA's staged physical shard; other datasets retain slide sharding."""
    if dataset.get("name") == "panda":
        if count != 48:
            raise ValueError("PANDA extraction requires the 48 materialized shards")
        return cast(
            pd.DataFrame, frame[frame["shard_index"].eq(index)].reset_index(drop=True)
        )
    slides = sorted(frame["slide_id"].astype(str).unique())
    return cast(
        pd.DataFrame,
        frame[frame["slide_id"].astype(str).isin(slides[index::count])].reset_index(
            drop=True
        ),
    )
