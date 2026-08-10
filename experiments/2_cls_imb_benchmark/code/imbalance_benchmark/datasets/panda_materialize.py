"""PANDA materialization orchestration and public dataset contracts."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.datasets.data.panda_grid import (
    audit_slide,
    canary_rows,
    copy_audited_tiles,
)
from imbalance_benchmark.datasets.data.panda_publish import (
    LOCKED_SLIDES,
    assert_locked_counts,
    balanced_shards,
    materialize_config,
    publish_inventory,
    publish_shards,
)
from imbalance_benchmark.datasets.features.panda_inventory import (
    load_materialized_inventory,
    reduce_feature_inventory,
    verify_feature_inventory,
)
from imbalance_benchmark.datasets.panda import load_slide_frame

logger = logging.getLogger(__name__)

__all__ = [
    "audit_canary",
    "audit_slide",
    "load_materialized_inventory",
    "materialize",
    "reduce_feature_inventory",
    "select_physical_shard",
    "verify_feature_inventory",
]


def materialize(config: dict[str, Any]) -> dict[str, Any]:
    """Audit candidate coordinates, gate global counts, then publish shards."""
    cfg = materialize_config(config)
    official = load_slide_frame(Path(cfg["raw_root"]))
    if len(official) != LOCKED_SLIDES:
        raise ValueError(f"PANDA official cohort changed: {len(official)} slides")
    scratch = Path(cfg["scratch_root"])
    audited = _audit_slides(official, cfg)
    inventory = pd.concat(audited, ignore_index=True) if audited else pd.DataFrame()
    assert_locked_counts(official, inventory)
    inventory = copy_audited_tiles(inventory, scratch)
    inventory["shard_index"] = balanced_shards(inventory, int(cfg["shard_count"]))
    publish_shards(inventory, scratch, cfg)
    return publish_inventory(official, inventory, cfg)


def _audit_slides(official: pd.DataFrame, cfg: dict[str, Any]) -> list[pd.DataFrame]:
    slides = official.sort_values("slide_id").reset_index(drop=True)
    workers = min(int(cfg.get("audit_workers", _worker_count())), len(slides))
    jobs = [
        (position, len(slides), row, cfg)
        for position, (_, row) in enumerate(slides.iterrows(), start=1)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_audit_slide_job, jobs))


def _worker_count() -> int:
    """Use the CPUs allocated to the materialization job."""
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, int(cpus)) if cpus else os.cpu_count() or 1


def _audit_slide_job(
    job: tuple[int, int, pd.Series, dict[str, Any]],
) -> pd.DataFrame:
    position, total, row, cfg = job
    logger.info("PANDA auditing slide %d/%d: %s", position, total, row.slide_id)
    return audit_slide(
        row,
        Path(cfg["legacy_tiles_dir"]) / str(row["slide_id"]),
        float(cfg["jpeg_mae_max"]),
        Path(cfg["legacy_manifest_dir"]) / f"{row['slide_id']}.csv",
    )


def audit_canary(config: dict[str, Any]) -> None:
    """Audit providers, grades, mask states and tile-count extremes without publishing."""
    cfg = materialize_config(config)
    for _, row in canary_rows(
        load_slide_frame(Path(cfg["raw_root"])), Path(cfg["legacy_tiles_dir"])
    ).iterrows():
        audit_slide(
            row,
            Path(cfg["legacy_tiles_dir"]) / str(row["slide_id"]),
            float(cfg["jpeg_mae_max"]),
            Path(cfg["legacy_manifest_dir"]) / f"{row['slide_id']}.csv",
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
