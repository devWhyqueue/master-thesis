"""PANDA materialization orchestration and public dataset contracts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.datasets.data.panda_grid import audit_slide, canary_rows
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
    """Audit legacy tiles, publish balanced immutable shards and provenance."""
    cfg = materialize_config(config)
    official = load_slide_frame(Path(cfg["raw_root"]))
    if len(official) != LOCKED_SLIDES:
        raise ValueError(f"PANDA official cohort changed: {len(official)} slides")
    scratch = Path(cfg["scratch_root"])
    audited = [
        audit_slide(
            row,
            Path(cfg["legacy_tiles_dir"]) / str(row["slide_id"]),
            scratch,
            float(cfg["jpeg_mae_max"]),
            Path(cfg["legacy_manifest_dir"]) / f"{row['slide_id']}.csv",
        )
        for _, row in official.sort_values("slide_id").iterrows()
    ]
    inventory = pd.concat(audited, ignore_index=True) if audited else pd.DataFrame()
    assert_locked_counts(official, inventory)
    inventory["shard_index"] = balanced_shards(inventory, int(cfg["shard_count"]))
    publish_shards(inventory, scratch, cfg)
    return publish_inventory(official, inventory, cfg)


def audit_canary(config: dict[str, Any]) -> None:
    """Audit providers, grades, mask states and tile-count extremes without publishing."""
    cfg = materialize_config(config)
    scratch = Path(cfg["scratch_root"]) / "canary"
    try:
        for _, row in canary_rows(
            load_slide_frame(Path(cfg["raw_root"])), Path(cfg["legacy_tiles_dir"])
        ).iterrows():
            audit_slide(
                row,
                Path(cfg["legacy_tiles_dir"]) / str(row["slide_id"]),
                scratch,
                float(cfg["jpeg_mae_max"]),
                Path(cfg["legacy_manifest_dir"]) / f"{row['slide_id']}.csv",
            )
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


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
