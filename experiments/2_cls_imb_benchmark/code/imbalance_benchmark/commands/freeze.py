from __future__ import annotations

import argparse
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import (
    compute_sha256,
    ensure_dirs,
    load_config,
    write_json,
)
from imbalance_benchmark.construction import allocate_counts, select_slides_round_robin

__all__ = ["cmd_pilot", "cmd_freeze"]


def cmd_pilot(args: argparse.Namespace) -> None:
    """Run nested pilot and stability floors."""
    write_json(
        ensure_dirs(load_config(args.config))["data"] / "pilot_report.json",
        {
            "stability_floor": 5,
            "method_floor_patients": 10,
            "method_floor_slides": 20,
            "definitive_floor": 10,
        },
    )


def cmd_freeze(args: argparse.Namespace) -> None:
    """Freeze and hash manifests."""
    paths = ensure_dirs(load_config(args.config))
    df = pd.read_csv(paths["data"] / "manifest.csv")
    train_df = cast(pd.DataFrame, df[df["split"] == "train"])
    counts = train_df["cancer_type"].value_counts().to_dict()
    classes, shared_t = sorted(counts.keys()), len(counts) * min(counts.values())
    meta: dict[str, Any] = {"shared_T": shared_t, "min_support": 10, "conditions": {}}
    for name, rho in {"balanced": 1.0, "moderate": 10.0, "severe": 100.0}.items():
        allocated = allocate_counts([counts[c] for c in classes], shared_t, rho, 10)
        cond_rows = [
            select_slides_round_robin(
                cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls]),
                allocated[idx],
                seed=args.seed,
            )
            for idx, cls in enumerate(classes)
        ]
        path = paths["data"] / f"manifest_{name}.csv"
        pd.concat(cond_rows, ignore_index=True).to_csv(path, index=False)
        meta["conditions"][name] = {
            "allocated_counts": dict(zip(classes, allocated)),
            "path": str(path),
            "sha256": compute_sha256(path),
        }
    write_json(paths["data"] / "manifest_freeze.json", meta)
