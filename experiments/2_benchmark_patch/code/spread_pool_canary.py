"""Read-only feasibility canary for the spread-pool independent-support arm.

For every (dataset, split, class) this calls the real ``designate_patch_pool``
against the real crossed ``required_counts`` -- the same call the freeze makes --
once at the current concentrated floor and once with the patient floor raised to
the full eligible inventory. Writes nothing; prints a JSON summary.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

from imbalance_benchmark.common import load_config, output_root
from imbalance_benchmark.manifest.construction_helpers import (
    assignment_allocations,
    class_construction_seed,
    required_counts_by_class,
)
from imbalance_benchmark.manifest.sampling.patch_pool import designate_patch_pool

CONFIGS = {
    "bracs": "experiments/2_benchmark_patch/configs/bracs_patch.yaml",
    "camelyon16": "experiments/2_benchmark_patch/configs/camelyon16_patch.yaml",
    "panda": "experiments/2_benchmark_patch/configs/panda_patch.yaml",
    "tcga_ut": "experiments/2_benchmark_patch/configs/tcga_ut_patch.yaml",
}


def _pool(df_class, floor, seed, counts, cap):
    return designate_patch_pool(
        df_class,
        floor,
        seed,
        max(counts),
        max_pool_units=max(cap, max(counts)),
        required_counts=tuple(sorted(counts)),
    )


def _class_row(train_df, cls, counts, seed, floor):
    df_class = cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls])
    eligible = int(df_class["case_id"].nunique())
    class_seed = class_construction_seed(seed, cls)
    row: dict[str, Any] = {"class": cls, "eligible_patients": eligible}
    try:
        wide = _pool(df_class, floor, class_seed, counts, floor)
        row["concentrated_patients"] = int(wide["case_id"].nunique())
        row["concentrated_slides"] = int(wide["slide_id"].nunique())
    except ValueError as exc:
        row["concentrated_error"] = str(exc)
        return row
    try:
        spread = _pool(df_class, eligible, class_seed, counts, eligible)
        row["spread_patients"] = int(spread["case_id"].nunique())
        row["spread_slides"] = int(spread["slide_id"].nunique())
        row["nested"] = bool(
            set(wide["case_id"]).issubset(set(spread["case_id"]))
        )
        ratio = row["spread_patients"] / row["concentrated_patients"]
        row["log_shortage"] = math.log(ratio) if ratio > 0 else float("nan")
    except ValueError as exc:
        row["spread_error"] = str(exc)
    return row


def _split(dataset: str, config_path: str, index: int) -> dict[str, Any]:
    config = load_config(config_path)
    # Build the split data path by hand: split_paths() creates directories, and
    # this canary must not touch the outputs tree at all.
    data = output_root(config) / f"split={index}" / "data"
    freeze = json.loads((data / "manifest_freeze.json").read_text())
    df = pd.read_csv(data / "manifest.csv")
    train_df = cast(pd.DataFrame, df[df["split"] == "train"])
    allocations = assignment_allocations(
        train_df,
        freeze["tail_assignments"],
        freeze["shared_T"],
        freeze["min_support"],
    )
    required = required_counts_by_class(allocations)
    seed = freeze["construction_seed"]
    floor = freeze["independent_floor"]
    rows = [
        _class_row(train_df, cls, counts, seed, floor)
        for cls, counts in sorted(required.items())
    ]
    return {"split": index, "independent_floor": floor, "classes": rows}


def main() -> None:
    only = sys.argv[1:] or list(CONFIGS)
    out = []
    for dataset in only:
        splits = []
        for index in range(3):
            try:
                splits.append(_split(dataset, CONFIGS[dataset], index))
            except Exception as exc:  # noqa: BLE001 - canary reports, never fails
                splits.append({"split": index, "error": repr(exc)})
            print(f"done {dataset} split {index}", file=sys.stderr, flush=True)
        out.append({"dataset": dataset, "splits": splits})
    print(json.dumps({"canary": out}, indent=1))


if __name__ == "__main__":
    main()
