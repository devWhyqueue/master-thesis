from __future__ import annotations

import argparse
from typing import Any, cast

import pandas as pd
import torch

from imbalance_benchmark.common import (
    bag_dataset_kwargs,
    ensure_dirs,
    load_config,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.construction import patient_equals_slide
from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
)
from imbalance_benchmark.datasets.data import load_training_dataset
from imbalance_benchmark.manifest.pilot import (
    frozen_pilot_quota,
    meets_method_floor,
    method_floor,
    pilot_levels_for,
    run_pilot_seed,
    stability_floor_from_curve,
)
from imbalance_benchmark.manifest.seeds import derive_seed

__all__ = ["cmd_pilot"]


def _pilot_setup(
    paths: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], bool, bool, list[int], dict[str, dict[str, int]]]:
    """Load the training manifest and derive the regime, unit type, and candidate levels."""
    df = pd.read_csv(paths["data"] / "manifest.csv")
    train_df = cast(pd.DataFrame, df[df["split"] == "train"])
    classes = sorted(train_df["cancer_type"].unique())
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    eq_slide = patient_equals_slide(train_df)
    unit_col = "slide_id" if (is_mil or eq_slide) else "case_id"
    available = train_df.groupby("cancer_type")[unit_col].nunique().to_dict()
    support = {
        cls: {
            "patients": int(
                cast(
                    pd.Series, train_df[train_df["cancer_type"] == cls]["case_id"]
                ).nunique()
            ),
            "slides": int(
                cast(
                    pd.Series, train_df[train_df["cancer_type"] == cls]["slide_id"]
                ).nunique()
            ),
        }
        for cls in classes
    }
    return train_df, classes, is_mil, eq_slide, pilot_levels_for(available), support


def _run_all_pilot_seeds(
    train_df: pd.DataFrame,
    classes: list[str],
    levels: list[int],
    is_mil: bool,
    base_seed: int,
    config: dict[str, Any],
    paths: dict[str, Any],
) -> tuple[
    list[int],
    dict[int, int | None],
    dict[int, list[float]],
    dict[int, list[list[float]]],
]:
    """Run every pilot construction seed and collect its candidate-level curves."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds: ImbalanceDataset | BagFeatureDataset = load_training_dataset(
        paths["data"] / "manifest.csv",
        is_mil,
        "validation",
        device=device,
        bag_kwargs=bag_dataset_kwargs(
            config, seed=derive_seed(base_seed, "instance_selection")
        )
        if is_mil
        else None,
    )
    scratch_dir = paths["data"] / "pilot"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    n_cls = len(classes)
    pilot_seeds = [derive_seed(base_seed, f"pilot_construction_{i}") for i in range(3)]
    # One frozen patch quota is shared by every ordering; MIL has no quota.
    quota = (
        None
        if is_mil
        else frozen_pilot_quota(train_df, classes, levels[-1], pilot_seeds)
    )
    quotas, ba_by_seed, recall_by_seed = {}, {}, {}
    for seed in pilot_seeds:
        _, ba_curve, recall_curve = run_pilot_seed(
            train_df,
            classes,
            levels,
            seed,
            val_ds,
            device,
            n_cls,
            is_mil,
            scratch_dir,
            quota,
            initialization_seed=derive_seed(base_seed, "initialization"),
        )
        quotas[seed], ba_by_seed[seed], recall_by_seed[seed] = (
            quota,
            ba_curve,
            recall_curve,
        )
    return pilot_seeds, quotas, ba_by_seed, recall_by_seed


def _pilot_report_payload(
    levels: list[int],
    is_mil: bool,
    eq_slide: bool,
    pilot_seeds: list[int],
    quotas: dict[int, int | None],
    ba_by_seed: dict[int, list[float]],
    recall_by_seed: dict[int, list[list[float]]],
    support: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Assemble the frozen pilot report: floors, curves, seeds, and exclusion status."""
    stability_floor = stability_floor_from_curve(levels, ba_by_seed, recall_by_seed)
    floor = method_floor(eq_slide)
    # Pilot levels count the regime's independent unit (slides for MIL or when
    # patient==slide, otherwise patients). The definitive floor combines the
    # stability floor with only the method floor for *that* unit; the second
    # unit's floor (e.g. 20 slides for patch patients) is enforced separately
    # by ``meets_method_floor`` and must not collapse onto the level count.
    level_unit = "slides" if (is_mil or eq_slide) else "patients"
    definitive_floor = max(stability_floor, floor[level_unit])
    floor_met = all(meets_method_floor(values, eq_slide) for values in support.values())
    return {
        "levels": levels,
        "pilot_construction_seeds": pilot_seeds,
        "quotas": {str(s): q for s, q in quotas.items()},
        "balanced_accuracy_by_seed": {str(s): v for s, v in ba_by_seed.items()},
        "per_class_recall_by_seed": {str(s): v for s, v in recall_by_seed.items()},
        "stability_floor": stability_floor,
        "method_floor": floor,
        "definitive_floor": definitive_floor,
        "patient_equals_slide": eq_slide,
        "available_independent_support": support,
        "pilot_exceptions": (
            ["five-slide MIL pilot uses one slide from each of five distinct patients"]
            if is_mil and 5 in levels
            else []
        ),
        "excluded": levels[-1] < definitive_floor or not floor_met,
    }


def cmd_pilot(args: argparse.Namespace) -> None:
    """Run the nested support-stability pilot and freeze the definitive floors."""
    if args.split_index is None:
        for index in range(3):
            cmd_pilot(argparse.Namespace(**{**vars(args), "split_index": index}))
        return
    config = load_config(args.config)
    paths = split_paths(ensure_dirs(config), args.split_index)
    train_df, classes, is_mil, eq_slide, levels, support = _pilot_setup(paths, config)
    pilot_seeds, quotas, ba_by_seed, recall_by_seed = _run_all_pilot_seeds(
        train_df, classes, levels, is_mil, args.seed, config, paths
    )
    payload = _pilot_report_payload(
        levels,
        is_mil,
        eq_slide,
        pilot_seeds,
        quotas,
        ba_by_seed,
        recall_by_seed,
        support,
    )
    report_path = paths["data"] / "pilot_report.json"
    write_json(report_path, payload)
    sign_file(report_path)
