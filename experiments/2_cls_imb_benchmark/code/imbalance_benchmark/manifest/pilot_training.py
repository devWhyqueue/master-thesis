from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from imbalance_benchmark.datasets.data import BagFeatureDataset, ImbalanceDataset
from imbalance_benchmark.modeling.models import AttentionMil, MLP
from imbalance_benchmark.modeling.training import fit_model

PATCH_PILOT_SMALL_COUNT_EXCEPTION_LEVEL = 5


def _patient_order(df_class: pd.DataFrame, seed: int) -> list[str]:
    patients = cast(np.ndarray, df_class["case_id"].unique())
    return list(np.random.default_rng(seed).permutation(patients))


def _apportion_quota(
    df: pd.DataFrame, patients: list[str], quota: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pat in patients:
        p_df = cast(pd.DataFrame, df[df["case_id"] == pat])
        slides = list(p_df["slide_id"].unique())
        rng.shuffle(slides)
        pools = {
            s: list(
                rng.permutation(cast(pd.DataFrame, p_df[p_df["slide_id"] == s]).index)
            )
            for s in slides
        }
        taken, s_idx = 0, 0
        while taken < quota and any(pools.values()):
            s = slides[s_idx % len(slides)]
            if pools[s]:
                rows.append(pools[s].pop(0))
                taken += 1
            s_idx += 1
    return df.loc[rows]


def patch_pilot_caps_hold(selection: pd.DataFrame, level: int) -> bool:
    """Apply contribution caps except at the recorded five-patient pilot exception."""
    if level == PATCH_PILOT_SMALL_COUNT_EXCEPTION_LEVEL:
        return True
    total = len(selection)
    patient_share = selection["case_id"].value_counts().max() / total
    slide_share = selection["slide_id"].value_counts().max() / total
    return patient_share <= 0.10 and slide_share <= 0.05


def compute_pilot_quota(
    df: pd.DataFrame, classes: list[str], level: int, seed: int
) -> int:
    """Largest per-patient patch quota feasible for every class at one pilot level."""
    quotas = []
    for cls in classes:
        c_df = cast(pd.DataFrame, df[df["cancer_type"] == cls])
        pats = _patient_order(c_df, seed)[:level]
        counts = c_df[c_df["case_id"].isin(pats)].groupby("case_id").size()
        if len(counts) != level:
            raise ValueError("Pilot ordering failed.")
        for q in range(int(counts.min()), 0, -1):
            sel = _apportion_quota(c_df, pats, q, seed)
            if patch_pilot_caps_hold(sel, level):
                quotas.append(q)
                break
        else:
            raise ValueError("Pilot inventory cannot satisfy patient and slide caps")
    return max(1, min(quotas))


def frozen_pilot_quota(
    df: pd.DataFrame, classes: list[str], levels: list[int], seeds: list[int]
) -> tuple[int, list[int]]:
    """Quota and levels feasible at every seed.

    The cap tightens as the level shrinks, so a level where no quota
    satisfies every seed is dropped rather than reused from a larger one.
    """
    feasible, quotas = [], []
    for level in levels:
        try:
            level_quotas = [compute_pilot_quota(df, classes, level, s) for s in seeds]
        except ValueError:
            continue
        feasible.append(level)
        quotas.extend(level_quotas)
    return min(quotas), feasible


def method_floor(patient_equals_slide: bool) -> dict[str, int]:
    """Return the fixed method-floor independent-unit minimums per class."""
    return {"slides": 20} if patient_equals_slide else {"patients": 10, "slides": 20}


def meets_method_floor(support: dict[str, int], patient_equals_slide: bool) -> bool:
    """Require every independent-unit floor applicable to the regime."""
    return all(
        support.get(unit, 0) >= minimum
        for unit, minimum in method_floor(patient_equals_slide).items()
    )


def stability_floor_from_curve(
    levels: list[int], ba: dict[int, list[float]], rcs: dict[int, list[list[float]]]
) -> int:
    """Return the first support level whose aggregate and classwise gains are stable.

    The report requires the balanced-accuracy increment below 0.01 and every
    class-recall increment below 0.02 *in all three orderings*. Averaging BA
    across orderings before differencing can let opposite-signed changes cancel,
    so the increment must be evaluated per ordering and the largest one gated.
    """
    for idx in range(len(levels) - 1):
        gain = max(abs(float(curve[idx + 1] - curve[idx])) for curve in ba.values())
        class_gain = max(
            abs(recalls[idx + 1][class_index] - recalls[idx][class_index])
            for recalls in rcs.values()
            for class_index in range(len(recalls[idx]))
        )
        if gain < 0.01 and class_gain < 0.02:
            return levels[idx]
    return levels[-1]


def _pilot_dataset(
    scratch_path: Path,
    device: torch.device,
    is_mil: bool,
    bag_kwargs: dict[str, int] | None,
) -> BagFeatureDataset | ImbalanceDataset:
    """Load the pilot's training manifest with the regime's fixed evidence controls."""
    if is_mil:
        controls = bag_kwargs or {}
        return BagFeatureDataset(
            scratch_path,
            max_instances=controls.get("max_instances", 500),
            instance_selection_seed=controls.get("instance_selection_seed", 0),
            device=device,
        )
    return ImbalanceDataset(scratch_path, device=device)


def fit_pilot_model(
    scratch_path: Path,
    device: torch.device,
    n_classes: int,
    is_mil: bool,
    val_loader: torch.utils.data.DataLoader,
    initialization_seed: int,
    config: dict[str, Any] | None = None,
    bag_kwargs: dict[str, int] | None = None,
) -> tuple[nn.Module, float]:
    """Construct reproducible pilot weights and fit the fixed CE baseline."""
    dataset = _pilot_dataset(scratch_path, device, is_mil, bag_kwargs)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    hidden_dim = 256 if is_mil else 512
    model = (AttentionMil if is_mil else MLP)(2560, hidden_dim, n_classes, 0.1).to(
        device
    )
    context: dict[str, Any] = {
        "method": "ce",
        "model": model,
        "train_dataset": dataset,
        "val_loader": val_loader,
        "device": device,
        "config": config or {},
        "param_config": {"lr": 1e-3},
        "seed": initialization_seed,
        "is_mil": is_mil,
        "n_classes": n_classes,
        "train_labels": dataset.get_int_targets(),
    }
    _, best_accuracy = fit_model(context)
    return model, best_accuracy
