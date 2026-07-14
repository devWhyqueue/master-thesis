from __future__ import annotations

from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from imbalance_benchmark.construction import select_slides_round_robin
from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    bag_collate,
)
from imbalance_benchmark.modeling.evaluation import per_class_recall
from imbalance_benchmark.modeling.models import MLP, AttentionMil
from imbalance_benchmark.modeling.training import fit_model
from imbalance_benchmark.manifest.floors import meets_method_floor, method_floor

__all__ = [
    "PILOT_CANDIDATE_LEVELS",
    "pilot_levels_for",
    "compute_pilot_quota",
    "frozen_pilot_quota",
    "build_patch_pilot_manifest",
    "mil_pilot_manifest",
    "evaluate_pilot_candidate",
    "run_pilot_seed",
    "method_floor",
    "meets_method_floor",
    "stability_floor_from_curve",
]

PILOT_CANDIDATE_LEVELS = (5, 10, 15, 20, 30)


def pilot_levels_for(available_per_class: dict[str, int]) -> list[int]:
    """Return nested candidate independent-unit counts capped by the scarcest class."""
    cap = min(available_per_class.values())
    levels = [c for c in PILOT_CANDIDATE_LEVELS if c <= cap]
    if cap >= PILOT_CANDIDATE_LEVELS[-1] or (levels and levels[-1] == cap):
        return levels
    return levels + [cap]


def _patient_order(df_class: pd.DataFrame, seed: int) -> list[str]:
    """Return one seeded master patient ordering for a class."""
    patients = cast(np.ndarray, df_class["case_id"].unique()).copy()
    np.random.default_rng(seed).shuffle(patients)
    return list(patients)


def compute_pilot_quota(
    train_df: pd.DataFrame, classes: list[str], level: int, seed: int
) -> int:
    """Largest per-patient patch quota feasible for every class at one pilot level."""
    quotas = []
    for cls in classes:
        df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls])
        pats = _patient_order(df, seed)[:level]
        counts = df[df["case_id"].isin(pats)].groupby("case_id").size()
        if len(counts) != level:
            raise ValueError("Pilot ordering did not retain every requested patient")
        quotas.append(int(counts.min()))
    return max(1, min(quotas))


def frozen_pilot_quota(
    train_df: pd.DataFrame, classes: list[str], level: int, seeds: list[int]
) -> int:
    """Determine the maximum patch quota feasible across all seeds."""
    return min(compute_pilot_quota(train_df, classes, level, seed) for seed in seeds)


def _apportion_quota(
    df_class: pd.DataFrame, patients: list[str], quota: int, seed: int
) -> pd.DataFrame:
    """Distribute a fixed patch quota round-robin across each patient's slides."""
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    for pat in patients:
        pat_df = cast(pd.DataFrame, df_class[df_class["case_id"] == pat])
        slides = cast(np.ndarray, pat_df["slide_id"].unique()).copy()
        rng.shuffle(slides)
        pools = {
            s: list(rng.permutation(pat_df[pat_df["slide_id"] == s].index))
            for s in slides
        }
        taken, s_idx = 0, 0
        while taken < quota and any(pools.values()):
            s = slides[s_idx % len(slides)]
            if pools[s]:
                rows.append(pools[s].pop(0))
                taken += 1
            s_idx += 1
    return df_class.loc[rows]


def build_patch_pilot_manifest(
    train_df: pd.DataFrame, classes: list[str], level: int, quota: int, seed: int
) -> pd.DataFrame:
    """Build one nested patch pilot manifest: `level` patients/class x `quota` patches."""
    parts = []
    for cls in classes:
        df_class = cast(pd.DataFrame, train_df[train_df["cancer_type"] == cls])
        pat = _patient_order(df_class, seed)[:level]
        sel = _apportion_quota(df_class, pat, quota, seed)
        counts = sel["case_id"].value_counts()
        if len(counts) != level or not (counts == quota).all():
            raise ValueError("Pilot quota is not feasible for every selected patient")
        parts.append(sel)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mil_pilot_manifest(
    train_df: pd.DataFrame, classes: list[str], level: int, seed: int
) -> pd.DataFrame:
    """Build one nested MIL pilot manifest of `level` slides per class."""
    parts = [
        select_slides_round_robin(
            cast(pd.DataFrame, train_df[train_df["cancer_type"] == c]), level, seed
        )
        for c in classes
    ]
    return pd.concat(parts, ignore_index=True)


def _fit_pilot_model(
    scratch_path: Path,
    device: torch.device,
    n_cls: int,
    is_mil: bool,
    val_loader: torch.utils.data.DataLoader,
) -> tuple[nn.Module, float]:
    """Instantiate and fit one balanced-CE pilot candidate model."""
    ds = (
        BagFeatureDataset(scratch_path, device=device)
        if is_mil
        else ImbalanceDataset(scratch_path, device=device)
    )
    model = (
        AttentionMil(2560, 256, n_cls, 0.1) if is_mil else MLP(2560, 512, n_cls, 0.1)
    ).to(device)
    ctx = {
        "method": "ce",
        "model": model,
        "train_dataset": ds,
        "val_loader": val_loader,
        "device": device,
        "config": {},
        "param_config": {"lr": 1e-3},
        "seed": 0,
        "is_mil": is_mil,
        "n_classes": n_cls,
        "train_labels": ds.get_int_targets(),
    }
    _, best_acc = fit_model(ctx)
    return model, best_acc


def _pilot_val_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    is_mil: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather validation predictions and targets for one fitted pilot model."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_mil:
                bags, tgt = batch
                dev_bags = [b.to(device) for b in bags]
                logits = cast(AttentionMil, model).forward_bags(dev_bags)[0]
            else:
                logits, tgt = model(batch["features"].to(device)), batch["target"]
            preds.append(logits.softmax(-1).argmax(-1).cpu())
            targets.append(tgt)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate_pilot_candidate(
    manifest_df: pd.DataFrame,
    val_ds: ImbalanceDataset | BagFeatureDataset,
    device: torch.device,
    n_cls: int,
    is_mil: bool,
    scratch_path: Path,
) -> tuple[float, list[float]]:
    """Fit one balanced-CE pilot candidate and return its validation BA and recalls."""
    manifest_df.to_csv(scratch_path, index=False)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, collate_fn=bag_collate if is_mil else None
    )
    model, best_acc = _fit_pilot_model(scratch_path, device, n_cls, is_mil, loader)
    preds, targets = _pilot_val_predictions(model, loader, device, is_mil)
    return best_acc, per_class_recall(preds, targets, n_cls)


def run_pilot_seed(
    train_df: pd.DataFrame,
    classes: list[str],
    levels: list[int],
    seed: int,
    val_ds: ImbalanceDataset | BagFeatureDataset,
    device: torch.device,
    n_cls: int,
    is_mil: bool,
    scratch_dir: Path,
    quota: int | None,
) -> tuple[int | None, list[float], list[list[float]]]:
    """Run every nested candidate level for one pilot construction seed at the frozen quota."""
    ba_curve, recall_curve = [], []
    for level in levels:
        if is_mil:
            manifest = mil_pilot_manifest(train_df, classes, level, seed)
        else:
            q = cast(int, quota)
            manifest = build_patch_pilot_manifest(train_df, classes, level, q, seed)
        path = scratch_dir / f"pilot_seed={seed}_level={level}.csv"
        ba, recalls = evaluate_pilot_candidate(
            manifest, val_ds, device, n_cls, is_mil, path
        )
        ba_curve.append(ba)
        recall_curve.append(recalls)
    return quota, ba_curve, recall_curve


def stability_floor_from_curve(
    levels: list[int],
    mean_ba_by_seed: dict[int, list[float]],
    class_recall_by_seed: dict[int, list[list[float]]],
) -> int:
    """Smallest level whose next-level gain is below the mean/per-class stability rule."""
    mean_ba = np.mean(np.stack(list(mean_ba_by_seed.values())), axis=0)
    for idx in range(len(levels) - 1):
        mean_gain = abs(float(mean_ba[idx + 1] - mean_ba[idx]))
        class_gain = max(
            abs(rc[idx + 1][c] - rc[idx][c])
            for rc in class_recall_by_seed.values()
            for c in range(len(rc[idx]))
        )
        if mean_gain < 0.01 and class_gain < 0.02:
            return levels[idx]
    return levels[-1]
