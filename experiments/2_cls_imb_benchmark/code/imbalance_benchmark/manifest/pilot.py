from __future__ import annotations

from pathlib import Path
from typing import Any, cast
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

ValDataset = ImbalanceDataset | BagFeatureDataset

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
    return (
        levels
        if (cap >= PILOT_CANDIDATE_LEVELS[-1] or (levels and levels[-1] == cap))
        else levels + [cap]
    )


def _patient_order(df_class: pd.DataFrame, seed: int) -> list[str]:
    pats = list(df_class["case_id"].unique())
    np.random.default_rng(seed).shuffle(pats)
    return pats


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
            n = len(sel)
            if (
                sel["case_id"].value_counts().max() / n <= 0.10
                and sel["slide_id"].value_counts().max() / n <= 0.05
            ):
                quotas.append(q)
                break
        else:
            raise ValueError("Pilot inventory cannot satisfy patient and slide caps")
    return max(1, min(quotas))


def frozen_pilot_quota(
    df: pd.DataFrame, classes: list[str], level: int, seeds: list[int]
) -> int:
    """Determine the maximum patch quota feasible across all seeds."""
    return min(compute_pilot_quota(df, classes, level, seed) for seed in seeds)


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


def build_patch_pilot_manifest(
    df: pd.DataFrame, classes: list[str], level: int, quota: int, seed: int
) -> pd.DataFrame:
    """Build one nested patch pilot manifest: `level` patients/class x `quota` patches."""
    parts = []
    for cls in classes:
        df_class = cast(pd.DataFrame, df[df["cancer_type"] == cls])
        pat = _patient_order(df_class, seed)[:level]
        sel = _apportion_quota(df_class, pat, quota, seed)
        v = sel["case_id"].value_counts()
        if len(v) != level or not (v == quota).all():
            raise ValueError("Pilot quota is not feasible for every selected patient")
        parts.append(sel)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mil_pilot_manifest(
    df: pd.DataFrame, classes: list[str], level: int, seed: int
) -> pd.DataFrame:
    """Build one nested MIL pilot manifest of `level` slides per class."""
    parts = [
        select_slides_round_robin(
            cast(pd.DataFrame, df[df["cancer_type"] == c]), level, seed
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
    ds_cls = BagFeatureDataset if is_mil else ImbalanceDataset
    ds = ds_cls(scratch_path, device=device)
    dim = 256 if is_mil else 512
    model = (AttentionMil if is_mil else MLP)(2560, dim, n_cls, 0.1).to(device)
    ctx: dict[str, Any] = dict(
        method="ce", model=model, train_dataset=ds, val_loader=val_loader, device=device
    )
    ctx |= dict(
        config={}, param_config={"lr": 1e-3}, seed=0, is_mil=is_mil, n_classes=n_cls
    )
    ctx["train_labels"] = ds.get_int_targets()
    _, best_acc = fit_model(ctx)
    return model, best_acc


def _pilot_val_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    is_mil: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_mil:
                bags, tgt = batch
                logits = cast(AttentionMil, model).forward_bags(
                    [b.to(device) for b in bags]
                )[0]
            else:
                logits, tgt = model(batch["features"].to(device)), batch["target"]
            preds.append(logits.softmax(-1).argmax(-1).cpu())
            targets.append(tgt)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def evaluate_pilot_candidate(
    df: pd.DataFrame,
    val_ds: ValDataset,
    device: torch.device,
    n_cls: int,
    is_mil: bool,
    scratch: Path,
) -> tuple[float, list[float]]:
    """Fit one balanced-CE pilot candidate and return its validation BA and recalls."""
    df.to_csv(scratch, index=False)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, collate_fn=bag_collate if is_mil else None
    )
    model, best_acc = _fit_pilot_model(scratch, device, n_cls, is_mil, loader)
    preds, targets = _pilot_val_predictions(model, loader, device, is_mil)
    return best_acc, per_class_recall(preds, targets, n_cls)


def run_pilot_seed(
    df: pd.DataFrame,
    classes: list[str],
    levels: list[int],
    seed: int,
    val_ds: ValDataset,
    device: torch.device,
    n_cls: int,
    is_mil: bool,
    scratch_dir: Path,
    quota: int | None,
) -> tuple[int | None, list[float], list[list[float]]]:
    """Run every nested candidate level for one pilot construction seed at the frozen quota."""
    ba_curve, recall_curve = [], []
    for level in levels:
        manifest = (
            mil_pilot_manifest(df, classes, level, seed)
            if is_mil
            else build_patch_pilot_manifest(df, classes, level, quota or 0, seed)
        )
        p = scratch_dir / f"pilot_seed={seed}_level={level}.csv"
        ba, recalls = evaluate_pilot_candidate(
            manifest, val_ds, device, n_cls, is_mil, p
        )
        ba_curve.append(ba)
        recall_curve.append(recalls)
    return quota, ba_curve, recall_curve


def stability_floor_from_curve(
    levels: list[int], ba: dict[int, list[float]], rcs: dict[int, list[list[float]]]
) -> int:
    """Smallest level whose next-level gain is below the mean/per-class stability rule."""
    mean_ba = np.mean(np.stack(list(ba.values())), axis=0)
    for idx in range(len(levels) - 1):
        g = abs(float(mean_ba[idx + 1] - mean_ba[idx]))
        cg = max(
            abs(rc[idx + 1][c] - rc[idx][c])
            for rc in rcs.values()
            for c in range(len(rc[idx]))
        )
        if g < 0.01 and cg < 0.02:
            return levels[idx]
    return levels[-1]
