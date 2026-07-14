from __future__ import annotations

from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd
import torch

from imbalance_benchmark.construction import select_slides_round_robin
from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    bag_collate,
)
from imbalance_benchmark.modeling.evaluation import per_class_recall
from imbalance_benchmark.modeling.models import AttentionMil
from imbalance_benchmark.manifest.pilot_training import (
    meets_method_floor,
    method_floor,
    stability_floor_from_curve,
)
from imbalance_benchmark.manifest.pilot_training import fit_pilot_model

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
PATCH_PILOT_SMALL_COUNT_EXCEPTION_LEVEL = 5


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
    patients = cast(np.ndarray, df_class["case_id"].unique())
    return list(np.random.default_rng(seed).permutation(patients))


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
            if _patch_pilot_caps_hold(sel, level):
                quotas.append(q)
                break
        else:
            raise ValueError("Pilot inventory cannot satisfy patient and slide caps")
    return max(1, min(quotas))


def _patch_pilot_caps_hold(selection: pd.DataFrame, level: int) -> bool:
    """Apply contribution caps except at the recorded five-patient pilot exception."""
    if level == PATCH_PILOT_SMALL_COUNT_EXCEPTION_LEVEL:
        return True
    total = len(selection)
    return (
        selection["case_id"].value_counts().max() / total <= 0.10
        and selection["slide_id"].value_counts().max() / total <= 0.05
    )


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
        if not _patch_pilot_caps_hold(sel, level):
            raise ValueError(
                "Pilot manifest violates the patient or slide contribution cap"
            )
        parts.append(sel)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mil_pilot_manifest(
    df: pd.DataFrame, classes: list[str], level: int, seed: int
) -> pd.DataFrame:
    """Build one nested MIL pilot manifest of `level` slides per class."""
    parts = [
        select_slides_round_robin(
            cast(pd.DataFrame, df[df["cancer_type"] == c]),
            level,
            seed,
            allow_small_count_cap_exception=True,
        )
        for c in classes
    ]
    return pd.concat(parts, ignore_index=True)


def _pilot_val_predictions(
    model: torch.nn.Module,
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
    initialization_seed: int,
    config: dict[str, object] | None = None,
    bag_kwargs: dict[str, int] | None = None,
) -> tuple[float, list[float]]:
    """Fit one balanced-CE pilot candidate and return its validation BA and recalls."""
    df.to_csv(scratch, index=False)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, collate_fn=bag_collate if is_mil else None
    )
    model, best_acc = fit_pilot_model(
        scratch,
        device,
        n_cls,
        is_mil,
        loader,
        initialization_seed=initialization_seed,
        config=config,
        bag_kwargs=bag_kwargs,
    )
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
    *,
    initialization_seed: int,
    config: dict[str, object] | None = None,
    bag_kwargs: dict[str, int] | None = None,
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
            manifest,
            val_ds,
            device,
            n_cls,
            is_mil,
            p,
            initialization_seed=initialization_seed,
            config=config,
            bag_kwargs=bag_kwargs,
        )
        ba_curve.append(ba)
        recall_curve.append(recalls)
    return quota, ba_curve, recall_curve
