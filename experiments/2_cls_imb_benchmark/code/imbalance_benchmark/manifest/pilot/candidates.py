from __future__ import annotations

import logging
from dataclasses import dataclass
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
    patch_collate,
)
from imbalance_benchmark.modeling.evaluation import per_class_recall
from imbalance_benchmark.modeling.models import AttentionMil
from imbalance_benchmark.manifest.pilot.training import (
    compute_pilot_quota,
    eligible_patient_order,
    fit_pilot_model,
    frozen_pilot_quota,
    meets_method_floor,
    method_floor,
    patch_pilot_caps_hold,
    stability_floor_from_curve,
)
from imbalance_benchmark.manifest.pilot.training import (
    _apportion_quota as apportion_quota,
)

ValDataset = ImbalanceDataset | BagFeatureDataset

__all__ = [
    "PILOT_CANDIDATE_LEVELS",
    "PilotFit",
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

logger = logging.getLogger(__name__)

PILOT_CANDIDATE_LEVELS = (10, 15, 20, 30, 50)


def pilot_levels_for(available_per_class: dict[str, int]) -> list[int]:
    """Return nested candidate independent-unit counts capped by the scarcest class."""
    cap = min(available_per_class.values())
    levels = [c for c in PILOT_CANDIDATE_LEVELS if c <= cap]
    if cap >= PILOT_CANDIDATE_LEVELS[-1] or (levels and levels[-1] == cap):
        return levels
    return levels + [cap]


def build_patch_pilot_manifest(
    df: pd.DataFrame, classes: list[str], level: int, quota: int, seed: int
) -> pd.DataFrame:
    """Build one nested patch pilot manifest: `level` patients/class x `quota` patches."""
    parts = []
    for cls in classes:
        df_class = cast(pd.DataFrame, df[df["cancer_type"] == cls])
        pat = eligible_patient_order(df_class, quota, level, seed)
        sel = apportion_quota(df_class, pat, quota, seed)
        v = sel["case_id"].value_counts()
        if len(v) != level or not (v == quota).all():
            raise ValueError("Pilot quota is not feasible for every selected patient")
        if not patch_pilot_caps_hold(sel):
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
            cast(pd.DataFrame, df[df["cancer_type"] == c]), level, seed
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
                bags = [b.to(device) for b in bags]
                logits = cast(AttentionMil, model).forward_bags(bags)[0]
            else:
                logits, tgt = model(batch["features"].to(device)), batch["target"]
            preds.append(logits.softmax(-1).argmax(-1).cpu())
            targets.append(tgt)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


@dataclass(frozen=True)
class PilotFit:
    """The fixed balanced-CE fitting context every pilot candidate level shares."""

    val_ds: ValDataset
    device: torch.device
    n_cls: int
    is_mil: bool
    initialization_seed: int
    config: dict[str, object] | None = None


def evaluate_pilot_candidate(
    df: pd.DataFrame, scratch: Path, fit: PilotFit
) -> tuple[float, list[float]]:
    """Fit one balanced-CE pilot candidate and return its validation BA and recalls."""
    df.to_csv(scratch, index=False)
    loader = torch.utils.data.DataLoader(
        fit.val_ds,
        batch_size=64,
        collate_fn=bag_collate if fit.is_mil else patch_collate,  # type: ignore[arg-type]
    )
    model, best_acc = fit_pilot_model(
        scratch,
        fit.device,
        fit.n_cls,
        fit.is_mil,
        loader,
        initialization_seed=fit.initialization_seed,
        config=fit.config,
    )
    preds, targets = _pilot_val_predictions(model, loader, fit.device, fit.is_mil)
    return best_acc, per_class_recall(preds, targets, fit.n_cls)


def run_pilot_seed(
    df: pd.DataFrame,
    classes: list[str],
    levels: list[int],
    seed: int,
    scratch_dir: Path,
    quota: int | None,
    fit: PilotFit,
) -> tuple[int | None, list[float], list[list[float]]]:
    """Run every nested candidate level for one pilot construction seed at the frozen quota."""
    ba_curve, recall_curve = [], []
    for position, level in enumerate(levels, start=1):
        logger.info(
            "pilot seed %d: fitting level %d (%d/%d)",
            seed,
            level,
            position,
            len(levels),
        )
        manifest = (
            mil_pilot_manifest(df, classes, level, seed)
            if fit.is_mil
            else build_patch_pilot_manifest(df, classes, level, quota or 0, seed)
        )
        p = scratch_dir / f"pilot_seed={seed}_level={level}.csv"
        ba, recalls = evaluate_pilot_candidate(manifest, p, fit)
        logger.info("pilot seed %d: level %d balanced accuracy %.4f", seed, level, ba)
        ba_curve.append(ba)
        recall_curve.append(recalls)
    return quota, ba_curve, recall_curve
