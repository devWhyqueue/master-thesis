"""Virchow2 pointer audit and the cross-encoder manifest join (exp-39, phase 03).

Virchow2 needs no extraction: both datasets' features are already cached and
were verified in phase 02 (``configs/input_audit.json``). This module only
re-verifies the existing pointers for the exact rows exp-39 requests, then
writes each encoder a model-specific manifest copy that changes only the
feature-reference columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    ensure_dirs,
    find_repo_root,
    output_root,
    sign_file,
    split_paths,
    write_json,
)
from imbalance_benchmark.datasets.features.cache import load_slide_features

from prevalence import patients_per_class

from transfer.extract import audit_uni2h
from transfer.features import (
    IDENTITY_COLS,
    _full_manifest,
    _split_requested_rows,
    _train_union_identity,
    requested_mask,
)
from transfer.schedule import load_train_identity

__all__ = ["audit_virchow2", "join_manifest", "run_audit"]

# BRACS-only correction (configs/input_audit.json, virchow2_audit.bracs.correction):
# exp-2's own bracs manifest.csv feature_path column points at an empty,
# experiment-local dead end. The row-for-row identical, real cache lives in
# this shared project manifest instead. TCGA-UT needs no such substitution.
_BRACS_VIRCHOW2_MANIFEST_ROOT = Path("/home/space/datasets/bracs/manifests")

type PatchKey = tuple[str, str, str]
type FeatureRef = tuple[str, int]


def _resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else find_repo_root() / path


def _virchow2_reference_frame(config: dict[str, Any], split_idx: int) -> pd.DataFrame:
    """Identity + feature_path/feature_index columns for one split's Virchow2 pointers."""
    ref_cols = list(IDENTITY_COLS) + ["feature_path", "feature_index"]
    if config.get("dataset", {}).get("name") == "bracs":
        shared = pd.read_csv(
            _BRACS_VIRCHOW2_MANIFEST_ROOT / f"split={split_idx}" / "manifest.csv"
        )
        return cast(pd.DataFrame, shared[ref_cols])
    full = _full_manifest(config, split_idx)
    return cast(pd.DataFrame, full[ref_cols])


def _virchow2_pointer(
    key: PatchKey, keyed: pd.DataFrame
) -> tuple[str, FeatureRef | None]:
    """Resolve and verify one existing Virchow2 pointer: ("ok"|"missing"|"corrupt", ref)."""
    try:
        row = keyed.loc[key]
    except KeyError:
        return "missing", None
    resolved = _resolve_repo_path(row["feature_path"])
    row_index = int(row["feature_index"])
    if not resolved.is_file():
        return "missing", None
    tensor = load_slide_features(str(resolved))
    if (
        tensor.ndim != 2
        or tensor.shape[1] != 2560
        or row_index >= tensor.shape[0]
        or not torch.isfinite(tensor[row_index]).all()
    ):
        return "corrupt", None
    return "ok", (str(resolved), row_index)


def _audit_split(
    config: dict[str, Any],
    split_idx: int,
    index: dict[PatchKey, FeatureRef],
    missing: list[PatchKey],
    corrupt: list[PatchKey],
) -> int:
    """Resolve one split's requested rows against its Virchow2 pointers; return its row count."""
    requested = _split_requested_rows(config, split_idx)
    keyed = _virchow2_reference_frame(config, split_idx).set_index(list(IDENTITY_COLS))
    for case, slide, patch in zip(
        requested["case_id"].astype(str),
        requested["slide_id"].astype(str),
        requested["patch_id"].astype(str),
    ):
        key = (case, slide, patch)
        if key in index or key in missing or key in corrupt:
            continue
        status, ref = _virchow2_pointer(key, keyed)
        if status == "ok":
            index[key] = ref  # type: ignore[assignment]
        elif status == "missing":
            missing.append(key)
        else:
            corrupt.append(key)
    return len(requested)


def audit_virchow2(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[PatchKey, FeatureRef]]:
    """Verify existing Virchow2 pointers for every requested row; never re-extract."""
    missing: list[PatchKey] = []
    corrupt: list[PatchKey] = []
    index: dict[PatchKey, FeatureRef] = {}
    requested_total = sum(
        _audit_split(config, split_idx, index, missing, corrupt)
        for split_idx in range(N_PATIENT_SPLITS)
    )
    return (
        {
            "encoder": "virchow2",
            "requested_patches": requested_total,
            "matched_patches": len(index),
            "missing_pointers": [list(k) for k in missing],
            "corrupt_pointers": [list(k) for k in corrupt],
            "unresolved_missing": len(missing),
            "unresolved_corrupt": len(corrupt),
        },
        index,
    )


def _resolve_refs(
    full: pd.DataFrame, mask: pd.Series, index: dict[PatchKey, FeatureRef]
) -> tuple[list[FeatureRef | None], list[PatchKey]]:
    keys = list(
        zip(
            full["case_id"].astype(str),
            full["slide_id"].astype(str),
            full["patch_id"].astype(str),
        )
    )
    refs = [index.get(k) if requested else None for k, requested in zip(keys, mask)]
    missing_required = [
        k for k, r, requested in zip(keys, refs, mask) if requested and r is None
    ]
    return refs, missing_required


def join_manifest(
    config: dict[str, Any],
    split_idx: int,
    index: dict[PatchKey, FeatureRef],
    encoder_name: str,
) -> Path:
    """Write a per-split, model-specific manifest: same rows/order/labels, new feature refs.

    Both encoders share the same requested-row mask, so their manifests match
    exactly once the feature-reference columns are dropped.
    """
    full = _full_manifest(config, split_idx)
    train_df, names = load_train_identity(config, split_idx)
    wanted = _train_union_identity(
        train_df, names, split_idx, patients_per_class(config)
    )
    mask = requested_mask(full, wanted)
    refs, missing_required = _resolve_refs(full, mask, index)
    if missing_required:
        raise ValueError(
            f"{encoder_name}: {len(missing_required)} requested rows have no matched "
            f"feature, e.g. {missing_required[:3]}"
        )
    joined = full.copy()
    joined["feature_path"] = [r[0] if r else pd.NA for r in refs]
    joined["feature_index"] = [r[1] if r else pd.NA for r in refs]
    out_dir = split_paths(ensure_dirs(config), split_idx)["data"]
    out_path = out_dir / f"manifest_{encoder_name}.csv"
    joined.to_csv(out_path, index=False)
    sign_file(out_path)
    return out_path


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Audit both encoders, join both manifests per split, and publish feature_audit.json."""
    uni_audit, uni_index = audit_uni2h(config)
    v2_audit, v2_index = audit_virchow2(config)
    for split_idx in range(N_PATIENT_SPLITS):
        join_manifest(config, split_idx, uni_index, "uni2h")
        join_manifest(config, split_idx, v2_index, "virchow2")
    audit = {
        "experiment": "39_encoder_transfer",
        "phase": "03_feature_extraction",
        "dataset": config["dataset"]["name"],
        "uni2h": uni_audit,
        "virchow2": v2_audit,
        "unresolved_missing": uni_audit["unresolved_missing"]
        + v2_audit["unresolved_missing"],
        "unresolved_corrupt": uni_audit["unresolved_corrupt"]
        + v2_audit["unresolved_corrupt"],
        "unresolved_mismatched": 0,
    }
    out_path = output_root(config) / "data" / "feature_audit.json"
    write_json(out_path, audit)
    sign_file(out_path)
    return audit
