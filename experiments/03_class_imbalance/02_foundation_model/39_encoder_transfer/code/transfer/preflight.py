"""Preflight gate: verify the frozen locks, schedule, and per-encoder feature caches are
intact and complete before any fit shard runs, re-checked together as one signed report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    ensure_dirs,
    find_repo_root,
    output_root,
    sign_file,
    split_paths,
    verify_signed_file,
    write_json,
)

__all__ = ["run_preflight"]

_LOCK_DIR = (
    find_repo_root()
    / "experiments/03_class_imbalance/02_foundation_model/39_encoder_transfer/configs"
)
_LOCKS = (
    "protocol_lock.json",
    "encoder_lock.json",
    "input_audit.json",
    "phase05_scope_amendment.json",
)


def _verify_locks() -> list[str]:
    """Every frozen phase-01/02 lock is present and matches its signed hash."""
    checked = []
    for name in _LOCKS:
        verify_signed_file(_LOCK_DIR / name)
        checked.append(name)
    return checked


def _verify_feature_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Phase 03's feature_audit.json is signed and reports zero unresolved rows."""
    audit_path = output_root(config) / "data" / "feature_audit.json"
    verify_signed_file(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit["unresolved_missing"]
        or audit["unresolved_corrupt"]
        or audit["unresolved_mismatched"]
    ):
        raise RuntimeError(f"feature_audit.json reports unresolved rows: {audit_path}")
    amendment = json.loads(
        (_LOCK_DIR / "phase05_scope_amendment.json").read_text(encoding="utf-8")
    )[config["dataset"]["name"]]
    expected = amendment["union_patches"]
    if (
        audit["uni2h"]["requested_patches"] != expected
        or audit["uni2h"]["matched_patches"] != expected
        or audit["uni2h"]["requested_slides"] != amendment["union_unique_slides"]
        or audit["virchow2"]["matched_patches"] != expected
    ):
        raise RuntimeError(
            f"feature_audit.json does not match corrected scope: {audit_path}"
        )
    return audit


def _verify_manifests(config: dict[str, Any]) -> dict[str, int]:
    """Every split has both encoders' joined manifests, signed and row-count-matched."""
    counts: dict[str, int] = {}
    for split_idx in range(N_PATIENT_SPLITS):
        data_dir = split_paths(ensure_dirs(config), split_idx)["data"]
        rows: dict[str, int] = {}
        for encoder in ("uni2h", "virchow2"):
            m_file = data_dir / f"manifest_{encoder}.csv"
            verify_signed_file(m_file)
            rows[encoder] = sum(1 for _ in m_file.open(encoding="utf-8")) - 1
        if rows["uni2h"] != rows["virchow2"]:
            raise RuntimeError(
                f"split {split_idx}: manifest row counts diverge: {rows}"
            )
        counts[f"split_{split_idx}"] = rows["uni2h"]
    return counts


def run_preflight(config: dict[str, Any]) -> Path:
    """Verify locks, feature audit, and joined manifests; write a signed preflight.json."""
    locks = _verify_locks()
    feature_audit = _verify_feature_audit(config)
    manifest_rows = _verify_manifests(config)
    report = {
        "status": "pass",
        "dataset": config.get("dataset", {}).get("name"),
        "verified_locks": locks,
        "feature_audit": {
            "uni2h_matched": feature_audit["uni2h"]["matched_patches"],
            "virchow2_matched": feature_audit["virchow2"]["matched_patches"],
        },
        "manifest_rows_per_split": manifest_rows,
    }
    out_p = output_root(config) / "data" / "preflight.json"
    write_json(out_p, report)
    sign_file(out_p)
    return out_p
