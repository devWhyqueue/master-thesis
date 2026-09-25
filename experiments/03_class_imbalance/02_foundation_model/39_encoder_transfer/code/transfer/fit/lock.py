"""Current signed input and source identity for exp-39 fit records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    compute_data_hash,
    compute_sha256,
    ensure_dirs,
    find_repo_root,
    output_root,
    split_paths,
    verify_signed_file,
)


def _signed_paths(
    config: dict[str, Any], encoder: str, split_idx: int, exp_root: Path
) -> dict[str, Path]:
    signed = {
        name: exp_root / "configs" / name
        for name in (
            "protocol_lock.json",
            "encoder_lock.json",
            "input_audit.json",
            "phase05_scope_amendment.json",
        )
    }
    signed.update(
        schedule=output_root(config) / "schedule.json",
        feature_audit=output_root(config) / "data" / "feature_audit.json",
        manifest=split_paths(ensure_dirs(config), split_idx)["data"]
        / f"manifest_{encoder}.csv",
    )
    return signed


def fit_lock(config: dict[str, Any], encoder: str, split_idx: int) -> dict[str, Any]:
    """Bind a fit to the signed data and current experiment source."""
    exp_root = (
        find_repo_root()
        / "experiments/03_class_imbalance/02_foundation_model/39_encoder_transfer"
    )
    signed = _signed_paths(config, encoder, split_idx, exp_root)
    for path in signed.values():
        verify_signed_file(path)
    code_hashes = {
        str(path.relative_to(exp_root)): compute_sha256(path)
        for path in sorted((exp_root / "code").rglob("*.py"))
    }
    return {
        "encoder": encoder,
        "split": split_idx,
        "config_sha256": compute_data_hash(
            {k: v for k, v in config.items() if k != "slurm"}
        ),
        "source_sha256": compute_data_hash(code_hashes),
        "signed_sha256": {name: compute_sha256(path) for name, path in signed.items()},
    }
