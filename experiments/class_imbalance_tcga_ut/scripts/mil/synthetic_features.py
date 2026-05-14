from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.common import output_root, write_json
from scripts.mil.bags import SyntheticBagFeatureDataset

REQUIRED_COLUMNS = {"cancer_type", "feature_path"}


def append_encoded_gan_features(
    train_dataset: SyntheticBagFeatureDataset,
    class_to_idx: dict[str, int],
    max_instances: int | None,
    config: dict,
    method: str,
    seed: int,
    result_dir: Path,
) -> bool:
    """Append encoded generated-image features when available."""
    if method != "feature_gan_mil":
        return False
    gan_config = config.get("synthetic_image_gan", {})
    manifest = _encoded_feature_manifest(config, gan_config, seed)
    if not manifest.exists():
        status, reason = _missing_manifest_status(manifest)
        _write_synthetic_feature_note(result_dir, manifest, status, reason)
        if bool(gan_config.get("require_encoded_features", False)):
            raise FileNotFoundError(f"Missing encoded GAN feature manifest: {manifest}")
        return False
    frame = pd.read_csv(manifest)
    validated, summary = _validated_feature_frame(frame, class_to_idx)
    added = train_dataset.append_rows(validated, class_to_idx, max_instances)
    _write_synthetic_feature_note(result_dir, manifest, "loaded", None, summary, added)
    if added == 0 and bool(gan_config.get("require_encoded_features", False)):
        raise ValueError(f"No usable encoded GAN feature rows in manifest: {manifest}")
    return added > 0


def _encoded_feature_manifest(config: dict, gan_config: dict, seed: int) -> Path:
    template = str(
        gan_config.get(
            "encoded_feature_manifest_template",
            "outputs/synthetic_features/seed={seed}/synthetic_feature_manifest.csv",
        )
    )
    configured = Path(template.format(seed=seed))
    if configured.is_absolute():
        return configured
    return output_root(config) / configured


def _missing_manifest_status(manifest: Path) -> tuple[str, str | None]:
    progress_path = manifest.parent / "progress.json"
    if not progress_path.exists():
        return "missing", None
    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    if progress.get("status") == "failed":
        return "encoder_failed", str(progress.get("reason", "unknown"))
    return "missing", None


def _validated_feature_frame(
    frame: pd.DataFrame, class_to_idx: dict[str, int]
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Encoded GAN feature manifest is missing columns: {columns}")
    known = frame["cancer_type"].astype(str).isin(class_to_idx)
    existing = frame["feature_path"].map(lambda value: Path(str(value)).exists())
    valid = frame.loc[known & existing].copy()
    return valid, {
        "manifest_rows": int(len(frame)),
        "valid_rows": int(len(valid)),
        "unknown_class_rows": int((~known).sum()),
        "missing_feature_files": int((~existing).sum()),
    }


def _write_synthetic_feature_note(
    result_dir: Path,
    manifest: Path,
    status: str,
    reason: str | None,
    summary: dict[str, int] | None = None,
    n_added: int = 0,
) -> None:
    payload: dict[str, object] = {
        "status": status,
        "manifest": str(manifest),
        "n_added": n_added,
    }
    if reason is not None:
        payload["reason"] = reason
    if summary is not None:
        payload.update(summary)
    write_json(
        result_dir / "synthetic_feature_inputs.json",
        payload,
    )
