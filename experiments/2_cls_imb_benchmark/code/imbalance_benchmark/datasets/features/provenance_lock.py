from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import (
    compute_data_hash,
    compute_sha256,
    dataset_provenance,
    sign_file,
    verify_signed_file,
    write_json,
)
from imbalance_benchmark.datasets.feature_provenance import (
    load_signed_feature_provenance,
    resolve_feature_provenance,
    validate_preextracted_features,
)

LOCK_NAME = "feature_provenance_lock.json"
LOCK_FIELDS = ("manifest_path", "manifest_sha256", "inventory_sha256")


def attach_frozen_provenance(
    meta: dict[str, Any],
    data_dir: Path,
    config: dict[str, Any],
    feature_provenance: dict[str, str] | None,
) -> None:
    """Attach prepared artifacts and encoder provenance to a freeze."""
    meta.update(
        pilot_report={
            "path": str(data_dir / "pilot_report.json"),
            "sha256": compute_sha256(data_dir / "pilot_report.json"),
        },
        prepared_manifest={
            "path": str(data_dir / "manifest.csv"),
            "sha256": compute_sha256(data_dir / "manifest.csv"),
        },
        dataset_provenance=dataset_provenance(config.get("dataset", {})),
        feature_encoder=resolve_feature_provenance(
            config.get("feature_extraction", {})
        ),
    )
    if feature_provenance is not None:
        meta["feature_provenance"] = feature_provenance


def current_feature_provenance_lock(
    config: dict[str, Any],
) -> dict[str, str] | None:
    """Return stable digests locking TCGA-UT's frozen evidence source.

    The WSI regime still locks the pre-extracted tensor-chunk inventory; the
    patch regime locks the materialized image SqFS instead (see
    :mod:`imbalance_benchmark.datasets.tcga_ut_source`).
    """
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict) or dataset.get("name") != "tcga_ut":
        return None
    if dataset.get("regime") == "wsi":
        return _current_tensor_lock(dataset)
    return _current_source_lock(dataset)


def _current_tensor_lock(dataset: dict[str, Any]) -> dict[str, str]:
    manifest_path = Path(dataset["feature_provenance_manifest"])
    payload = load_signed_feature_provenance(manifest_path)
    chunks = payload.get("chunks")
    if not isinstance(chunks, dict):
        raise RuntimeError("TCGA-UT provenance chunk inventory is missing")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": compute_sha256(manifest_path),
        "inventory_sha256": compute_data_hash(chunks),
    }


def _current_source_lock(dataset: dict[str, Any]) -> dict[str, str]:
    sidecar_path = Path(dataset["materialization_sidecar"])
    verify_signed_file(sidecar_path)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(sidecar_path),
        "manifest_sha256": compute_sha256(sidecar_path),
        "inventory_sha256": str(payload["sqfs_sha256"]),
    }


def write_prepared_feature_provenance(config: dict[str, Any], data_dir: Path) -> None:
    """Persist the TCGA-UT provenance digests validated during prepare."""
    if lock := current_feature_provenance_lock(config):
        lock_path = data_dir / LOCK_NAME
        write_json(lock_path, lock)
        sign_file(lock_path)


def verify_prepared_feature_provenance(
    config: dict[str, Any], data_dir: Path
) -> dict[str, str] | None:
    """Revalidate prepared TCGA-UT provenance and all tensor hashes."""
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict) or dataset.get("name") != "tcga_ut":
        return None
    lock_path = data_dir / LOCK_NAME
    if not lock_path.is_file():
        raise RuntimeError("Prepared TCGA-UT feature provenance lock is missing")
    verify_signed_file(lock_path)
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(locked, dict):
        raise RuntimeError("Prepared TCGA-UT feature provenance lock is invalid")
    _verify_feature_provenance(config, locked)
    return {
        **{field: str(locked[field]) for field in LOCK_FIELDS},
        "prepared_lock_path": str(lock_path),
        "prepared_lock_sha256": compute_sha256(lock_path),
    }


def verify_frozen_feature_provenance(meta: dict[str, Any]) -> None:
    """Revalidate TCGA-UT tensors against the provenance frozen before fitting."""
    config = meta.get("runtime_config", {})
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    if not isinstance(dataset, dict) or dataset.get("name") != "tcga_ut":
        return
    locked = meta.get("feature_provenance")
    if not isinstance(locked, dict):
        raise RuntimeError("Frozen TCGA-UT feature provenance is missing")
    lock_path = Path(str(locked.get("prepared_lock_path", "")))
    if not lock_path.is_file() or locked.get("prepared_lock_sha256") != compute_sha256(
        lock_path
    ):
        raise RuntimeError("Prepared TCGA-UT feature provenance lock was altered")
    _verify_feature_provenance(config, locked)


def _verify_feature_provenance(
    config: dict[str, Any], locked: dict[str, object]
) -> None:
    try:
        dataset = config["dataset"]
        current = current_feature_provenance_lock(config)
        if current is None or any(
            current[field] != locked.get(field) for field in LOCK_FIELDS
        ):
            raise ValueError("manifest or complete chunk inventory digest changed")
        if dataset.get("regime") == "wsi":
            _verify_tensor_provenance(dataset, current, config)
        else:
            _verify_source_provenance(dataset, locked)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Frozen TCGA-UT feature provenance invalid: {error}"
        ) from error


def _verify_tensor_provenance(
    dataset: dict[str, Any], current: dict[str, str], config: dict[str, Any]
) -> None:
    payload = load_signed_feature_provenance(Path(current["manifest_path"]))
    chunks = payload.get("chunks")
    if not isinstance(chunks, dict):
        raise ValueError("chunk inventory is missing")
    validate_preextracted_features(
        Path(current["manifest_path"]),
        [Path(dataset["feature_dir"]) / name for name in sorted(chunks)],
        resolve_feature_provenance(config.get("feature_extraction", {})),
    )


def _verify_source_provenance(
    dataset: dict[str, Any], locked: dict[str, object]
) -> None:
    """Re-hash the published SqFS itself, not just its signed sidecar."""
    sidecar_path = Path(dataset["materialization_sidecar"])
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sqfs_path = Path(str(payload["sqfs_path"]))
    if compute_sha256(sqfs_path) != locked.get("inventory_sha256"):
        raise ValueError("materialized SqFS content changed since it was frozen")
