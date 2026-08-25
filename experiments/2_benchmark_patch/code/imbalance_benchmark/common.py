from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from collections.abc import Collection
from typing import Any, cast
import numpy as np
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "find_repo_root",
    "load_config",
    "output_root",
    "ensure_dirs",
    "split_paths",
    "split_indices",
    "write_json",
    "compute_sha256",
    "sign_file",
    "verify_signed_file",
    "compute_data_hash",
    "write_run_record",
    "read_run_record",
]


def find_repo_root() -> Path:
    """Walk up from this file to find the git repository root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[4]


REPO_ROOT = find_repo_root()
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "2_benchmark_patch"
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "default.yaml"
N_PATIENT_SPLITS = 3
RUN_RECORD_NAME = "run.json"
EVAL_ARRAYS_NAME = "eval_arrays.npz"
ARRAY_FIELDS = (
    "labels",
    "preds",
    "probabilities",
    "logits",
    "raw_probabilities",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return config


def output_root(config: dict[str, Any]) -> Path:
    """Resolve the configured output root path."""
    configured = Path(
        config.get("paths", {}).get("outputs", "experiments/2_benchmark_patch/outputs")
    )
    return configured if configured.is_absolute() else REPO_ROOT / configured


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create and return all output directories used by the benchmark."""
    root = output_root(config)
    paths = {
        "root": root,
        "data": root / "data",
        "db": root / "results.sqlite",
        "figures": root / "figures",
        "tables": root / "tables",
        "results": root / "results",
        "logs": root / "logs",
    }
    for key, path in paths.items():
        (path.parent if key == "db" else path).mkdir(parents=True, exist_ok=True)
    return paths


def split_paths(paths: dict[str, Path], split_index: int) -> dict[str, Path]:
    """Return an isolated output namespace for one locked patient split."""
    if split_index not in range(N_PATIENT_SPLITS):
        raise ValueError(f"split_index must be in [0, {N_PATIENT_SPLITS - 1}]")
    root = paths["root"] / f"split={split_index}"
    scoped = {
        "root": root,
        "data": root / "data",
        "db": root / "results.sqlite",
        "figures": root / "figures",
        "tables": root / "tables",
        "results": root / "results",
        "logs": root / "logs",
    }
    for key, path in scoped.items():
        (path.parent if key == "db" else path).mkdir(parents=True, exist_ok=True)
    return scoped


def split_indices(split_index: int | None) -> range | tuple[int]:
    """Return every required split for local commands or one SLURM-array split."""
    return range(N_PATIENT_SPLITS) if split_index is None else (split_index,)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable sorting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file for freeze validation."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sign_file(path: Path) -> Path:
    """Write a ``.sha256`` sidecar locking a file's content, and return its path."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(compute_sha256(path), encoding="utf-8")
    return sidecar


def verify_signed_file(path: Path) -> None:
    """Refuse to proceed if a signed file is missing its lock or no longer matches it."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise RuntimeError(
            f"{path.name} has no signed lock; re-run the step that produces it "
            "so it is signed before it is used as frozen evidence."
        )
    if sidecar.read_text(encoding="utf-8").strip() != compute_sha256(path):
        raise RuntimeError(
            f"{path.name} no longer matches its signed lock; "
            "refusing to proceed on altered evidence."
        )


def compute_data_hash(data: dict[str, Any] | list[Any]) -> str:
    """Compute SHA-256 of JSON-serialized data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def dataset_provenance(dataset: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the required dataset provenance frozen with a run."""
    version = dataset.get("version")
    eligibility_rules = dataset.get("eligibility_rules")
    target = dataset.get("target")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("dataset.version is required before definitive freeze")
    if not isinstance(eligibility_rules, dict) or not eligibility_rules:
        raise ValueError(
            "dataset.eligibility_rules are required before definitive freeze"
        )
    if not isinstance(target, str) or not target.strip():
        raise ValueError("dataset.target is required before definitive freeze")
    return {
        "name": dataset.get("name"),
        "regime": dataset.get("regime"),
        "target": target,
        "version": version,
        "eligibility_rules": eligibility_rules,
    }


def write_run_record(
    result_dir: Path, record: dict[str, Any], keep_arrays: bool = True
) -> None:
    """Write run record with evaluator arrays separated into npz sidecar."""
    result_dir.mkdir(parents=True, exist_ok=True)
    slim = dict(record)
    arrays: dict[str, np.ndarray] = {}
    splits = record.get("splits", {})
    slim_splits = {}
    for split, payload in splits.items():
        if not isinstance(payload, dict):
            slim_splits[split] = payload
            continue
        slim_payload = dict(payload)
        for field in ARRAY_FIELDS:
            value = slim_payload.pop(field, None)
            if value is not None:
                arrays[f"{split}_{field}"] = np.asarray(value)
        slim_splits[split] = slim_payload
    slim["splits"] = slim_splits
    if keep_arrays and arrays:
        np.savez_compressed(result_dir / EVAL_ARRAYS_NAME, **cast(Any, arrays))
    write_json(result_dir / RUN_RECORD_NAME, slim)


def _load_requested_arrays(
    record: dict[str, Any], npz_path: Path, requested_fields: set[str]
) -> None:
    """Merge requested sidecar arrays into a loaded run record."""
    with np.load(npz_path, allow_pickle=False) as arrays:
        for split, payload in record.get("splits", {}).items():
            if not isinstance(payload, dict):
                continue
            for field in requested_fields:
                name = f"{split}_{field}"
                if name in arrays:
                    payload[field] = arrays[name].tolist()


def read_run_record(
    result_dir: Path,
    splits: Collection[str] | None = None,
    array_fields: Collection[str] | None = None,
) -> dict[str, Any] | None:
    """Load a run record with optional split and NPZ-array filters."""
    path = result_dir / RUN_RECORD_NAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if splits is not None:
        requested_splits = set(splits)
        record["splits"] = {
            split: payload
            for split, payload in record.get("splits", {}).items()
            if split in requested_splits
        }
    requested_fields = set(ARRAY_FIELDS if array_fields is None else array_fields)
    npz_path = result_dir / EVAL_ARRAYS_NAME
    if requested_fields and npz_path.exists():
        _load_requested_arrays(record, npz_path, requested_fields)
    return record
