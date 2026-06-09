from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RUN_RECORD_NAME = "run.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load experiment configuration from YAML."""
    config_path = Path(path) if path else EXPERIMENT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return config


def output_root(config: dict[str, Any]) -> Path:
    """Resolve the configured output root path."""
    configured = Path(config["paths"]["outputs"])
    return (
        configured
        if configured.is_absolute()
        else EXPERIMENT_ROOT.parents[1] / configured
    )


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Create and return all output directories used by the experiment."""
    root = output_root(config)
    artifacts_root = root / "outputs"
    paths = {
        "root": root,
        "data": root / "data",
        "db": artifacts_root / "results.sqlite",
        "figures": artifacts_root / "figures",
        "tables": artifacts_root / "tables",
        "results": artifacts_root / "results",
        "patch_results": artifacts_root / "results_patch",
        "wsi_results": artifacts_root / "results_wsi_bag",
        "logs": root / "logs",
    }
    for key, path in paths.items():
        target = path.parent if key == "db" else path
        target.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    """Write progress payload with an update timestamp."""
    payload["updated_unix"] = time.time()
    write_json(path, payload)


def write_run_record(result_dir: Path, record: dict[str, Any]) -> None:
    """Write the consolidated per-run record used by aggregate ingestion."""
    write_json(result_dir / RUN_RECORD_NAME, record)


def read_run_record(result_dir: Path) -> dict[str, Any] | None:
    """Load a run record, falling back to legacy per-split JSON files."""
    path = result_dir / RUN_RECORD_NAME
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Run record must be a mapping: {path}")
    return _read_legacy_run_record(result_dir)


def _read_legacy_run_record(result_dir: Path) -> dict[str, Any] | None:
    config_path = result_dir / "config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    splits: dict[str, Any] = {}
    for split in ("val", "test"):
        split_path = result_dir / f"{split}_results.json"
        if split_path.exists():
            with split_path.open("r", encoding="utf-8") as handle:
                splits[split] = json.load(handle)
    if not splits:
        return None
    diagnostics_path = result_dir / "activation_diagnostics.json"
    diagnostics = None
    if diagnostics_path.exists():
        with diagnostics_path.open("r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
    model_path = "model.pt" if (result_dir / "model.pt").exists() else None
    return {
        "benchmark": config.get("benchmark", "unknown"),
        "method": config.get("method", "unknown"),
        "seed": config.get("seed"),
        "smoke": config.get("smoke", False),
        "tuning_id": config.get("tuning_id"),
        "tuning_params": config.get("tuning_params", {}),
        "model_path": model_path,
        "method_metadata": config.get("method_metadata"),
        "class_names": config.get("class_names"),
        "deterministic": config.get("deterministic"),
        "diagnostics": diagnostics,
        "splits": splits,
    }
