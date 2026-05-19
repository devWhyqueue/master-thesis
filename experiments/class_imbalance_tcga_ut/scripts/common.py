from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None) -> dict[str, Any]:
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
        "figures": artifacts_root / "figures",
        "tables": artifacts_root / "tables",
        "results": artifacts_root / "results",
        "patch_results": artifacts_root / "results_patch",
        "wsi_results": artifacts_root / "results_wsi_bag",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
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
