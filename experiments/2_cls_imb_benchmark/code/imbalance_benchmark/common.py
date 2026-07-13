from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, cast
import numpy as np
import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "find_repo_root",
    "load_config",
    "output_root",
    "ensure_dirs",
    "write_json",
    "compute_sha256",
    "compute_data_hash",
    "write_run_record",
    "read_run_record",
    "render_sbatch",
    "submit_sbatch",
    "cmd_submit",
]


def find_repo_root() -> Path:
    """Walk up from this file to find the git repository root."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[4]


REPO_ROOT = find_repo_root()
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "2_cls_imb_benchmark"
DEFAULT_CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "default.yaml"
RUN_RECORD_NAME = "run.json"
EVAL_ARRAYS_NAME = "eval_arrays.npz"
ARRAY_FIELDS = (
    "labels",
    "preds",
    "probabilities",
    "logits",
    "raw_logits",
    "raw_probabilities",
    "balanced_decision_logits",
    "target_prior_logits",
    "target_prior_probabilities",
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
        config.get("paths", {}).get(
            "outputs", "experiments/2_cls_imb_benchmark/outputs"
        )
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


def compute_data_hash(data: dict[str, Any] | list[Any]) -> str:
    """Compute SHA-256 of JSON-serialized data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


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


def read_run_record(result_dir: Path) -> dict[str, Any] | None:
    """Load a run record and merge sidecar arrays if present."""
    path = result_dir / RUN_RECORD_NAME
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    npz_path = result_dir / EVAL_ARRAYS_NAME
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as arrays:
            for split, payload in record.get("splits", {}).items():
                if not isinstance(payload, dict):
                    continue
                for f in ARRAY_FIELDS:
                    if f"{split}_{f}" in arrays:
                        payload[f] = arrays[f"{split}_{f}"].tolist()
    return record


def render_sbatch(config: dict[str, Any], job_name: str, python_command: str) -> str:
    """Render a standard SLURM SBATCH script based on config."""
    sl_cfg = config.get("slurm", {})
    partition = sl_cfg.get("partition", "gpu-2h")
    container = sl_cfg.get("container", "./environment.sif")
    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={partition}",
            "#SBATCH --output=logs/%x-%j.out",
            "#SBATCH --error=logs/%x-%j.err",
            "",
            "set -euo pipefail",
            "mkdir -p logs",
            "",
            f'CONTAINER="{container}"',
            f'COMMAND="python -m imbalance_benchmark {python_command}"',
            "",
            'if [ -f "$CONTAINER" ]; then',
            '  apptainer exec --nv "$CONTAINER" $COMMAND',
            "else",
            "  eval $COMMAND",
            "fi\n",
        ]
    )


def submit_sbatch(sbatch_content: str, dry_run: bool = False) -> str | None:
    """Save the SBATCH script and submit to sbatch. Return job ID."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    script_path = logs_dir / "temp_job.sbatch"
    script_path.write_text(sbatch_content, encoding="utf-8")
    if dry_run:
        logger.info("=== Dry-run SBATCH Script ===\n%s", sbatch_content)
        return "dry-run-job-id"
    try:
        res = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=True
        )
        output = res.stdout.strip()
        logger.info("Submitted job: %s", output)
        return output.split()[-1] if output.split() else None
    except Exception as e:
        logger.warning(
            "Failed to submit to SLURM (sbatch not available or error): %s", e
        )
    return None


def cmd_submit(args: argparse.Namespace) -> None:
    """Generate/submit SLURM jobs."""
    config = load_config(args.config)
    submit_sbatch(render_sbatch(config, "imbalance_job", "tune"), dry_run=args.dry_run)
