import argparse
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    """A command plus its scheduler metadata."""

    cmd: list[str]
    name: str
    log_path: str
    partition: str = "cpu-2h"
    gpus_per_node: int = 0


def load_config(config_path: str) -> dict[str, str]:
    """Load the JSON job configuration."""
    try:
        with open(config_path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.warning("Konfigurationsdatei %s nicht gefunden.", config_path)
        return {}


def execute(job: Job, config: dict[str, str], local: bool, dry_run: bool) -> None:
    """Execute one job locally or submit it to SLURM."""
    if local:
        _run_local(job.cmd, dry_run)
        return
    _submit_slurm(job, config, dry_run)


def pythonpath_env(config: dict[str, str]) -> str:
    """Return PYTHONPATH for the shared common_code package and design_dataset."""
    working = Path(config.get("working_dir", ".")).resolve()
    common = working.parent.parent / "shared"
    return f"{common}{os.pathsep}{working}"


def prefix(config: dict[str, str], args: argparse.Namespace) -> list[str]:
    """Build the Python execution prefix."""
    pythonpath = pythonpath_env(config)
    if args.local and args.no_container:
        os.environ["PYTHONPATH"] = pythonpath
        return ["python3"]
    return [
        "apptainer",
        "run",
        "-B",
        "/home/space:/home/space:rw",
        "--env",
        f"PYTHONPATH={pythonpath}",
        config.get("environment_sif", ""),
        "python3",
    ]


def parameters(args: argparse.Namespace) -> list[float]:
    """Return configured imbalance parameters."""
    return (
        [round(index / 10.0, 1) for index in range(14)]
        if args.sweep
        else [args.parameter]
    )


def train_base(config: dict[str, str], ds: str, val: str, out: str) -> list[str]:
    """Build common training command arguments."""
    args = [
        "-m",
        "tcga_ut_imbalanced.cli.train",
        f"--dataset-structure-path={ds}",
        f"--validation-dataset-structure-path={val}",
        f"--feature-path={config.get('feature_path', '')}",
        "--preload-features",
        f"--results-save-path={out}",
    ]
    cache_path = config.get("feature_cache_path", "")
    if cache_path:
        args.append(f"--feature-cache-path={cache_path}")
    return args


def train_csvs(imbalanced_dir: str, parameter: float) -> tuple[str, str]:
    """Return train and validation CSV paths for one parameter."""
    stem = f"{imbalanced_dir}/TCGA-UT_imbalanced_parameter={parameter}_dataset_size=500_seed=0"
    return f"{stem}/imbalanced_dataset.csv", f"{stem}/validation_dataset.csv"


def _run_local(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        logging.info("Dry-run (lokal): %s", " ".join(cmd))
        return
    logging.info("Führe lokal aus: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _submit_slurm(job: Job, config: dict[str, str], dry_run: bool) -> None:
    script = _slurm_script(job, config.get("working_dir", ""))
    if dry_run:
        logging.info("Dry-run (SLURM %s):\n%s", job.name, script)
        return
    logging.info("Reiche Job %s bei SLURM ein...", job.name)
    subprocess.run(["sbatch"], input=script, text=True, check=True)


def _slurm_script(job: Job, working_dir: str) -> str:
    command = " ".join(shlex.quote(part) for part in job.cmd)
    workdir = shlex.quote(working_dir) if working_dir else '""'
    return (
        f"#!/bin/bash\n#SBATCH --job-name={job.name}\n"
        f"#SBATCH --partition={job.partition}\n"
        f"#SBATCH --gpus-per-node={job.gpus_per_node}\n"
        f"#SBATCH --ntasks-per-node=8\n"
        f"#SBATCH --output={job.log_path}\n#SBATCH -D {workdir}\n\n"
        f"{command}\n"
    )
