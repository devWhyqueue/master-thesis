import argparse
import dataclasses
import json
import logging
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


@dataclass(frozen=True)
class Job:
    """A command plus its scheduler metadata."""

    cmd: list[str]
    name: str
    log_path: str
    partition: str = "cpu-2h"
    gpus_per_node: int = 0
    array_spec: str | None = None
    cpus_per_task: int = 4
    time_limit: str | None = None
    mem: str | None = None
    dependency: str | None = None


def load_config(config_path: str) -> dict[str, str]:
    """Load the JSON job configuration."""
    try:
        with open(config_path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.warning("Konfigurationsdatei %s nicht gefunden.", config_path)
        return {}


def execute(job: Job, config: dict[str, str], local: bool, dry_run: bool) -> str | None:
    """Execute one job locally or submit it to SLURM; return SLURM job ID if applicable."""
    if local:
        _run_local(job, config, dry_run)
        return None
    return _submit_slurm(job, config, dry_run)


def execute_progan_pipeline(
    shard_jobs: list[Job],
    finalize_job: Job,
    config: dict[str, str],
    local: bool,
    dry_run: bool,
) -> None:
    """Submit N shard jobs then a finalize job that depends on all of them."""
    job_ids = [execute(j, config, local, dry_run) for j in shard_jobs]
    real_ids = [jid for jid in job_ids if jid]
    if real_ids:
        dep = "afterok:" + ":".join(real_ids)
        finalize_job = dataclasses.replace(finalize_job, dependency=dep)
    execute(finalize_job, config, local, dry_run)


def pythonpath_env(config: dict[str, str]) -> str:
    """Return PYTHONPATH for shared code, class_imbalance scripts, and design_dataset."""
    working = Path(config.get("working_dir", ".")).resolve()
    experiments_root = working.parent.parent
    common = experiments_root / "shared"
    class_imbalance = experiments_root / "class_imbalance"
    return f"{common}{os.pathsep}{class_imbalance}{os.pathsep}{working}"


def prefix(
    config: dict[str, str],
    args: argparse.Namespace,
    *,
    gpu: bool = False,
) -> list[str]:
    """Build the Python execution prefix."""
    pythonpath = pythonpath_env(config)
    if args.local and args.no_container:
        os.environ["PYTHONPATH"] = pythonpath
        return ["python3"]
    os.environ["APPTAINERENV_PYTHONPATH"] = pythonpath
    command = [
        "apptainer",
        "run",
    ]
    if gpu:
        command.append("--nv")
    command.extend(
        [
            "-B",
            "/home/space:/home/space:rw",
            config.get("environment_sif", ""),
            "python3",
        ]
    )
    return command


def parameters(args: argparse.Namespace) -> list[float]:
    """Return configured imbalance parameters."""
    return (
        [round(index / 10.0, 1) for index in range(14)]
        if args.sweep
        else [args.parameter]
    )


def train_base(config: dict[str, str], ds: str, val: str, out: str) -> list[str]:
    """Build common training command arguments."""
    cmd = [
        "-m",
        "cli.train",
        f"--dataset-structure-path={ds}",
        f"--validation-dataset-structure-path={val}",
        f"--feature-path={config.get('feature_path', '')}",
        "--preload-features",
        f"--results-save-path={out}",
    ]
    cache_path = config.get("feature_cache_path", "")
    if cache_path:
        cmd.append(f"--feature-cache-path={cache_path}")
    return cmd


def train_csvs(imbalanced_dir: str, parameter: float) -> tuple[str, str]:
    """Return train and validation CSV paths for one parameter."""
    stem = f"{imbalanced_dir}/TCGA-UT_imbalanced_parameter={parameter}_dataset_size=500_seed=0"
    return f"{stem}/imbalanced_dataset.csv", f"{stem}/validation_dataset.csv"


def _run_local(job: Job, config: dict[str, str], dry_run: bool) -> None:
    if dry_run:
        logging.info("Dry-run (lokal): %s", " ".join(job.cmd))
        return
    os.environ["APPTAINERENV_PYTHONPATH"] = pythonpath_env(config)
    logging.info("Führe lokal aus: %s", " ".join(job.cmd))
    subprocess.run(job.cmd, check=True)


def _submit_slurm(job: Job, config: dict[str, str], dry_run: bool) -> str | None:
    script = _slurm_script(job, config)
    if dry_run:
        logging.info("Dry-run (SLURM %s):\n%s", job.name, script)
        return None
    logging.info("Reiche Job %s bei SLURM ein...", job.name)
    result = subprocess.run(
        ["sbatch"], input=script, text=True, check=True, capture_output=True
    )
    match = _SBATCH_JOB_ID_RE.search(result.stdout)
    return match.group(1) if match else None


def _slurm_script(job: Job, config: dict[str, str]) -> str:
    command = " ".join(shlex.quote(part) for part in job.cmd)
    working_dir = config.get("working_dir", "")
    workdir = shlex.quote(working_dir) if working_dir else '""'
    pythonpath = shlex.quote(pythonpath_env(config))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --gpus-per-node={job.gpus_per_node}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={job.cpus_per_task}",
        f"#SBATCH --output={job.log_path}",
        f"#SBATCH -D {workdir}",
    ]
    if job.time_limit:
        lines.append(f"#SBATCH --time={job.time_limit}")
    if job.mem:
        lines.append(f"#SBATCH --mem={job.mem}")
    if job.array_spec:
        lines.append(f"#SBATCH --array={job.array_spec}")
    if job.dependency:
        lines.append(f"#SBATCH --dependency={job.dependency}")
    lines.extend(
        [
            "",
            f"export APPTAINERENV_PYTHONPATH={pythonpath}",
            command,
            "",
        ]
    )
    return "\n".join(lines)
