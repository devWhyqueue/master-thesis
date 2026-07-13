"""Render and submit the dependency-linked Hydra benchmark workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
import posixpath
import shlex
import subprocess
from typing import Any, Callable

from imbalance_benchmark.common import EXPERIMENT_ROOT, load_config
from imbalance_benchmark.modeling.context import CONDITIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlurmJob:
    """One benchmark command and its SLURM resource requirements."""

    name: str
    command: str
    partition: str
    gpus: int
    cpus: int
    time_limit: str
    dependency: str | None = None
    array_splits: tuple[int, ...] = ()
    array_conditions: tuple[str, ...] = ()


def _resources(config: dict[str, Any], stage: str, gpu: bool) -> dict[str, Any]:
    """Return resource settings for a workflow stage with safe defaults."""
    slurm = config.get("slurm", {})
    stage_resources = slurm.get("resources", {}).get(stage, {})
    return {
        "partition": stage_resources.get(
            "partition", slurm.get("partition", "gpu-2h" if gpu else "cpu-2h")
        ),
        "gpus": int(stage_resources.get("gpus", 1 if gpu else 0)),
        "cpus": int(stage_resources.get("cpus", 4)),
        "time_limit": str(stage_resources.get("time", "02:00:00")),
    }


def _job(config: dict[str, Any], stage: str, command: str, gpu: bool) -> SlurmJob:
    """Create a stage job using the configured SLURM resources."""
    return SlurmJob(stage, command, **_resources(config, stage, gpu))


def build_workflow(config: dict[str, Any], smoke: bool = False) -> list[SlurmJob]:
    """Build the benchmark DAG, or its test-partition synthetic smoke variant."""
    if smoke:
        resources = _resources(config, "smoke", True)
        resources["partition"] = config.get("slurm", {}).get(
            "test_partition", "gpu-test"
        )
        return [SlurmJob("smoke", "smoke", **resources)]
    split_array = (0, 1, 2)
    prepare = replace(_job(config, "prepare", "prepare", gpu=True), array_splits=split_array)
    pilot = replace(_job(config, "pilot", "pilot", gpu=False), array_splits=split_array)
    freeze = replace(_job(config, "freeze", "freeze", gpu=False), array_splits=split_array)
    tune = _job(config, "tune", "tune", gpu=True)
    confirm = replace(_job(config, "confirm", "confirm", gpu=True), array_splits=split_array)
    analyze = _job(config, "analyze", "analyze", gpu=False)
    return [
        prepare,
        replace(pilot, dependency=prepare.name),
        replace(freeze, dependency=pilot.name),
        replace(tune, dependency=freeze.name, array_conditions=CONDITIONS),
        replace(confirm, dependency=tune.name, array_conditions=CONDITIONS),
        replace(analyze, dependency=confirm.name),
    ]


def _config_argument(config_path: str | None) -> str:
    """Return the optional CLI fragment selecting a non-default configuration."""
    return f" --config {shlex.quote(config_path)}" if config_path else ""


def _stage_images(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Validate configured SquashFS source/mount pairs before rendering a job."""
    images = config.get("slurm", {}).get("squashfs", [])
    pairs = []
    for image in images:
        if not isinstance(image, dict) or not {"source", "mount"} <= image.keys():
            raise ValueError("Each slurm.squashfs entry needs source and mount fields")
        pairs.append((str(image["source"]), str(image["mount"])))
    return pairs


def _staging_lines(images: list[tuple[str, str]]) -> list[str]:
    """Stage SquashFS images to job-local storage and bind their requested mounts."""
    if not images:
        return []
    lines = [
        'STAGE_DIR="/tmp/imbalance-benchmark-${SLURM_JOB_ID}"',
        'mkdir -p "$STAGE_DIR"',
    ]
    for index, (source, mount) in enumerate(images):
        local = f'"$STAGE_DIR/{index}.sqfs"'
        lines.extend(
            [
                f"cp {shlex.quote(source)} {local}",
                f'BINDS+=("{local}:{mount}:image-src=/")',
            ]
        )
    return lines


def _command(job: SlurmJob, config_path: str | None, code_dir: str) -> str:
    """Build the benchmark command, mapping each array task to one condition."""
    prefix = f"python {shlex.quote(code_dir)}/__main__.py{_config_argument(config_path)}"
    command = f"{prefix} {job.command}"
    if not job.array_splits and not job.array_conditions:
        return command
    lines = []
    if job.array_splits:
        values = " ".join(str(value) for value in job.array_splits)
        lines.append(f"SPLITS=({values})")
    if job.array_conditions:
        values = " ".join(shlex.quote(value) for value in job.array_conditions)
        lines.append(f"CONDITIONS=({values})")
    if job.array_splits and job.array_conditions:
        lines.extend(
            [
                f"N_CONDITIONS={len(job.array_conditions)}",
                'SPLIT_INDEX="${SPLITS[$SLURM_ARRAY_TASK_ID / $N_CONDITIONS]}"',
                'CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID % $N_CONDITIONS]}"',
                f'{prefix} --split-index "$SPLIT_INDEX" {job.command} --condition "$CONDITION"',
            ]
        )
    elif job.array_splits:
        lines.append(f'{prefix} --split-index "${{SPLITS[$SLURM_ARRAY_TASK_ID]}}" {job.command}')
    else:
        lines.append(f'{command} --condition "${{CONDITIONS[$SLURM_ARRAY_TASK_ID]}}"')
    return "\n".join(lines)


def _array_size(job: SlurmJob) -> int:
    """Return the number of scheduler tasks needed for a split/condition grid."""
    return max(1, len(job.array_splits)) * max(1, len(job.array_conditions))


def _directives(job: SlurmJob, root: str) -> list[str]:
    """Render scheduler directives, including array and dependency metadata."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=imb-{job.name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --gpus-per-node={job.gpus}",
        f"#SBATCH --cpus-per-task={job.cpus}",
        f"#SBATCH --time={job.time_limit}",
        f"#SBATCH --output={shlex.quote(root)}/experiments/2_cls_imb_benchmark/outputs/logs/%x-%A_%a.out",
        f"#SBATCH --error={shlex.quote(root)}/experiments/2_cls_imb_benchmark/outputs/logs/%x-%A_%a.err",
    ]
    if job.array_splits or job.array_conditions:
        lines.append(f"#SBATCH --array=0-{_array_size(job) - 1}")
    if job.dependency:
        lines.append(f"#SBATCH --dependency={job.dependency}")
    return lines


def _execution_lines(
    job: SlurmJob,
    root: str,
    code_dir: str,
    output_dir: str,
    container: str,
    command: str,
    images: list[tuple[str, str]],
) -> list[str]:
    """Render the container and local fallback execution blocks."""
    lines = [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(root)}",
        f"mkdir -p {shlex.quote(output_dir)}/logs",
        f"export APPTAINERENV_PYTHONPATH={shlex.quote(code_dir)}",
        "BINDS=()",
    ]
    lines.extend(_staging_lines(images))
    apptainer = f'apptainer exec {"--nv " if job.gpus else ""}-B "{root}:{root}:ro" -B "{output_dir}:{output_dir}:rw" "${{BINDS[@]}}" {shlex.quote(container)} bash -lc {shlex.quote(command)}'
    lines.extend(
        [
            f"if [ -f {shlex.quote(container)} ]; then",
            f"  {apptainer}",
            "else",
            f"  PYTHONPATH={shlex.quote(code_dir)}${{PYTHONPATH:+:$PYTHONPATH}} bash -lc {shlex.quote(command)}",
            "fi",
            "",
        ]
    )
    return lines


def _cluster_paths(config: dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve POSIX paths used by jobs running on the Hydra cluster."""
    slurm = config.get("slurm", {})
    root = str(slurm.get("project_root", EXPERIMENT_ROOT.parent.parent))
    benchmark = posixpath.join(root, "experiments/2_cls_imb_benchmark")
    code_dir = str(slurm.get("code_dir", posixpath.join(benchmark, "code")))
    output_dir = str(slurm.get("output_dir", posixpath.join(benchmark, "outputs")))
    container = str(
        slurm.get("container", posixpath.join(benchmark, "environment.sif"))
    )
    return root, code_dir, output_dir, container


def render_sbatch(
    job: SlurmJob, config: dict[str, Any], config_path: str | None = None
) -> str:
    """Render one self-contained, job-ID-logged Apptainer SLURM script."""
    root, code_dir, output_dir, container = _cluster_paths(config)
    lines = _directives(job, root)
    lines.extend(
        _execution_lines(
            job,
            root,
            code_dir,
            output_dir,
            container,
            _command(job, config_path, code_dir),
            _stage_images(config),
        )
    )
    return "\n".join(lines)


def _submit_script(script: str, dry_run: bool) -> str:
    """Submit a rendered script, or return a deterministic dry-run job identifier."""
    if dry_run:
        return "dry-run"
    result = subprocess.run(
        ["sbatch", "--parsable"],
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    return result.stdout.strip().split(";", maxsplit=1)[0]


def submit_workflow(
    config: dict[str, Any],
    config_path: str | None = None,
    dry_run: bool = False,
    smoke: bool = False,
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit the workflow in topological order, returning job IDs by stage."""
    submitted: dict[str, str] = {}
    for job in build_workflow(config, smoke):
        dependency = submitted.get(job.dependency) if job.dependency else None
        scheduled = replace(
            job, dependency=f"afterok:{dependency}" if dependency else None
        )
        script = render_sbatch(scheduled, config, config_path)
        job_id = f"dry-run-{job.name}" if dry_run else submit(script, False)
        submitted[job.name] = job_id
        logger.info("%s: %s", job.name, job_id)
        if dry_run:
            logger.info("%s", script)
    return submitted


def cmd_submit(args: argparse.Namespace) -> None:
    """Render or submit the full Hydra workflow from the selected configuration."""
    config = load_config(args.config)
    submit_workflow(config, args.config, args.dry_run, getattr(args, "smoke", False))
