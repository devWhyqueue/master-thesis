from __future__ import annotations

from dataclasses import dataclass
import posixpath
import shlex
from typing import Any

from imbalance_benchmark.common import EXPERIMENT_ROOT
from imbalance_benchmark.hydra.squashfs import (
    _generated_tile_squashfs,
    _mount_generated_tile_lines,
    _pack_generated_tile_lines,
    _stage_images,
    _staging_lines,
    shared_squashfs,
    sharded_staging_lines,
)


@dataclass(frozen=True)
class SlurmJob:
    """One benchmark command and its SLURM resource requirements."""

    name: str
    command: str
    partition: str
    gpus: int
    cpus: int
    memory: str | None = None
    time_limit: str | None = None
    constraint: str | None = None
    dependencies: tuple[str, ...] = ()
    array_splits: tuple[int, ...] = ()
    array_conditions: tuple[str, ...] = ()
    array_size: int = 0
    array_indices: tuple[int, ...] = ()
    on_host: bool = False


def render_sbatch(
    job: SlurmJob, config: dict[str, Any], config_path: str | None = None
) -> str:
    """Render one self-contained, job-ID-logged Apptainer SLURM script."""
    root, code, output, container = _cluster_paths(config)
    command = _command(job, config_path, code)
    lines = _directives(job, root, config)
    images = _stage_images(config, job.name)
    dataset = _prepare_dataset(config, job.name)
    lines += (
        _host_execution_lines(root, command)
        if job.on_host
        else _execution_lines(
            job,
            root,
            code,
            output,
            container,
            command,
            images,
            shared_squashfs(config, job.name),
            sharded_staging_lines(config, job.name),
            dataset,
            _generated_tile_squashfs(config, job.name),
        )
    )
    return "\n".join(lines)


def _command(job: SlurmJob, config_path: str | None, code_dir: str) -> str:
    config = f" --config {shlex.quote(config_path)}" if config_path else ""
    prefix = f"python {shlex.quote(code_dir)}/__main__.py{config}"
    command = f"{prefix} {job.command}"
    if job.array_size:
        return f'{command} --shard-index "$SLURM_ARRAY_TASK_ID"'
    if not job.array_splits and not job.array_conditions:
        return command
    lines = []
    if job.array_splits:
        lines.append(f"SPLITS=({' '.join(str(value) for value in job.array_splits)})")
    if job.array_conditions:
        conditions = " ".join(shlex.quote(value) for value in job.array_conditions)
        lines.append(f"CONDITIONS=({conditions})")
    if job.array_splits and job.array_conditions:
        lines.extend(_crossed_array_lines(job, prefix))
    elif job.array_splits:
        lines.append(
            f'{prefix} --split-index "${{SPLITS[$SLURM_ARRAY_TASK_ID]}}" {job.command}'
        )
    else:
        lines.append(f'{command} --condition "${{CONDITIONS[$SLURM_ARRAY_TASK_ID]}}"')
    return "\n".join(lines)


def _prepare_dataset(config: dict[str, Any], stage: str) -> str:
    return str(config.get("dataset", {}).get("root", "")) if stage == "prepare" else ""


def _crossed_array_lines(job: SlurmJob, prefix: str) -> list[str]:
    run = f'{prefix} --split-index "$SPLIT_INDEX" {job.command}'
    return [
        f"N_CONDITIONS={len(job.array_conditions)}",
        'SPLIT_INDEX="${SPLITS[$SLURM_ARRAY_TASK_ID / $N_CONDITIONS]}"',
        'CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID % $N_CONDITIONS]}"',
        f'{run} --condition "$CONDITION"',
    ]


def _directives(job: SlurmJob, root: str, config: dict[str, Any]) -> list[str]:
    log_dir = f"{shlex.quote(root)}/experiments/2_cls_imb_benchmark/outputs/logs"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=imb-{job.name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --gpus-per-node={job.gpus}",
        f"#SBATCH --cpus-per-task={job.cpus}",
        f"#SBATCH --output={log_dir}/%x-%A_%a.out",
        f"#SBATCH --error={log_dir}/%x-%A_%a.err",
    ]
    if job.time_limit:
        lines.append(f"#SBATCH --time={job.time_limit}")
    if job.memory:
        lines.append(f"#SBATCH --mem={job.memory}")
    if job.constraint:
        lines.append(f"#SBATCH --constraint={job.constraint}")
    array_size = job.array_size or (
        max(1, len(job.array_splits)) * max(1, len(job.array_conditions))
    )
    if job.array_size or job.array_splits or job.array_conditions:
        indices = (
            ",".join(str(index) for index in job.array_indices)
            if job.array_indices
            else f"0-{array_size - 1}"
        )
        concurrency = config.get("slurm", {}).get("max_array_concurrency")
        throttle = f"%{int(concurrency)}" if concurrency else ""
        lines.append(f"#SBATCH --array={indices}{throttle}")
    if job.dependencies:
        lines.append(f"#SBATCH --dependency=afterok:{':'.join(job.dependencies)}")
    return lines


def _execution_lines(
    job: SlurmJob,
    root: str,
    code_dir: str,
    output_dir: str,
    container: str,
    command: str,
    images: list[tuple[str, str]],
    shared_images: list[tuple[str, str]],
    sharded_images: list[str],
    dataset_root: str,
    generated_squashfs: tuple[str, str] | None,
) -> list[str]:
    quoted_root, quoted_code = shlex.quote(root), shlex.quote(code_dir)
    quoted_output, quoted_container = shlex.quote(output_dir), shlex.quote(container)
    quoted_command = shlex.quote(command)
    lines = [
        "",
        "set -euo pipefail",
        f"cd {quoted_root}",
        f"mkdir -p {quoted_output}/logs",
        f"export APPTAINERENV_PYTHONPATH={quoted_code}",
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
        'export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"',
        "BINDS=()",
    ]
    lines.extend(_staging_lines(images))
    lines.extend(sharded_images)
    lines.extend(_mount_generated_tile_lines(generated_squashfs))
    binds = f'-B "{root}:{root}:ro" -B "{output_dir}:{output_dir}:rw" -B "/home/space:/home/space:ro"'
    if dataset_root:
        binds += f' -B "{dataset_root}:{dataset_root}:ro"'
    for source, mount in shared_images:
        binds += f' -B "{source}:{mount}:image-src=/"'
    gpu = "--nv " if job.gpus else ""
    app = (
        f'apptainer exec {gpu}{binds} "${{BINDS[@]}}" '
        f"{quoted_container} bash -lc {quoted_command}"
    )
    lines.extend(
        [
            f"if [ -f {quoted_container} ]; then",
            f"  {app}",
            "else",
            f"  PYTHONPATH={quoted_code}${{PYTHONPATH:+:$PYTHONPATH}} bash -lc {quoted_command}",
            "fi",
        ]
    )
    lines.extend(_pack_generated_tile_lines(generated_squashfs))
    lines.append("")
    return lines


def _host_execution_lines(root: str, command: str) -> list[str]:
    """Run a job on the login/compute-node host, outside Apptainer.

    Self-chaining stages (``tune-decide``) shell out to ``sbatch``/``squeue``
    themselves - the Apptainer container has no SLURM client, so those
    calls must run on the host. ``uv run`` (rather than the container)
    supplies the same pinned dependencies.
    """
    quoted_root = shlex.quote(root)
    quoted_command = shlex.quote(f"uv run {command}")
    return [
        "",
        "set -euo pipefail",
        f"cd {quoted_root}",
        f"bash -lc {quoted_command}",
        "",
    ]


def _cluster_paths(config: dict[str, Any]) -> tuple[str, str, str, str]:
    slurm = config.get("slurm", {})
    root = str(slurm.get("project_root", EXPERIMENT_ROOT.parent.parent))
    benchmark = posixpath.join(root, "experiments/2_cls_imb_benchmark")
    code = str(slurm.get("code_dir", posixpath.join(benchmark, "code")))
    output = str(slurm.get("output_dir", posixpath.join(benchmark, "outputs")))
    container = str(
        slurm.get("container", posixpath.join(benchmark, "environment.sif"))
    )
    return root, code, output, container
