from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
import posixpath
import shlex
import subprocess
from typing import Any, Callable

from imbalance_benchmark.common import EXPERIMENT_ROOT, load_config
from imbalance_benchmark.hydra.squashfs import (
    _generated_tile_squashfs,
    _mount_generated_tile_lines,
    _pack_generated_tile_lines,
    _stage_images,
    _staging_lines,
)
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
    sl = config.get("slurm", {})
    sr = sl.get("resources", {}).get(stage, {})
    part = sr.get("partition", sl.get("partition", "gpu-2h" if gpu else "cpu-2h"))
    return {
        "partition": part,
        "gpus": int(sr.get("gpus", 1 if gpu else 0)),
        "cpus": int(sr.get("cpus", 4)),
        "time_limit": str(sr.get("time", "02:00:00")),
    }


def _job(
    config: dict[str, Any], stage: str, cmd: str, gpu: bool, dep: str | None = None
) -> SlurmJob:
    return SlurmJob(stage, cmd, dependency=dep, **_resources(config, stage, gpu))


def build_workflow(config: dict[str, Any], smoke: bool = False) -> list[SlurmJob]:
    """Build the benchmark DAG, or its test-partition synthetic smoke variant."""
    if smoke:
        res = _resources(config, "smoke", True)
        res["partition"] = config.get("slurm", {}).get("test_partition", "gpu-test")
        return [SlurmJob("smoke", "smoke", **res)]
    arr = (0, 1, 2)
    p = _job(config, "prepare", "prepare", True)
    pi = replace(_job(config, "pilot", "pilot", False, p.name), array_splits=arr)
    fr = replace(_job(config, "freeze", "freeze", False, pi.name), array_splits=arr)
    tu = replace(
        _job(config, "tune", "tune", True, fr.name), array_conditions=CONDITIONS
    )
    co = replace(
        _job(config, "confirm", "confirm", True, tu.name),
        array_splits=arr,
        array_conditions=CONDITIONS,
    )
    an = _job(config, "analyze", "analyze", False, co.name)
    return [p, pi, fr, tu, co, an]


def _command(job: SlurmJob, config_path: str | None, code_dir: str) -> str:
    cfg_arg = f" --config {shlex.quote(config_path)}" if config_path else ""
    prefix = f"python {shlex.quote(code_dir)}/__main__.py{cfg_arg}"
    command = f"{prefix} {job.command}"
    if not job.array_splits and not job.array_conditions:
        return command
    lines = []
    if job.array_splits:
        lines.append(f"SPLITS=({' '.join(str(v) for v in job.array_splits)})")
    if job.array_conditions:
        lines.append(
            f"CONDITIONS=({' '.join(shlex.quote(v) for v in job.array_conditions)})"
        )
    if job.array_splits and job.array_conditions:
        cmd_run = f'{prefix} --split-index "$SPLIT_INDEX" {job.command}'
        lines.extend(
            [
                f"N_CONDITIONS={len(job.array_conditions)}",
                'SPLIT_INDEX="${SPLITS[$SLURM_ARRAY_TASK_ID / $N_CONDITIONS]}"',
                'CONDITION="${CONDITIONS[$SLURM_ARRAY_TASK_ID % $N_CONDITIONS]}"',
                f'{cmd_run} --condition "$CONDITION"',
            ]
        )
    elif job.array_splits:
        lines.append(
            f'{prefix} --split-index "${{SPLITS[$SLURM_ARRAY_TASK_ID]}}" {job.command}'
        )
    else:
        lines.append(f'{command} --condition "${{CONDITIONS[$SLURM_ARRAY_TASK_ID]}}"')
    return "\n".join(lines)


def _array_size(job: SlurmJob) -> int:
    return max(1, len(job.array_splits)) * max(1, len(job.array_conditions))


def _directives(job: SlurmJob, root: str) -> list[str]:
    q = shlex.quote(root)
    log_dir = f"{q}/experiments/2_cls_imb_benchmark/outputs/logs"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=imb-{job.name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --gpus-per-node={job.gpus}",
        f"#SBATCH --cpus-per-task={job.cpus}",
        f"#SBATCH --time={job.time_limit}",
        f"#SBATCH --output={log_dir}/%x-%A_%a.out",
        f"#SBATCH --error={log_dir}/%x-%A_%a.err",
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
    dataset_root: str,
    generated_squashfs: tuple[str, str] | None,
) -> list[str]:
    r, c = shlex.quote(root), shlex.quote(code_dir)
    o, co = shlex.quote(output_dir), shlex.quote(container)
    cmd = shlex.quote(command)
    lines = [
        "",
        "set -euo pipefail",
        f"cd {r}",
        f"mkdir -p {o}/logs",
        f"export APPTAINERENV_PYTHONPATH={c}",
        "BINDS=()",
    ]
    lines.extend(_staging_lines(images))
    lines.extend(_mount_generated_tile_lines(generated_squashfs))
    binds = f'-B "{root}:{root}:ro" -B "{output_dir}:{output_dir}:rw" -B "/home/space:/home/space:ro"'
    if dataset_root:
        binds += f' -B "{dataset_root}:{dataset_root}:ro"'
    nv = "--nv " if job.gpus else ""
    app = f'apptainer exec {nv}{binds} "${{BINDS[@]}}" {co} bash -lc {cmd}'
    lines.extend(
        [
            f"if [ -f {co} ]; then",
            f"  {app}",
            "else",
            f"  PYTHONPATH={c}${{PYTHONPATH:+:$PYTHONPATH}} bash -lc {cmd}",
            "fi",
        ]
    )
    lines.extend(_pack_generated_tile_lines(generated_squashfs))
    lines.append("")
    return lines


def _cluster_paths(config: dict[str, Any]) -> tuple[str, str, str, str]:
    sl = config.get("slurm", {})
    r = str(sl.get("project_root", EXPERIMENT_ROOT.parent.parent))
    b = posixpath.join(r, "experiments/2_cls_imb_benchmark")
    c = str(sl.get("code_dir", posixpath.join(b, "code")))
    o = str(sl.get("output_dir", posixpath.join(b, "outputs")))
    co = str(sl.get("container", posixpath.join(b, "environment.sif")))
    return r, c, o, co


def render_sbatch(
    job: SlurmJob, config: dict[str, Any], config_path: str | None = None
) -> str:
    """Render one self-contained, job-ID-logged Apptainer SLURM script."""
    r, c, o, co = _cluster_paths(config)
    cmd = _command(job, config_path, c)
    lines = _directives(job, r)
    images = _stage_images(config, job.name)
    data = (
        str(config.get("dataset", {}).get("root", "")) if job.name == "prepare" else ""
    )
    lines += _execution_lines(
        job, r, c, o, co, cmd, images, data, _generated_tile_squashfs(config, job.name)
    )
    return "\n".join(lines)


def _submit_script(script: str, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    cmd = ["sbatch", "--parsable"]
    res = subprocess.run(cmd, input=script, text=True, check=True, capture_output=True)
    return res.stdout.strip().split(";", maxsplit=1)[0]


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
        dep = submitted.get(job.dependency) if job.dependency else None
        scheduled = replace(job, dependency=f"afterok:{dep}" if dep else None)
        script = render_sbatch(scheduled, config, config_path)
        jid = f"dry-run-{job.name}" if dry_run else submit(script, False)
        submitted[job.name] = jid
        logger.info("%s: %s", job.name, jid)
        if dry_run:
            logger.info("%s", script)
    return submitted


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit the Hydra workflow."""
    config = load_config(args.config)
    submit_workflow(config, args.config, args.dry_run, getattr(args, "smoke", False))
