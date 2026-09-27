"""Command-line entry point for the mitigation-methods experiment (exp-40)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config
from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.rendering import SlurmJob

from breadth.slurm import submit_workflow

from analyze import run_analyze

from mitigation.fit import decode_shard_index, run_fit_stage1, shard_count
from mitigation.fit_stage2 import run_fit_stage2
from mitigation.grid import Stage1Job, stage1_jobs

__all__ = ["main"]

logger = logging.getLogger(__name__)

_SUBMIT_STAGES = ("fit", "stage2", "analyze")


def _fit_job(config: dict, job: Stage1Job) -> SlurmJob:
    """One stage-one grid combination as a full-shard SLURM array.

    ``job.name`` stays the literal stage key ``"fit"`` (not a per-combo name):
    ``render_sbatch`` keys shared-squashfs mounts and data binds off ``job.name``
    against the config's ``stages: [fit]`` lists, so a per-combo name would
    silently drop those mounts for every array but the first one.
    """
    command = f"fit --method {job.method} --arm {job.arm}"
    if job.param is not None:
        command += f" --param {job.param}"
    return replace(
        build_job(config, "fit", command, True, resource="fit"),
        array_size=shard_count(),
    )


def _stage_jobs(config: dict, stage: str) -> list[SlurmJob]:
    """Build one submission stage's jobs: every grid combo for ``fit``, one array for ``stage2``."""
    if stage == "fit":
        return [_fit_job(config, job) for job in stage1_jobs(config)]
    if stage == "stage2":
        stage2_job = build_job(config, "stage2", "stage2", False, resource="stage2")
        return [replace(stage2_job, array_size=shard_count())]
    if stage == "analyze":
        return [build_job(config, "analyze", "analyze", False, resource="analyze")]
    raise ValueError(f"unknown submission stage: {stage}")


def _parser() -> argparse.ArgumentParser:
    """Create the mitigation-methods command-line argument parser."""
    parser = argparse.ArgumentParser(description="Mitigation Methods Experiment CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--shard-index", type=int, required=True)
    fit.add_argument("--method", required=True)
    fit.add_argument("--param", type=float, default=None)
    fit.add_argument("--arm", required=True)
    stage2 = sub.add_parser("stage2")
    stage2.add_argument("--shard-index", type=int, required=True)
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_SUBMIT_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_fit(args: argparse.Namespace) -> None:
    """Train and record one stage-one arm of one (split, draw) shard."""
    split_idx, draw_idx = decode_shard_index(args.shard_index)
    run_fit_stage1(
        load_config(args.config), split_idx, draw_idx, args.arm, args.method, args.param
    )


def cmd_stage2(args: argparse.Namespace) -> None:
    """Fit every pending stage-two method of one (split, draw) shard."""
    split_idx, draw_idx = decode_shard_index(args.shard_index)
    run_fit_stage2(load_config(args.config), split_idx, draw_idx)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pool every arm's validation-selected recovery and write the analysis."""
    run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage's SLURM jobs.

    Every ``fit``-stage array shares the job name ``"fit"`` (see ``_fit_job``),
    so this logs each array's command ahead of submission -- otherwise the
    grid combo behind a given array/job id would be unrecoverable from the log.
    """
    config = load_config(args.config)
    jobs = _stage_jobs(config, args.stage)
    for job in jobs:
        logger.info("submitting %s array: %s", job.name, job.command)
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI dispatch table."""
    return {
        "fit": cmd_fit,
        "stage2": cmd_stage2,
        "analyze": cmd_analyze,
        "submit": cmd_submit,
    }


def main() -> None:
    """Parse command-line arguments and dispatch subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
