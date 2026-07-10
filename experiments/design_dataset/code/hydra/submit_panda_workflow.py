"""Submit the native PANDA benchmark workflow on Hydra.

PANDA ships raw slides only, so the workflow tiles from scratch before the usual
native pipeline. Phases are chained by SLURM afterok dependencies:
select -> tile -> prepare -> features -> {patch-cache -> tune, tune-wsi}
-> tune-aggregate -> report. ProGAN augmentation is intentionally omitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from types import SimpleNamespace

import _bootstrap  # noqa: F401
from jobs import COMMAND_HANDLERS, execute, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse workflow arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Submit the PANDA native workflow and log job IDs by phase."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_config(args.config)
    runner_args = SimpleNamespace(config=args.config, local=False, no_container=False)
    submitted = submit_workflow(runner_args, config, bool(args.dry_run))
    for phase, ids in submitted.items():
        logger.info("%s: %s", phase, ",".join(ids) if ids else "no job ids")


def submit_workflow(
    runner_args: SimpleNamespace, config: dict[str, str], dry_run: bool
) -> dict[str, list[str]]:
    """Submit the full PANDA native workflow and return job IDs by phase."""
    submitted: dict[str, list[str]] = {}

    def run(phase: str, dependencies: list[str], kind: str = "afterok") -> list[str]:
        """Submit one phase, record its ids, and return them for downstream deps."""
        submitted[phase] = submit_phase(
            phase, runner_args, config, dependencies, dry_run, kind
        )
        return submitted[phase]

    prepare = run("panda-prepare", run("panda-tile", run("panda-select", [])))
    features = run("panda-features", prepare)
    patch_cache = run("panda-patch-cache", features)
    progan = run("panda-progan-cache", features)
    patch_tune = run("panda-tune", [*patch_cache, *progan])
    wsi_tune = run("panda-tune-wsi", features)
    # afterany, not afterok: a single failed tuning task should not block the report.
    aggregate = run("panda-tune-aggregate", [*patch_tune, *wsi_tune], "afterany")
    run("panda-report", aggregate)
    return submitted


def submit_phase(
    phase: str,
    runner_args: SimpleNamespace,
    config: dict[str, str],
    dependencies: list[str],
    dry_run: bool,
    kind: str = "afterok",
) -> list[str]:
    """Submit all jobs in one phase with optional afterok/afterany dependencies."""
    jobs = COMMAND_HANDLERS[phase](runner_args, config)
    if dependencies:
        dep = f"{kind}:" + ":".join(dependencies)
        jobs = [dataclasses.replace(job, dependency=dep) for job in jobs]
    return [
        job_id
        for job in jobs
        if (job_id := execute(job, config, local=False, dry_run=dry_run))
    ]


if __name__ == "__main__":
    main()
