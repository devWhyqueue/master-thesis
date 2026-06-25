"""Submit BRACS native-or-power-law benchmark workflow on Hydra."""

from __future__ import annotations

import argparse
import logging
import dataclasses
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
    """Submit BRACS common stages and both mode-gated downstream branches."""
    args = parse_args()
    config = load_config(args.config)
    runner_args = SimpleNamespace(config=args.config, local=False, no_container=False)
    submitted = submit_workflow(runner_args, config, bool(args.dry_run))
    for phase, ids in submitted.items():
        logger.info(f"{phase}: {','.join(ids) if ids else 'no job ids'}")


def submit_workflow(
    runner_args: SimpleNamespace, config: dict[str, str], dry_run: bool
) -> dict[str, list[str]]:
    """Submit the full BRACS workflow and return job IDs by phase."""
    submitted: dict[str, list[str]] = {}

    stage = submit_phase("bracs-stage", runner_args, config, [], dry_run)
    submitted["bracs-stage"] = stage
    prepare = submit_phase("bracs-prepare", runner_args, config, stage, dry_run)
    submitted["bracs-prepare"] = prepare
    features = submit_phase("bracs-features", runner_args, config, prepare, dry_run)
    submitted["bracs-features"] = features
    submit_native_branch(submitted, runner_args, config, features, dry_run)
    submit_power_law_branch(submitted, runner_args, config, features, dry_run)
    return submitted


def submit_native_branch(
    submitted: dict[str, list[str]],
    runner_args: SimpleNamespace,
    config: dict[str, str],
    features: list[str],
    dry_run: bool,
) -> None:
    """Submit the mode-gated native BRACS branch."""
    native_patch_cache = submit_phase(
        "bracs-patch-cache", runner_args, config, features, dry_run
    )
    native_wsi_cache = submit_phase(
        "bracs-wsi-cache", runner_args, config, features, dry_run
    )
    native_progan = submit_phase(
        "bracs-progan-cache", runner_args, config, features, dry_run
    )
    submitted["bracs-patch-cache"] = native_patch_cache
    submitted["bracs-wsi-cache"] = native_wsi_cache
    submitted["bracs-progan-cache"] = native_progan

    native_patch_tune = submit_phase(
        "bracs-tune",
        runner_args,
        config,
        [*native_patch_cache, *native_progan],
        dry_run,
    )
    native_wsi_tune = submit_phase(
        "bracs-tune-wsi", runner_args, config, native_wsi_cache, dry_run
    )
    submitted["bracs-tune"] = native_patch_tune
    submitted["bracs-tune-wsi"] = native_wsi_tune
    native_agg = submit_phase(
        "bracs-tune-aggregate",
        runner_args,
        config,
        [*native_patch_tune, *native_wsi_tune],
        dry_run,
    )
    submitted["bracs-tune-aggregate"] = native_agg
    submitted["bracs-report"] = submit_phase(
        "bracs-report", runner_args, config, native_agg, dry_run
    )


def submit_power_law_branch(
    submitted: dict[str, list[str]],
    runner_args: SimpleNamespace,
    config: dict[str, str],
    features: list[str],
    dry_run: bool,
) -> None:
    """Submit the mode-gated BRACS power-law fallback branch."""
    power_law = submit_phase("bracs-power-law", runner_args, config, features, dry_run)
    power_law_progan = submit_phase(
        "bracs-progan-power-law", runner_args, config, power_law, dry_run
    )
    submitted["bracs-power-law"] = power_law
    submitted["bracs-progan-power-law"] = power_law_progan
    pw_patch_tune = submit_phase(
        "bracs-tune-power-law",
        runner_args,
        config,
        [*power_law, *power_law_progan],
        dry_run,
    )
    pw_wsi_tune = submit_phase(
        "bracs-tune-wsi-power-law", runner_args, config, power_law, dry_run
    )
    submitted["bracs-tune-power-law"] = pw_patch_tune
    submitted["bracs-tune-wsi-power-law"] = pw_wsi_tune
    pw_agg = submit_phase(
        "bracs-tune-aggregate-power-law",
        runner_args,
        config,
        [*pw_patch_tune, *pw_wsi_tune],
        dry_run,
    )
    submitted["bracs-tune-aggregate-power-law"] = pw_agg
    submitted["bracs-report-power-law"] = submit_phase(
        "bracs-report-power-law", runner_args, config, pw_agg, dry_run
    )


def submit_phase(
    phase: str,
    runner_args: SimpleNamespace,
    config: dict[str, str],
    dependencies: list[str],
    dry_run: bool,
) -> list[str]:
    """Submit all jobs in one phase with optional afterok dependencies."""
    jobs = COMMAND_HANDLERS[phase](runner_args, config)
    if dependencies:
        dep = "afterok:" + ":".join(dependencies)
        jobs = [dataclasses.replace(job, dependency=dep) for job in jobs]
    return [
        job_id
        for job in jobs
        if (job_id := execute(job, config, local=False, dry_run=dry_run))
    ]


if __name__ == "__main__":
    main()
