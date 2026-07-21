from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import subprocess
from typing import Any, Callable

from imbalance_benchmark.common import ensure_dirs, load_config, split_paths
from imbalance_benchmark.hydra.rendering import SlurmJob, render_sbatch
from imbalance_benchmark.modeling.context import CONDITIONS
from imbalance_benchmark.modeling.context import roster_for_regime
from imbalance_benchmark.manifest.freeze import verify_manifest_freeze
from imbalance_benchmark.modeling.workflows.tuning.tuning_schedule import (
    DEPENDENT_METHODS,
    candidate_array_size,
)

logger = logging.getLogger(__name__)


def _resources(
    config: dict[str, Any], stage: str, gpu: bool, fallback: str | None = None
) -> dict[str, Any]:
    sl = config.get("slurm", {})
    resources = sl.get("resources", {})
    sr = resources.get(stage, resources.get(fallback, {}))
    part = sr.get("partition", sl.get("partition", "gpu-2h" if gpu else "cpu-2h"))
    time = sr.get("time")
    return {
        "partition": part,
        "gpus": int(sr.get("gpus", 1 if gpu else 0)),
        "cpus": int(sr.get("cpus", 4)),
        # Unset unless a stage needs less than its partition's own time limit;
        # SLURM already applies that limit when --time is omitted.
        "time_limit": str(time) if time else None,
    }


def _job(
    config: dict[str, Any],
    stage: str,
    cmd: str,
    gpu: bool,
    dependencies: tuple[str, ...] = (),
    resource: str | None = None,
    fallback: str | None = None,
) -> SlurmJob:
    return SlurmJob(
        stage,
        cmd,
        dependencies=dependencies,
        **_resources(config, resource or stage, gpu, fallback),
    )


def build_workflow(
    config: dict[str, Any], smoke: bool = False, resume_tuning: bool = False
) -> list[SlurmJob]:
    """Build the benchmark DAG, or its test-partition synthetic smoke variant."""
    if smoke:
        res = _resources(config, "smoke", True)
        res["partition"] = config.get("slurm", {}).get("test_partition", "gpu-test")
        return [SlurmJob("smoke", "smoke", **res)]
    arr = (0, 1, 2)
    setup = _setup_jobs(config, arr) if not resume_tuning else []
    freeze_dependency = ("freeze",) if setup else ()
    tuning = _tuning_jobs(config, freeze_dependency)
    co = replace(
        _job(config, "confirm", "confirm", True, ("tune-final-reduce",)),
        array_splits=arr,
        array_conditions=CONDITIONS,
    )
    an = _job(config, "analyze", "analyze", False, (co.name,))
    return [*setup, *tuning, co, an]


def _setup_jobs(config: dict[str, Any], splits: tuple[int, ...]) -> list[SlurmJob]:
    prepare = _job(config, "prepare", "prepare", True)
    pilot = replace(
        _job(config, "pilot", "pilot", False, (prepare.name,)),
        array_splits=splits,
    )
    freeze = replace(
        _job(config, "freeze", "freeze", False, (pilot.name,)),
        array_splits=splits,
    )
    return [prepare, pilot, freeze]


def _tuning_jobs(
    config: dict[str, Any], freeze_dependency: tuple[str, ...]
) -> list[SlurmJob]:
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    roster = roster_for_regime(is_mil)
    base_methods = tuple(method for method in roster if method not in DEPENDENT_METHODS)
    dependent_methods = tuple(
        method for method in roster if method in DEPENDENT_METHODS
    )
    base_natural = replace(
        _job(
            config,
            "tune-base-natural",
            "tune-shard --phase base --group natural",
            True,
            freeze_dependency,
            "tune_natural",
            "tune",
        ),
        array_size=candidate_array_size(base_methods),
    )
    base_controlled = replace(
        _job(
            config,
            "tune-base-controlled",
            "tune-shard --phase base --group controlled",
            True,
            freeze_dependency,
            "tune_controlled",
            "tune",
        ),
        array_size=3 * candidate_array_size(base_methods),
    )
    base_reduce = _job(
        config,
        "tune-base-reduce",
        "tune-reduce --phase base",
        False,
        (base_natural.name, base_controlled.name),
        "tune_reduce",
    )
    dependent_natural = replace(
        _job(
            config,
            "tune-dependent-natural",
            "tune-shard --phase dependent --group natural",
            True,
            (base_reduce.name,),
            "tune_natural",
            "tune",
        ),
        array_size=candidate_array_size(dependent_methods),
    )
    dependent_controlled = replace(
        _job(
            config,
            "tune-dependent-controlled",
            "tune-shard --phase dependent --group controlled",
            True,
            (base_reduce.name,),
            "tune_controlled",
            "tune",
        ),
        array_size=3 * candidate_array_size(dependent_methods),
    )
    final_reduce = _job(
        config,
        "tune-final-reduce",
        "tune-reduce --phase final",
        False,
        (dependent_natural.name, dependent_controlled.name),
        "tune_reduce",
    )
    return [
        base_natural,
        base_controlled,
        base_reduce,
        dependent_natural,
        dependent_controlled,
        final_reduce,
    ]


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
    resume_tuning: bool = False,
    submit: Callable[[str, bool], str] = _submit_script,
) -> dict[str, str]:
    """Render and submit the workflow in topological order, returning job IDs by stage."""
    if resume_tuning:
        _verify_resume_freezes(config)
    submitted: dict[str, str] = {}
    for job in build_workflow(config, smoke, resume_tuning):
        dependencies = tuple(submitted[name] for name in job.dependencies)
        scheduled = replace(job, dependencies=dependencies)
        script = render_sbatch(scheduled, config, config_path)
        jid = f"dry-run-{job.name}" if dry_run else submit(script, False)
        submitted[job.name] = jid
        logger.info("%s: %s", job.name, jid)
        if dry_run:
            logger.info("%s", script)
    return submitted


def _verify_resume_freezes(config: dict[str, Any]) -> None:
    base = ensure_dirs(config)
    for index in range(3):
        path = split_paths(base, index)["data"] / "manifest_freeze.json"
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume tuning without {path}")
        verify_manifest_freeze(json.loads(path.read_text()))


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit the Hydra workflow."""
    config = load_config(args.config)
    submit_workflow(
        config,
        args.config,
        args.dry_run,
        getattr(args, "smoke", False),
        getattr(args, "resume_tuning", False),
    )
