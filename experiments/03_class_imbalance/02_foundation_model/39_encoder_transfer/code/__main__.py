"""Command-line entry point for exp-39's encoder-transfer schedule (phase 01) and
UNI2-h feature extraction/audit (phase 03)."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import (
    N_PATIENT_SPLITS,
    load_config,
    output_root,
    sign_file,
    write_json,
)
from imbalance_benchmark.hydra.job_resources import build_job
from imbalance_benchmark.hydra.rendering import SlurmJob

from breadth.slurm import submit_workflow

from prevalence import patients_per_class

from transfer import MAIN_DRAWS, extract, manifest
from transfer import analyze as analyze_stage
from transfer import fit as fit_stage
from transfer import preflight as preflight_stage
from transfer.fit import shard_count
from transfer.schedule import draw_schedule, load_train_identity

logger = logging.getLogger(__name__)

__all__ = ["main"]

_SUBMIT_STAGES = "extract audit-features preflight pilot fit analyze".split()


def _extraction_jobs(config: dict) -> list[SlurmJob]:
    """extract-features (GPU array) -> merge-features -> audit-features -> preflight."""
    n = int(config.get("slurm", {}).get("extract_shards", 1))
    extract_job = replace(
        build_job(
            config,
            "extract-features",
            f"extract-features --shards {n} --dtype float32",
            True,
        ),
        array_size=n,
    )
    merge_job = build_job(
        config, "merge-features", "merge-features", False, (extract_job.name,)
    )
    audit_job = build_job(
        config, "audit-features", "audit-features", False, (merge_job.name,)
    )
    preflight_job = build_job(
        config, "preflight", "preflight", False, (audit_job.name,)
    )
    return [extract_job, merge_job, audit_job, preflight_job]


def _fit_job(config: dict, dependencies: tuple[str, ...]) -> SlurmJob:
    return replace(
        build_job(config, "fit", "fit", False, dependencies), array_size=shard_count()
    )


def _stage_jobs(config: dict, stage: str) -> list[SlurmJob]:
    """Jobs for one submission stage: ``extract`` (through preflight), ``fit``, ``analyze``, or ``all``."""
    if stage == "extract":
        return _extraction_jobs(config)
    if stage == "audit-features":
        audit = build_job(config, "audit-features", "audit-features", False)
        return [
            audit,
            build_job(config, "preflight", "preflight", False, (audit.name,)),
        ]
    if stage == "preflight":
        return [build_job(config, "preflight", "preflight", False)]
    if stage == "pilot":
        extraction = build_job(
            config, "extract-pilot", "extract-pilot", True, resource="extract-features"
        )
        return [
            extraction,
            build_job(
                config,
                "pilot-virchow2",
                "pilot-fit --encoder virchow2",
                False,
                resource="pilot",
            ),
            build_job(
                config,
                "pilot-uni2h",
                "pilot-fit --encoder uni2h",
                False,
                (extraction.name,),
                resource="pilot",
            ),
        ]
    if stage == "fit":
        return [_fit_job(config, ())]
    if stage == "analyze":
        audit = build_job(config, "audit-fits", "audit-fits", False, ())
        return [audit, build_job(config, "analyze", "analyze", False, (audit.name,))]
    raise ValueError(f"Unknown submission stage: {stage}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exp-39 encoder-transfer schedule CLI")
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schedule")
    p_extract = sub.add_parser("extract-features")
    p_extract.add_argument("--shard-index", type=int, required=True)
    p_extract.add_argument("--shards", type=int, required=True)
    p_extract.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    sub.add_parser("merge-features")
    sub.add_parser("audit-features")
    sub.add_parser("extract-pilot")
    sub.add_parser("preflight")
    p_fit = sub.add_parser("fit")
    p_fit.add_argument(
        "--shard-index", type=int, choices=range(shard_count()), required=True
    )
    p_pilot = sub.add_parser("pilot-fit")
    p_pilot.add_argument("--encoder", choices=("virchow2", "uni2h"), required=True)
    sub.add_parser("audit-fits")
    sub.add_parser("analyze")
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=_SUBMIT_STAGES, required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_schedule(args: argparse.Namespace) -> None:
    """Build and hash this dataset's frozen draws-10-19 schedule."""
    config = load_config(args.config)
    g = patients_per_class(config)
    cells = []
    for split_idx in range(N_PATIENT_SPLITS):
        train_df, names = load_train_identity(config, split_idx)
        for draw_idx in MAIN_DRAWS:
            cells.append(draw_schedule(train_df, names, split_idx, draw_idx, g))
    out_path = output_root(config) / "schedule.json"
    write_json(out_path, {"dataset": config["dataset"]["name"], "cells": cells})
    sign_file(out_path)
    logger.info(f"wrote {out_path}")


def cmd_extract_features(args: argparse.Namespace) -> None:
    """Extract this shard's assigned, not-yet-cached UNI2-h slide tensors."""
    config = load_config(args.config)
    extract.extract_shard(config, args.shard_index, args.shards, dtype=args.dtype)
    logger.info(f"shard {args.shard_index}/{args.shards} done")


def cmd_merge_features(args: argparse.Namespace) -> None:
    """One-time merge of every completed shard's pending UNI2-h records."""
    extract.merge_features(load_config(args.config))
    logger.info("merge complete")


def cmd_audit_features(args: argparse.Namespace) -> None:
    """Verify both encoders' caches and publish feature_audit.json + manifests."""
    audit = manifest.run_audit(load_config(args.config))
    logger.info(
        f"uni2h missing={audit['uni2h']['unresolved_missing']} "
        f"corrupt={audit['uni2h']['unresolved_corrupt']}; "
        f"virchow2 missing={audit['virchow2']['unresolved_missing']} "
        f"corrupt={audit['virchow2']['unresolved_corrupt']}"
    )


def cmd_extract_pilot(args: argparse.Namespace) -> None:
    """Extract the reserved draw into a separate cache for the engineering pilot."""
    path = extract.extract_pilot(load_config(args.config))
    logger.info(f"wrote {path}")


def cmd_preflight(args: argparse.Namespace) -> None:
    """Verify frozen locks, the feature audit, and joined manifests; write preflight.json."""
    preflight_stage.run_preflight(load_config(args.config))
    logger.info("preflight passed")


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit every pending arm of one (encoder, split, draw) shard."""
    fit_stage.run_fit_shard(load_config(args.config), args.shard_index)


def cmd_pilot_fit(args: argparse.Namespace) -> None:
    """Fit reserved draw 10000 for one encoder on split 0."""
    fit_stage.run_pilot_fit(load_config(args.config), args.encoder)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pool encoder/arm accuracy and probability quality, decompose, and write analysis.json."""
    config = load_config(args.config)
    fit_stage.audit_fits(config)
    path = analyze_stage.run_analyze(config)
    logger.info(f"wrote {path}")


def cmd_audit_fits(args: argparse.Namespace) -> None:
    """Write the selected-arm completeness report and reject missing evidence."""
    path = fit_stage.audit_fits(load_config(args.config))
    logger.info(f"wrote {path}")


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit one stage of the extract -> fit -> analyze DAG, or the full chain."""
    config = load_config(args.config)
    jobs = _stage_jobs(config, args.stage)
    submit_workflow(config, str(Path(args.config).resolve()), args.dry_run, jobs=jobs)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    return {
        "schedule": cmd_schedule,
        "extract-features": cmd_extract_features,
        "merge-features": cmd_merge_features,
        "audit-features": cmd_audit_features,
        "extract-pilot": cmd_extract_pilot,
        "preflight": cmd_preflight,
        "fit": cmd_fit,
        "pilot-fit": cmd_pilot_fit,
        "audit-fits": cmd_audit_fits,
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
