"""Command-line entry point for the diversity-at-fixed-support experiment (exp-3).

Mirrors experiments/2_benchmark_patch/code/__main__.py's dispatch style.
Importing ``_bootstrap`` first prepends exp-2's code directory to
``sys.path`` so ``imbalance_benchmark`` (imported as a library, never
edited) and the top-level ``derive_deficit_thresholds`` script are
importable -- the one path hack the plan calls for, sufficient on the
cluster once ``render_sbatch`` exports ``APPTAINERENV_PYTHONPATH=<this code
dir>`` (this directory itself needs no extra prepend: Python already puts a
script's own directory on ``sys.path`` when it's run as ``python __main__.py``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401  (side effect: prepends exp-2's code dir to sys.path)
from imbalance_benchmark.common import load_config

from diversity import analyze as analyze_stage
from diversity import check as check_stage
from diversity import fit as fit_stage
from diversity import manifests
from diversity import slurm as slurm_stage

FIT_GROUPS = ("standard", "semantic_scale")


def _parser() -> argparse.ArgumentParser:
    """Create exp-3's command-line parser."""
    parser = argparse.ArgumentParser(
        description="Diversity-at-Fixed-Support Experiment CLI"
    )
    # No config default: exp-2's own default.yaml is the wrong fallback for
    # exp-3 (different dataset, no slurm.exp2_outputs), so a config is
    # required rather than silently loading exp-2's.
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-index", type=int, choices=range(3), default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "check", "import-anchor", "fit", "analyze"):
        sub.add_parser(command)
    fit_shard = sub.add_parser("fit-shard")
    fit_shard.add_argument("--group", choices=FIT_GROUPS, required=True)
    fit_shard.add_argument("--shard-index", type=int, required=True)
    fit_shard.add_argument("--shards-per-task", type=int, default=1)
    submit = sub.add_parser("submit")
    submit.add_argument("--stage", choices=("build", "fit"), required=True)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def cmd_build(args: argparse.Namespace) -> None:
    """Build the six (allocation x level) manifests and derived freeze for each split."""
    config = load_config(args.config)
    for split_index in range(3) if args.split_index is None else (args.split_index,):
        manifests.build_split(config, split_index)


def cmd_check(args: argparse.Namespace) -> None:
    """Run Gate 0 (headroom, semantic volume, S_nom/S_ind audit, gate) for this dataset."""
    check_stage.run_check(load_config(args.config))


def cmd_import_anchor(args: argparse.Namespace) -> None:
    """Import exp-2's confirmed 'random'-anchor runs, gated on manifest/tuning equality."""
    fit_stage.import_anchor(load_config(args.config))


def cmd_fit(args: argparse.Namespace) -> None:
    """Fit every narrow/wide work item locally (no sharding); for manual/local runs."""
    config = load_config(args.config)
    for group in FIT_GROUPS:
        units = fit_stage.fit_units(group)
        fit_stage.run_fit_shard(config, group, 0, len(units))


def cmd_fit_shard(args: argparse.Namespace) -> None:
    """Fit one SLURM array task's bundle of work items."""
    config = load_config(args.config)
    fit_stage.run_fit_shard(config, args.group, args.shard_index, args.shards_per_task)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run the damage / interaction / recovery analysis and write its Holm-adjusted table."""
    analyze_stage.run_analyze(load_config(args.config))


def cmd_submit(args: argparse.Namespace) -> None:
    """Render and submit one phase of exp-3's SLURM DAG ('build' Gate 0, or 'fit')."""
    config = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    slurm_stage.submit_workflow(config, args.stage, config_path, args.dry_run)


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI command handlers keyed by their subcommand names."""
    return {
        "build": cmd_build,
        "check": cmd_check,
        "import-anchor": cmd_import_anchor,
        "fit": cmd_fit,
        "fit-shard": cmd_fit_shard,
        "analyze": cmd_analyze,
        "submit": cmd_submit,
    }


def main() -> None:
    """Parse exp-3's command-line arguments and dispatch the selected command."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
