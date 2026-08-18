"""Command-line entry point for the unified imbalance benchmark."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from PIL import PngImagePlugin

from imbalance_benchmark.commands import (
    cmd_amend_grids,
    cmd_analyze,
    cmd_analyze_combine,
    cmd_combine_rq3,
    cmd_confirm,
    cmd_confirm_shard,
    cmd_freeze,
    cmd_match,
    cmd_materialize_panda,
    cmd_materialize_panda_audit,
    cmd_materialize_panda_combine,
    cmd_materialize_panda_pack,
    cmd_materialize_panda_publish,
    cmd_materialize_tcga_ut,
    cmd_pilot,
    cmd_prepare,
    cmd_prepare_extract_reduce,
    cmd_prepare_extract_shard,
    cmd_refreeze_preflight,
    cmd_signals,
    cmd_smoke,
    cmd_submit,
    cmd_tile_wsi,
    cmd_tile_wsi_reduce,
    cmd_tune,
    cmd_tune_decide,
    cmd_tune_reduce,
    cmd_tune_shard,
    cmd_tune_wave,
)


# BRACS WSI tiles carry a large embedded scanner ICC profile (unused by this
# pipeline) that overruns Pillow's default 1 MiB decompression-bomb guard.
PngImagePlugin.MAX_TEXT_CHUNK = 50 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    """Create the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description="Class-Imbalance Benchmark CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-index", type=int, choices=range(3), default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "pilot",
        "freeze",
        "amend-grids",
        "refreeze-preflight",
        "signals",
        "match",
        "tune",
        "tune-shard",
        "tune-reduce",
        "tune-wave",
        "confirm",
        "confirm-shard",
        "analyze",
        "analyze-combine",
        "combine-rq3",
        "smoke",
        "tile-wsi-reduce",
        "materialize-tcga-ut",
        "materialize-panda",
        "materialize-panda-combine",
        "materialize-panda-publish",
        "prepare-extract-reduce",
    ):
        sub.add_parser(command)
    submit = sub.add_parser("submit")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--smoke", action="store_true")
    submit.add_argument("--resume-tuning", action="store_true")
    submit.add_argument("--confirm-only", action="store_true")
    submit.add_argument(
        "--stage", choices=("materialize", "extract", "prepare", "pilot", "freeze")
    )
    tile_wsi = sub.add_parser("tile-wsi")
    tile_wsi.add_argument("--slide-index", type=int, required=True)
    tile_wsi.add_argument("--shard-size", type=int, default=1)
    extract_shard = sub.add_parser("prepare-extract-shard")
    extract_shard.add_argument("--shard-index", type=int, required=True)
    extract_shard.add_argument("--shard-count", type=int, required=True)
    sub.choices["materialize-panda"].add_argument("--canary", action="store_true")
    materialize_audit = sub.add_parser("materialize-panda-audit")
    materialize_audit.add_argument("--shard-index", type=int, required=True)
    materialize_pack = sub.add_parser("materialize-panda-pack")
    materialize_pack.add_argument("--shard-index", type=int, required=True)
    for command in ("tune", "confirm"):
        sub.choices[command].add_argument(
            "--condition", choices=("natural", "balanced", "moderate", "severe")
        )
    shard = sub.choices["tune-shard"]
    shard.add_argument("--phase", choices=("base", "dependent"), required=True)
    shard.add_argument("--group", choices=("natural", "controlled"))
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--observation-index", type=int)
    shard.add_argument("--observations-per-candidate", type=int, default=1)
    shard.add_argument("--shard-offset", type=int, default=0)
    shard.add_argument("--shards-per-task", type=int, default=1)
    shard.add_argument("--bundle-by-observation", action="store_true")
    shard.add_argument("--round", type=int, default=0)
    shard.add_argument(
        "--condition", choices=("natural", "balanced", "moderate", "severe")
    )
    wave = sub.choices["tune-wave"]
    wave.add_argument("--phase", choices=("base", "dependent"), default="base")
    wave.add_argument(
        "--condition", choices=("natural", "balanced", "moderate", "severe")
    )
    wave.add_argument("--group", choices=("controlled",))
    wave.add_argument("--round", type=int, default=0)
    wave.add_argument("--stalled-waves", type=int, default=0)
    reduce = sub.choices["tune-reduce"]
    reduce.add_argument("--phase", choices=("base", "final"), required=True)
    reduce.add_argument(
        "--condition", choices=("natural", "balanced", "moderate", "severe")
    )
    decide = sub.add_parser("tune-decide")
    decide.add_argument("--phase", choices=("base", "dependent"), required=True)
    decide.add_argument(
        "--condition",
        choices=("natural", "balanced", "moderate", "severe"),
        required=True,
    )
    decide.add_argument("--round", type=int, default=0)
    confirm_shard = sub.choices["confirm-shard"]
    confirm_shard.add_argument(
        "--group", choices=("natural", "controlled"), required=True
    )
    confirm_shard.add_argument("--shard-index", type=int, required=True)
    confirm_shard.add_argument("--shards-per-task", type=int, default=1)
    return parser


def _commands() -> dict[str, Callable[[argparse.Namespace], None]]:
    """Return CLI command handlers keyed by their subcommand names."""
    return {
        "prepare": cmd_prepare,
        "prepare-extract-shard": cmd_prepare_extract_shard,
        "prepare-extract-reduce": cmd_prepare_extract_reduce,
        "pilot": cmd_pilot,
        "freeze": cmd_freeze,
        "amend-grids": cmd_amend_grids,
        "refreeze-preflight": cmd_refreeze_preflight,
        "signals": cmd_signals,
        "match": cmd_match,
        "tune": cmd_tune,
        "tune-shard": cmd_tune_shard,
        "tune-wave": cmd_tune_wave,
        "tune-reduce": cmd_tune_reduce,
        "tune-decide": cmd_tune_decide,
        "confirm": cmd_confirm,
        "confirm-shard": cmd_confirm_shard,
        "analyze": cmd_analyze,
        "analyze-combine": cmd_analyze_combine,
        "combine-rq3": cmd_combine_rq3,
        "submit": cmd_submit,
        "smoke": cmd_smoke,
        "tile-wsi": cmd_tile_wsi,
        "tile-wsi-reduce": cmd_tile_wsi_reduce,
        "materialize-tcga-ut": cmd_materialize_tcga_ut,
        "materialize-panda": cmd_materialize_panda,
        "materialize-panda-audit": cmd_materialize_panda_audit,
        "materialize-panda-combine": cmd_materialize_panda_combine,
        "materialize-panda-pack": cmd_materialize_panda_pack,
        "materialize-panda-publish": cmd_materialize_panda_publish,
    }


def main() -> None:
    """Parse benchmark command-line arguments and dispatch the selected command."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = _parser().parse_args()
    _commands()[args.command](args)


if __name__ == "__main__":
    main()
