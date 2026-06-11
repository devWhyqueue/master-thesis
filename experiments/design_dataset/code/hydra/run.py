import argparse
import logging
import sys
from pathlib import Path

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[3]
_CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (_EXPERIMENTS_ROOT / "shared", _CODE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jobs import COMMAND_HANDLERS, execute, load_config


def _add_balanced_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sample-balanced")
    parser.add_argument("--n-slides-per-class", type=int, default=100)
    parser.add_argument("--n-patches-per-slide", type=int, default=30)


def _add_imbalanced_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sample-imbalanced")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)


def _add_full_scale_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sample-full-scale")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-order-name", default="native_prevalence")
    parser.add_argument("--class-order-file", default=None)


def _add_train_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("model", choices=["mlp", "knn", "ncc"])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--method", default="patch_feature_ce")
    parser.add_argument("--constructed", action="store_true")
    parser.add_argument("--class-order-name", default="native_prevalence")


def _add_train_wsi_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train-wsi")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-bags-per-class", type=int, default=0)
    parser.add_argument(
        "--method",
        default="all",
        choices=[
            "all",
            "mil_ce",
            "mil_weighted_ce",
            "mil_balanced_sampler_ce",
            "mil_focal",
            "rankmix_mil",
            "sc_mil",
            "mde_mil",
        ],
    )
    parser.add_argument("--class-order-name", default="native_prevalence")


def _add_visualize_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("visualize")
    parser.add_argument("type", choices=["standard", "point-plot"])
    parser.add_argument("--class-order-name", default="native_prevalence")


def _add_wsi_cache_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("wsi-cache")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-order-name", default="native_prevalence")


def _add_tune_args(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("tune")
    subparsers.add_parser("tune-wsi")


def _add_tune_aggregate_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("tune-aggregate")
    parser.add_argument("--allow-incomplete", action="store_true")


def _add_report_args(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("report")


def _add_verify_args(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("verify-features")


def create_parser() -> argparse.ArgumentParser:
    """Create the Hydra job runner parser."""
    parser = argparse.ArgumentParser(description="Run TCGA-UT jobs.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--no-container", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_balanced_args(subparsers)
    _add_imbalanced_args(subparsers)
    _add_full_scale_args(subparsers)
    _add_train_args(subparsers)
    _add_train_wsi_args(subparsers)
    _add_wsi_cache_args(subparsers)
    _add_tune_args(subparsers)
    _add_tune_aggregate_args(subparsers)
    _add_visualize_args(subparsers)
    _add_report_args(subparsers)
    _add_verify_args(subparsers)
    return parser


def main() -> None:
    """Run the selected job command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = create_parser().parse_args()
    config = load_config(args.config)
    for job in COMMAND_HANDLERS[args.command](args, config):
        execute(job, config, args.local, args.dry_run)


if __name__ == "__main__":
    main()
