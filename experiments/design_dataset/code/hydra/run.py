import _bootstrap  # noqa: F401 - side-effect: sets sys.path for shared/code roots
import argparse
import logging

from jobs import COMMAND_HANDLERS, execute, load_config
from progan_cache_jobs import execute_progan


def _add_full_scale_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sample-full-scale")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--class-order-name", default="native_prevalence")
    parser.add_argument("--class-order-file", default=None)


def _add_max_pool_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("max-feasible-pool-size")
    parser.add_argument("--parameter", type=float, default=1.3)
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
    parser.set_defaults(constructed=True)
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


def _add_patch_cache_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("patch-cache")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-order-name", default="native_prevalence")


def _add_progan_cache_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("patch-cache-progan")
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


def _add_bracs_args(subparsers: argparse._SubParsersAction) -> None:
    for command in (
        "bracs-stage",
        "bracs-prepare",
        "bracs-features",
        "bracs-patch-cache",
        "bracs-wsi-cache",
        "bracs-progan-cache",
        "bracs-progan-power-law",
        "bracs-power-law",
        "bracs-tune",
        "bracs-tune-wsi",
        "bracs-tune-power-law",
        "bracs-tune-wsi-power-law",
        "bracs-tune-aggregate",
        "bracs-tune-aggregate-power-law",
        "bracs-report",
        "bracs-report-power-law",
        "camelyon16-prepare",
        "camelyon16-features",
        "camelyon16-patch-cache",
        "camelyon16-progan-cache",
        "camelyon16-tune",
        "camelyon16-tune-wsi",
        "camelyon16-tune-aggregate",
        "camelyon16-report",
    ):
        subparsers.add_parser(command)


def create_parser() -> argparse.ArgumentParser:
    """Create the Hydra job runner parser."""
    parser = argparse.ArgumentParser(description="Run TCGA-UT jobs.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--no-container", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_max_pool_args(subparsers)
    _add_full_scale_args(subparsers)
    _add_train_args(subparsers)
    _add_train_wsi_args(subparsers)
    _add_wsi_cache_args(subparsers)
    _add_patch_cache_args(subparsers)
    _add_progan_cache_args(subparsers)
    _add_tune_args(subparsers)
    _add_tune_aggregate_args(subparsers)
    _add_visualize_args(subparsers)
    _add_report_args(subparsers)
    _add_verify_args(subparsers)
    _add_bracs_args(subparsers)
    return parser


def main() -> None:
    """Run the selected job command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = create_parser().parse_args()
    config = load_config(args.config)
    if args.command == "patch-cache-progan" and not args.local:
        execute_progan(args, config, args.local, args.dry_run)
        return
    for job in COMMAND_HANDLERS[args.command](args, config):
        execute(job, config, args.local, args.dry_run)


if __name__ == "__main__":
    main()
