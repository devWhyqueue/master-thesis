import argparse
import logging

from jobs import COMMAND_HANDLERS, execute, load_config


def _add_balanced_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sample-balanced")
    parser.add_argument("--n-slides-per-class", type=int, default=100)
    parser.add_argument("--n-patches-per-slide", type=int, default=30)


def _add_imbalanced_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("sample-imbalanced")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)


def _add_train_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("model", choices=["mlp", "knn", "ncc"])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--parameter", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=9)


def _add_visualize_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("visualize")
    parser.add_argument("type", choices=["standard", "point-plot"])


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
    _add_train_args(subparsers)
    _add_visualize_args(subparsers)
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
