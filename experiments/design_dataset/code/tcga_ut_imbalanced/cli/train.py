import argparse
import logging

import torch

from tcga_ut_imbalanced.cli.train_core import run_training


def positive_int(value: str) -> int:
    """Parse a non-negative integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"Number needs to be positive, but {parsed} is not."
        )
    return parsed


def positive_float(value: str) -> float:
    """Parse a non-negative float."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"Number needs to be positive, but {parsed} is not."
        )
    return parsed


def get_args() -> argparse.Namespace:
    """Parse training command arguments."""
    parser = argparse.ArgumentParser()
    _add_dataset_args(parser)
    _add_model_args(parser)
    _add_training_args(parser)
    _add_visualization_args(parser)
    return parser.parse_args()


def main() -> None:
    """Run the training command."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    torch.manual_seed(args.seed)
    run_training(args)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-structure-path", type=str, required=True)
    parser.add_argument("--dataset-split", type=str, default=None)
    parser.add_argument("--validation-dataset-structure-path", type=str, default=None)
    parser.add_argument("--validation-dataset-split", type=str, default=None)
    parser.add_argument("--test-dataset-structure-path", type=str, default=None)
    parser.add_argument("--test-dataset-split", type=str, default=None)
    parser.add_argument("--feature-path", type=str, required=True)
    parser.add_argument("--feature-cache-path", type=str, default=None)
    parser.add_argument("--args-path", type=str, default=None)
    parser.add_argument("--preload-features", action="store_true")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="mlp", choices=["mlp", "knn", "ncc"])
    parser.add_argument("--k", type=positive_int)
    parser.add_argument("--n-nodes-per-layer", default=[512], nargs="+", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--results-save-path", type=str, required=True)
    parser.add_argument("--store-timestamp", action="store_true")


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loss",
        default="cross_entropy",
        choices=["cross_entropy", "focal_loss", "ce_soft_f1", "ce_soft_mcc"],
    )
    parser.add_argument("--gamma", default=2.0, type=positive_float)
    parser.add_argument("--metric-loss-weight", default=1.0, type=positive_float)
    parser.add_argument(
        "--alpha", default="uniform", choices=["uniform", "inverse_class_frequency"]
    )
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--training-method", default="patch_feature_ce")
    parser.add_argument("--tuning-id", default=None)
    parser.add_argument("--tuning-params", default=None)
    parser.add_argument("--cfal-lambda", type=float, default=0.1)
    parser.add_argument("--cfal-sigma", type=float, default=1.0)
    parser.add_argument("--cfal-gamma", type=float, default=2.0)
    parser.add_argument("--cfal-beta", type=float, default=0.999)
    parser.add_argument("--dnc-k-clusters", type=int, default=10)
    parser.add_argument("--dnc-zscore-bins", type=int, default=5)
    parser.add_argument("--dnc-expert-epochs", type=int, default=20)
    parser.add_argument("--oko-k", type=int, default=1)
    parser.add_argument("--batch-balancing", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)


def _add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--class-names-path", type=str, default=None)


if __name__ == "__main__":
    main()
