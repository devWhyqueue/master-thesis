import argparse
import logging

from tcga_ut_imbalanced.cli.sample_imbalanced_core import sample_imbalanced


def region_int(value: str) -> int:
    """Parse a region patch count from 1 to 10."""
    parsed = int(value)
    if parsed <= 0 or parsed > 10:
        raise argparse.ArgumentTypeError(
            f"Number needs to be at least 1 and at most 10, but {parsed} is not.",
        )
    return parsed


def get_args() -> argparse.Namespace:
    """Parse imbalanced-sampling arguments."""
    parser = argparse.ArgumentParser()
    _add_required_args(parser)
    _add_sampling_args(parser)
    _add_validation_args(parser)
    args = parser.parse_args()
    if args.sample_balanced_validation and args.n_slides_per_class is None:
        raise argparse.ArgumentTypeError(
            "--sample-balanced-validation requires --n-slides-per-class."
        )
    return args


def main() -> None:
    """Run imbalanced dataset sampling."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sample_imbalanced(get_args())


def _add_required_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--balanced-dataset-path", type=str, required=True)
    parser.add_argument("--file-save-path", type=str, required=True)
    parser.add_argument("--parameter", type=float, required=True)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--n-regions-per-slide", type=int, default=None, required=True)
    parser.add_argument(
        "--n-patches-per-region", type=region_int, default=10, required=True
    )


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overflow-strategy", choices=["none", "redistribute"], default="none"
    )
    parser.add_argument("--class-order-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--store-class-names", action="store_true")


def _add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-balanced-validation", action="store_true")
    parser.add_argument("--n-slides-per-class", type=int, default=None)


if __name__ == "__main__":
    main()
