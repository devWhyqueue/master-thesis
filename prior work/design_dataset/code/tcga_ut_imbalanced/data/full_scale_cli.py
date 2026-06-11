import argparse
import logging

from tcga_ut_imbalanced.data.full_scale_sampling import (
    attach_splits,
    class_order,
    constructed_payload,
    load_manifest,
    output_dir_for_args,
    split_frames,
    write_constructed_outputs,
)

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse full-scale constructed-split arguments."""
    parser = argparse.ArgumentParser()
    _add_input_args(parser)
    _add_construction_args(parser)
    return parser.parse_args()


def main() -> None:
    """Create full-scale constructed TCGA-UT manifests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    manifest = attach_splits(
        load_manifest(args.slide_manifest_path),
        args.split_assignment_path,
        args.split_column,
    )
    ordered_classes = class_order(manifest, args.class_order_file)
    splits = split_frames(args, manifest)
    frames, targets = constructed_payload(args, splits, ordered_classes)
    output_dir = output_dir_for_args(args)
    write_constructed_outputs(frames, targets, ordered_classes, output_dir, vars(args))
    logger.info("Stored constructed manifests in %s.", output_dir)


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slide-manifest-path", required=True)
    parser.add_argument("--split-assignment-path", default=None)
    parser.add_argument("--file-save-path", required=True)


def _add_construction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parameter", type=float, required=True)
    parser.add_argument("--class-order-file", default=None)
    parser.add_argument("--class-order-name", default="native_prevalence")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--validation-name", default="validation")
    parser.add_argument("--test-name", default="test")
    parser.add_argument("--n-patches-per-slide", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overflow-strategy",
        default="redistribute",
        choices=["redistribute", "replacement"],
    )


if __name__ == "__main__":
    main()
