"""CLI entry point for full-scale constructed sampling."""

import argparse
import json
import logging
from typing import cast

import pandas as pd

from data.feature_store import (
    DEFAULT_FEATURE_DIR,
    verify_feature_store,
)
from data.full_scale.sampling import (
    attach_splits,
    class_order,
    constructed_payload,
    load_manifest,
    max_feasible_total,
    output_dir_for_args,
    split_frames,
    write_constructed_outputs,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Create full-scale constructed TCGA-UT manifests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if args.verify_features:
        report = verify_feature_store(args.feature_dir or DEFAULT_FEATURE_DIR)
        logger.info(json.dumps(report, indent=2))
        if not report["dim_matches"]:
            raise SystemExit("Feature dimension mismatch.")
        return
    if args.max_feasible_pool_size:
        max_total = _max_feasible_pool_size(args)
        logger.info(
            json.dumps(
                {
                    "class_order_name": args.class_order_name,
                    "lambda": args.parameter,
                    "max_feasible_pool_size": max_total,
                },
                indent=2,
            )
        )
        return
    output_dir = _run_sampling(args)
    logger.info("Stored constructed manifests in %s.", output_dir)


def _run_sampling(args: argparse.Namespace) -> str:
    manifest = attach_splits(
        load_manifest(args.slide_manifest_path),
        args.split_assignment_path,
        args.split_column,
    )
    ordered_classes = class_order(manifest, args.class_order_file)
    splits = split_frames(args, manifest)
    frames, targets = constructed_payload(args, splits, ordered_classes)
    output_dir = output_dir_for_args(args)
    write_constructed_outputs(
        frames,
        targets,
        ordered_classes,
        output_dir,
        vars(args),
        feature_dir=args.feature_dir,
    )
    return output_dir


def _max_feasible_pool_size(args: argparse.Namespace) -> int:
    manifest = attach_splits(
        load_manifest(args.slide_manifest_path),
        args.split_assignment_path,
        args.split_column,
    )
    ordered_classes = class_order(manifest, args.class_order_file)
    splits = split_frames(args, manifest)
    available = cast(
        pd.Series, splits[args.train_name].groupby("cancer_type")["slide_id"].nunique()
    )
    return max_feasible_total(available, ordered_classes, args.parameter)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-manifest-path")
    parser.add_argument("--split-assignment-path", default=None)
    parser.add_argument("--file-save-path")
    parser.add_argument("--parameter", type=float)
    parser.add_argument("--class-order-file", default=None)
    parser.add_argument("--class-order-name", default="native_prevalence")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--validation-name", default="validation")
    parser.add_argument("--test-name", default="test")
    parser.add_argument("--n-patches-per-slide", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--verify-features", action="store_true")
    parser.add_argument("--max-feasible-pool-size", action="store_true")
    args = parser.parse_args()
    if args.verify_features or args.max_feasible_pool_size:
        return args
    missing = [
        name
        for name, value in (
            ("slide-manifest-path", args.slide_manifest_path),
            ("file-save-path", args.file_save_path),
            ("parameter", args.parameter),
        )
        if value is None
    ]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    return args


if __name__ == "__main__":
    main()
