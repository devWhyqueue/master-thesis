import argparse
import json
import logging
import os

from tcga_ut_imbalanced.data.sampling import (
    get_dataset_structure,
    sample_balanced_from_dataset_structure,
)
from tcga_ut_imbalanced.data.utils import convert_dataset_structure_to_dataframe

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse balanced-sampling arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--file-save-path", type=str, required=True)
    parser.add_argument("--n-slides-per-class", type=int, required=True)
    parser.add_argument("--n-patches-per-slide", type=int, required=True)
    parser.add_argument("--slide-id-exclusion-path", type=str, default=None)
    parser.add_argument("--store-slide-ids", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run balanced dataset sampling."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    df = convert_dataset_structure_to_dataframe(
        get_dataset_structure(args.dataset_path, _slides_to_exclude(args)),
    )
    sampled = sample_balanced_from_dataset_structure(
        df,
        args.n_slides_per_class,
        n_patches=args.n_patches_per_slide,
        seed=args.seed,
    )
    file_path = _balanced_csv_path(args)
    os.makedirs(args.file_save_path, exist_ok=True)
    sampled.to_csv(file_path, index=False)
    if args.store_slide_ids:
        _save_slide_ids(args, sampled["slide_id"].unique().tolist())
    logger.info("Stored balanced TCGA-UT dataset in %s.", file_path)


def _slides_to_exclude(args: argparse.Namespace) -> list[str]:
    if args.slide_id_exclusion_path is None:
        return []
    with open(args.slide_id_exclusion_path) as file:
        return json.load(file)


def _balanced_csv_path(args: argparse.Namespace) -> str:
    filename = (
        f"TCGA-UT_{args.n_slides_per_class}_slides_per_class_"
        f"{args.n_patches_per_slide}_patches_per_slide_seed={args.seed}.csv"
    )
    return os.path.join(args.file_save_path, filename)


def _save_slide_ids(args: argparse.Namespace, slide_ids: list[str]) -> None:
    filename = (
        f"slide_ids_TCGA-UT_{args.n_slides_per_class}_slides_per_class_"
        f"{args.n_patches_per_slide}_patches_per_slide_seed={args.seed}.json"
    )
    with open(os.path.join(args.file_save_path, filename), "w") as file:
        json.dump(slide_ids, file)


if __name__ == "__main__":
    main()
