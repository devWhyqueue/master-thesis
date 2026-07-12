"""Select the stratified PANDA slide subset shared by tiling and features.

Writes ``<output_root>/selected_slides.csv`` (one row per chosen slide) so that
the tiling array, feature array, and prepare step all agree on the same cohort.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data.panda.metadata import load_slide_frame, select_subset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse PANDA subset-selection arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-slides", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Write the selected-slides manifest for the PANDA benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    frame = load_slide_frame(Path(args.data_root))
    subset = select_subset(frame, args.n_slides, args.seed)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selected_slides.csv"
    subset.to_csv(path, index=False)
    logger.info("Selected %d/%d slides -> %s", len(subset), len(frame), path)
    logger.info("ISUP counts: %s", subset["slide_label"].value_counts().to_dict())
    logger.info("Provider counts: %s", subset["provider"].value_counts().to_dict())


if __name__ == "__main__":
    main()
