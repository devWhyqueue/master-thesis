"""Prepare native BRACS manifests from the downloaded ROI set and summary sheet."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.bracs.metadata import (
    LABELS,
    index_roi_images,
    load_roi_metadata,
    normalize_label,
)
from data.bracs.splitting import (
    assert_patient_disjoint,
    split_cases,
    write_seed_manifests,
)
from data.bracs.tiling import tile_rois

MIN_NATIVE_IMBALANCE_RATIO = 5.0
__all__ = [
    "assert_patient_disjoint",
    "normalize_label",
    "split_cases",
    "tile_rois",
]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse BRACS preparation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bracs-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--max-tiles-per-roi", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    """Build BRACS native benchmark manifests."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root = Path(args.bracs_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    roi_frame = load_roi_metadata(root, metadata_csv)
    image_index = index_roi_images(root)
    tiled = tile_rois(
        roi_frame,
        image_index,
        output / "tiles",
        int(args.tile_size),
        int(args.max_tiles_per_roi),
    )
    if tiled.empty:
        raise RuntimeError("No BRACS ROI tiles were generated.")
    write_seed_manifests(tiled, output / "manifests", [int(s) for s in args.seeds])
    write_report(root, output, roi_frame, tiled)


def write_report(
    root: Path, output: Path, metadata: pd.DataFrame, tiled: pd.DataFrame
) -> None:
    """Write BRACS preparation metadata and dataset summary."""
    report = _report_payload(root, metadata, tiled)
    _write_json(output / "bracs_prepare_report.json", report)
    (output / "README.md").write_text(_readme_text(report), encoding="utf-8")


def _report_payload(root: Path, metadata: pd.DataFrame, tiled: pd.DataFrame) -> dict:
    counts = (
        tiled.drop_duplicates(["slide_id", "cancer_type"])
        .groupby("cancer_type")["slide_id"]
        .nunique()
        .reindex(LABELS, fill_value=0)
        .astype(int)
    )
    ratio = float(counts.max() / max(1, counts.min()))
    return {
        "dataset": "BRACS",
        "source_root": str(root),
        "n_patients": int(metadata["case_id"].nunique()),
        "n_roi_metadata_rows": int(len(metadata)),
        "n_tiled_rows": int(len(tiled)),
        "n_wsis_with_tiles": int(tiled["slide_id"].nunique()),
        "n_classes": int(tiled["cancer_type"].nunique()),
        "class_counts_wsi": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio_wsi": ratio,
        "min_native_imbalance_ratio": MIN_NATIVE_IMBALANCE_RATIO,
        "recommended_benchmark_mode": (
            "native" if ratio >= MIN_NATIVE_IMBALANCE_RATIO else "power_law"
        ),
        "labels": list(LABELS),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _readme_text(report: dict[str, Any]) -> str:
    return (
        "# BRACS Dataset Staging\n\n"
        "Source: https://www.bracs.icar.cnr.it/\n\n"
        "License: non-commercial research use, Creative Commons Attribution-"
        "NonCommercial 4.0 International.\n\n"
        "Expected source downloads: `BRACS_WSI`, `BRACS_RoI`, "
        "`BRACS_WSI_Annotations`, and `BRACS.xlsx` from "
        "`ftp://histoimage.na.icar.cnr.it/`.\n\n"
        f"Prepared patients: {report['n_patients']}\n\n"
        f"Prepared ROI tiles: {report['n_tiled_rows']}\n"
    )


if __name__ == "__main__":
    main()
