from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def collect_slide_labels(raw_root: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Collect slide-to-class mapping from raw TCGA-UT folders."""
    labels: dict[str, str] = {}
    conflicts: dict[str, list[str]] = defaultdict(list)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw TCGA-UT root does not exist: {raw_root}")

    for class_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        for split_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            for slide_entry in sorted(split_dir.iterdir()):
                slide_id = slide_entry.name
                existing = labels.get(slide_id)
                if existing is None:
                    labels[slide_id] = class_name
                elif existing != class_name:
                    conflicts[slide_id].extend([existing, class_name])
    return labels, conflicts


def strip_feature_suffix(feature_stem: str, suffix_pattern: str) -> str:
    """Remove feature chunk suffix from a feature identifier."""
    return re.sub(f"{suffix_pattern}$", "", feature_stem)


def _resolve_sources(config: dict) -> tuple[Path, Path, str, str]:
    """Resolve source paths and feature naming settings."""
    raw_root = Path(config["paths"]["raw_root"])
    feature_dir = Path(config["paths"]["feature_dir"])
    feature_glob = str(config["data"].get("feature_glob", "*.pt"))
    suffix_pattern = str(config["data"].get("feature_suffix_pattern", "_[0-9]+"))
    return raw_root, feature_dir, feature_glob, suffix_pattern


def _build_manifest_rows(
    feature_dir: Path,
    feature_glob: str,
    suffix_pattern: str,
    labels: dict[str, str],
) -> tuple[list[dict[str, str]], list[str], set[str]]:
    """Build manifest rows and collect unmatched feature paths."""
    rows: list[dict[str, str]] = []
    unmatched_features: list[str] = []
    matched_slide_ids: set[str] = set()
    for feature_path in sorted(feature_dir.glob(feature_glob)):
        feature_id = feature_path.stem
        slide_id = strip_feature_suffix(feature_id, suffix_pattern)
        class_name = labels.get(slide_id)
        if class_name is None:
            unmatched_features.append(str(feature_path))
            continue
        matched_slide_ids.add(slide_id)
        rows.append(_manifest_row(feature_id, slide_id, class_name, feature_path))
    return rows, unmatched_features, matched_slide_ids


def _manifest_row(
    feature_id: str, slide_id: str, class_name: str, feature_path: Path
) -> dict[str, str]:
    """Create one manifest row."""
    return {
        "feature_id": feature_id,
        "slide_id": slide_id,
        "cancer_type": class_name,
        "feature_path": str(feature_path),
    }


def _slide_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level rows to slide-level manifest."""
    grouped = manifest.groupby(["slide_id", "cancer_type"], as_index=False).agg(
        n_feature_chunks=("feature_id", "count")
    )
    rows = [
        {
            "slide_id": str(row["slide_id"]),
            "cancer_type": str(row["cancer_type"]),
            "n_feature_chunks": int(row["n_feature_chunks"]),
        }
        for _, row in grouped.iterrows()
    ]
    ordered_rows = sorted(
        rows,
        key=lambda row: (row["cancer_type"], row["slide_id"]),
    )
    return pd.DataFrame(ordered_rows)


def _manifest_report(
    manifest: pd.DataFrame,
    slide_manifest: pd.DataFrame,
    class_counts: pd.Series,
    unmatched_features: list[str],
    unmatched_labels: list[str],
    conflicts: dict[str, list[str]],
) -> dict:
    """Build manifest quality report payload."""
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    return {
        "n_feature_rows": int(len(manifest)),
        "n_slides": int(slide_manifest["slide_id"].nunique()),
        "n_classes": int(slide_manifest["cancer_type"].nunique()),
        "imbalance_ratio_slides": imbalance_ratio,
        "min_slides_per_class": int(class_counts.min()),
        "max_slides_per_class": int(class_counts.max()),
        "unmatched_feature_count": len(unmatched_features),
        "unmatched_feature_examples": unmatched_features[:25],
        "label_without_feature_count": len(unmatched_labels),
        "label_without_feature_examples": unmatched_labels[:25],
        "label_conflict_count": len(conflicts),
        "label_conflict_examples": {k: v for k, v in list(conflicts.items())[:25]},
    }


def _validate_manifest(manifest: pd.DataFrame) -> None:
    """Validate required columns and referenced feature files."""
    required_columns = {"feature_id", "slide_id", "cancer_type", "feature_path"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing_columns)}")
    required_frame = manifest.loc[:, list(required_columns)]
    if bool(required_frame.isna().to_numpy().any()):
        raise RuntimeError("Manifest contains empty values in required columns.")
    missing_files = [
        path for path in manifest["feature_path"].tolist() if not Path(path).exists()
    ]
    if missing_files:
        raise RuntimeError(
            f"Manifest references missing features: {missing_files[:10]}"
        )


def _write_manifest_outputs(
    paths: dict, manifest: pd.DataFrame, slide_manifest: pd.DataFrame, report: dict
) -> None:
    """Persist manifest datasets and report files."""
    manifest_path = paths["data"] / "manifest.csv"
    slide_manifest_path = paths["data"] / "slide_manifest.csv"
    report_path = paths["data"] / "manifest_report.json"
    manifest.to_csv(manifest_path, index=False)
    slide_manifest.to_csv(slide_manifest_path, index=False)
    write_json(report_path, report)
    logger.info(f"Wrote {manifest_path}")
    logger.info(f"Wrote {slide_manifest_path}")
    logger.info(f"Wrote {report_path}")


def _build_manifest_payload(
    feature_dir: Path,
    feature_glob: str,
    suffix_pattern: str,
    labels: dict[str, str],
    conflicts: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build manifest data frames and quality report."""
    rows, unmatched_features, matched_slide_ids = _build_manifest_rows(
        feature_dir, feature_glob, suffix_pattern, labels
    )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No feature files could be matched to TCGA-UT labels.")
    _validate_manifest(manifest)
    slide_manifest = _slide_manifest(manifest)
    class_counts = slide_manifest["cancer_type"].value_counts().sort_values()
    unmatched_labels = sorted(set(labels).difference(matched_slide_ids))
    report = _manifest_report(
        manifest,
        slide_manifest,
        class_counts,
        unmatched_features,
        unmatched_labels,
        conflicts,
    )
    return manifest, slide_manifest, report


def main() -> None:
    """Build feature and slide manifests for the TCGA-UT experiment."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    raw_root, feature_dir, feature_glob, suffix_pattern = _resolve_sources(config)
    labels, conflicts = collect_slide_labels(raw_root)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")
    manifest, slide_manifest, report = _build_manifest_payload(
        feature_dir,
        feature_glob,
        suffix_pattern,
        labels,
        conflicts,
    )
    _write_manifest_outputs(paths, manifest, slide_manifest, report)


if __name__ == "__main__":
    main()
