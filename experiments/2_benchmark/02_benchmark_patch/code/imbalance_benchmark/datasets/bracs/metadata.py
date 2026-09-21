"""BRACS ROI metadata and patient-disjoint splitting."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

LABELS = ("N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC")
LESION_TYPES = {
    "N": "benign",
    "PB": "benign",
    "UDH": "benign",
    "FEA": "atypical",
    "ADH": "atypical",
    "DCIS": "malignant",
    "IC": "malignant",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_LABEL_ALIASES = {
    "NORMAL": "N",
    "N": "N",
    "PB": "PB",
    "PATHOLOGICALBENIGN": "PB",
    "UDH": "UDH",
    "USUALDUCTALHYPERPLASIA": "UDH",
    "FEA": "FEA",
    "FLATEPITHELIALATYPIA": "FEA",
    "ADH": "ADH",
    "ATYPICALDUCTALHYPERPLASIA": "ADH",
    "DCIS": "DCIS",
    "DUCTALCARCINOMAINSITU": "DCIS",
    "IC": "IC",
    "INVASIVECARCINOMA": "IC",
}
_ROI_STEM = re.compile(rf"^(?P<slide_id>BRACS_\d+)_(?P<label>{'|'.join(LABELS)})_\d+$")


def normalize_label(value: object) -> str | None:
    """Normalize BRACS subtype labels to the seven report classes."""
    text = str(value).strip()
    upper = re.sub(r"[^A-Z0-9]+", "", text.upper())
    return _LABEL_ALIASES.get(upper)


def find_summary_file(root: Path) -> Path:
    """Return the BRACS summary spreadsheet path."""
    matches = list(root.rglob("BRACS.xlsx"))
    if not matches:
        raise FileNotFoundError(f"BRACS.xlsx not found under {root}")
    return matches[0]


def index_roi_images(root: Path) -> dict[str, Path]:
    """Index ROI images by filename stem."""
    roi_roots = [path for path in root.rglob("BRACS_RoI") if path.is_dir()]
    latest_roots = [path / "latest_version" for path in roi_roots]
    search_roots = (
        [path for path in latest_roots if path.is_dir()] or roi_roots or [root]
    )
    index: dict[str, Path] = {}
    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem, path)
    if not index:
        raise FileNotFoundError(f"No ROI images found under {search_roots}")
    return index


def load_roi_metadata(root: Path, metadata_csv: Path | None = None) -> pd.DataFrame:
    """Load and normalize ROI metadata from a pre-converted CSV or BRACS.xlsx."""
    if metadata_csv is not None and metadata_csv.exists():
        return cast(pd.DataFrame, pd.read_csv(metadata_csv, dtype={"case_id": str}))
    workbook = find_summary_file(root)
    sheets = pd.read_excel(workbook, sheet_name=None)
    frames = [
        frame
        for frame in (_normalize_sheet(sheet) for sheet in sheets.values())
        if not frame.empty
    ]
    if not frames:
        columns = {name: list(sheet.columns) for name, sheet in sheets.items()}
        raise ValueError(f"Could not identify a BRACS ROI metadata sheet: {columns}")
    slides = max(frames, key=len).drop_duplicates(subset=["slide_id"])
    frame = _roi_metadata(slides, index_roi_images(root))
    if frame.empty:
        raise ValueError("BRACS ROI filenames did not match workbook slides.")
    frame["dataset"] = "bracs"
    frame["lesion_type"] = frame["cancer_type"].map(lambda label: LESION_TYPES[label])
    return cast(pd.DataFrame, frame)


def _normalize_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    raw = sheet.dropna(how="all").copy()
    if raw.empty:
        return pd.DataFrame()
    columns = {_canonical(column): column for column in raw.columns}
    slide_col = _first(columns, ("wsi", "slide", "slide_id", "wsi_filename"))
    case_col = _first(columns, ("patient", "patient_id", "case", "case_id"))
    if slide_col is None or case_col is None:
        return pd.DataFrame()
    return pd.DataFrame(
        row
        for _, raw_row in raw.iterrows()
        if (row := _slide_metadata_row(raw_row, slide_col, case_col))
    )


def _slide_metadata_row(
    row: pd.Series, slide_col: object, case_col: object
) -> dict[str, str] | None:
    slide_id, case_id = _clean_id(row[slide_col]), _clean_id(row[case_col])
    if not slide_id or not case_id:
        return None
    return {"case_id": case_id, "slide_id": slide_id}


def _roi_metadata(slides: pd.DataFrame, image_index: dict[str, Path]) -> pd.DataFrame:
    cases = dict(zip(slides["slide_id"], slides["case_id"], strict=True))
    rows = []
    for roi_id in sorted(image_index):
        match = _ROI_STEM.fullmatch(roi_id)
        if match and (case_id := cases.get(match["slide_id"])):
            rows.append(
                {
                    "case_id": case_id,
                    "slide_id": match["slide_id"],
                    "roi_id": roi_id,
                    "cancer_type": match["label"],
                }
            )
    return pd.DataFrame(rows)


def _canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _first(columns: dict[str, object], names: tuple[str, ...]) -> object | None:
    for name in names:
        if name in columns:
            return columns[name]
    for key, column in columns.items():
        if any(name in key for name in names):
            return column
    return None


def _clean_id(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() == "nan" else Path(text).stem


def split_cases(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return approximate stratified patient-level split assignments."""
    case_labels = (
        frame.drop_duplicates(["case_id", "slide_id", "cancer_type"])
        .groupby("case_id")["cancer_type"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    rows = []
    for _, group in case_labels.groupby("cancer_type", sort=False):
        rows.extend(_split_group(group, rng))
    return pd.DataFrame(rows)


def assert_patient_disjoint(frame: pd.DataFrame) -> None:
    """Raise if any patient appears in multiple split labels."""
    split_counts = cast(pd.Series, frame.groupby("case_id")["split"].nunique())
    leaking = [
        str(case_id) for case_id, count in split_counts.items() if int(count) > 1
    ]
    if leaking:
        raise ValueError(f"BRACS patient leakage: {leaking[:5]}")


def _split_group(group: pd.DataFrame, rng: np.random.Generator) -> list[dict[str, str]]:
    cases = group["case_id"].astype(str).to_numpy()
    rng.shuffle(cases)
    n_cases = len(cases)
    n_train = (
        max(1, int(round(n_cases * 0.70))) if n_cases >= 3 else max(1, n_cases - 1)
    )
    n_val = max(1, int(round(n_cases * 0.15))) if n_cases >= 3 else 0
    if n_train + n_val >= n_cases and n_cases > 1:
        n_val = max(0, n_cases - n_train - 1)
    return (
        [{"case_id": case, "split": "train"} for case in cases[:n_train]]
        + [
            {"case_id": case, "split": "validation"}
            for case in cases[n_train : n_train + n_val]
        ]
        + [{"case_id": case, "split": "test"} for case in cases[n_train + n_val :]]
    )
