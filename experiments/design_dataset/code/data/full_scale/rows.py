from typing import cast

import pandas as pd


def slide_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one manifest row per slide."""
    columns = [
        column for column in ["slide_id", "case_id", "cancer_type"] if column in frame
    ]
    if not {"slide_id", "cancer_type"}.issubset(columns):
        raise ValueError("Manifest requires slide_id and cancer_type columns.")
    return cast(pd.DataFrame, frame.loc[:, columns].drop_duplicates(subset="slide_id"))


def rows_for_slides(frame: pd.DataFrame, slide_ids: list[str]) -> pd.DataFrame:
    """Return all rows for selected slides in deterministic slide order."""
    if len(set(slide_ids)) != len(slide_ids):
        return rows_for_slide_instances(frame, slide_ids)
    slide_order = {slide_id: index for index, slide_id in enumerate(slide_ids)}
    rows = cast(pd.DataFrame, frame[frame["slide_id"].isin(slide_ids)].copy())
    rows["_slide_order"] = rows["slide_id"].apply(
        lambda slide_id: slide_order[slide_id]
    )
    sort_columns = ["_slide_order"] + deterministic_row_sort_columns(rows)
    rows = rows.sort_values(by=sort_columns).drop(columns="_slide_order")
    return cast(pd.DataFrame, rows.reset_index(drop=True))


def rows_for_slide_instances(frame: pd.DataFrame, slide_ids: list[str]) -> pd.DataFrame:
    """Return rows for selected slide instances, preserving duplicates."""
    sort_columns = ["slide_id"] + deterministic_row_sort_columns(frame)
    sorted_frame = frame.sort_values(by=sort_columns)
    rows_by_slide = {
        str(slide_id): group.copy()
        for slide_id, group in sorted_frame.groupby("slide_id", sort=False)
    }
    parts = []
    for instance_index, slide_id in enumerate(slide_ids):
        rows = rows_by_slide[str(slide_id)].copy()
        rows["sample_instance"] = instance_index
        parts.append(rows)
    return cast(pd.DataFrame, pd.concat(parts, ignore_index=True))


def cap_rows_per_slide(frame: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Keep a deterministic per-slide row budget."""
    group_columns = (
        ["sample_instance"] if "sample_instance" in frame.columns else ["slide_id"]
    )
    sort_columns = group_columns + ["slide_id"] + deterministic_row_sort_columns(frame)
    sorted_frame = frame.sort_values(by=sort_columns).copy()
    return cast(
        pd.DataFrame,
        sorted_frame.groupby(group_columns, sort=False)
        .head(n_rows)
        .reset_index(drop=True),
    )


def deterministic_row_sort_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that identify a stable within-slide row order."""
    for column in ["patch_id", "feature_id", "image_path", "feature_path"]:
        if column in frame.columns:
            return [column]
    return []
