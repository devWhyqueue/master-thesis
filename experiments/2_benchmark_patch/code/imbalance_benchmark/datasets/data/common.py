from __future__ import annotations

from typing import Any, cast

import pandas as pd


def class_names(values: pd.Series) -> list[str]:
    """Return the canonical class order for a manifest label column."""
    names = sorted(values.astype(str).unique().tolist())
    return (
        sorted(names, key=lambda name: int(name.removeprefix("ISUP")))
        if names and all(name.startswith("ISUP") for name in names)
        else names
    )


def validate_class_names(frame: pd.DataFrame, locked: list[str]) -> None:
    """Reject manifest labels that are absent from the locked class set."""
    unexpected = sorted(set(frame["cancer_type"].astype(str)) - set(locked))
    if unexpected:
        raise ValueError(
            f"Manifest contains classes absent from the locked target: {unexpected}"
        )


def slide_level_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse feature chunks to one labelled row per slide after consistency checks."""
    label_counts = cast(
        pd.Series, frame.groupby("slide_id", sort=False)["cancer_type"].nunique()
    )
    mixed = cast(pd.Series, label_counts[label_counts != 1])
    if not mixed.empty:
        slides = list(cast(Any, mixed.index))[:5]
        raise ValueError(
            f"Each WSI must have exactly one class; mixed labels: {slides}"
        )
    case_counts = cast(
        pd.Series, frame.groupby("slide_id", sort=False)["case_id"].nunique()
    )
    inconsistent = cast(pd.Series, case_counts[case_counts != 1])
    if not inconsistent.empty:
        slides = list(cast(Any, inconsistent.index))[:5]
        raise ValueError(
            f"Each WSI must have exactly one patient; inconsistent slides: {slides}"
        )
    return frame.drop_duplicates("slide_id", keep="first").reset_index(drop=True)
