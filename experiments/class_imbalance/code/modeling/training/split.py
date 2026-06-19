from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _slice_split_rows(
    frame: pd.DataFrame, max_train: int | None, max_eval: int | None
) -> pd.DataFrame:
    parts = []
    split_limits = [("train", max_train), ("val", max_eval), ("test", max_eval)]
    for split, max_rows in split_limits:
        split_frame = frame[frame["split"] == split]
        if max_rows:
            split_frame = split_frame.groupby("cancer_type", group_keys=False).head(
                int(max_rows)
            )
        parts.append(split_frame)
    return pd.concat(parts, ignore_index=True)
