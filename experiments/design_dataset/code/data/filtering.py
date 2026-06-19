from typing import cast

import pandas as pd


def filter_synthetic_rows(
    frame: pd.DataFrame,
    include_synthetic: bool,
    synthetic_variant_epochs: int | None,
) -> pd.DataFrame:
    """Return the requested real or synthetic subset for one manifest frame."""
    if "is_synthetic" not in frame.columns:
        return frame
    if synthetic_variant_epochs is not None:
        real = cast(pd.DataFrame, frame[~frame["is_synthetic"].astype(bool)])
        synthetic = cast(pd.DataFrame, frame[frame["is_synthetic"].astype(bool)])
        if synthetic.empty:
            return real
        selected = cast(
            pd.DataFrame,
            synthetic[
                synthetic["final_depth_epochs"].astype(int) == synthetic_variant_epochs
            ],
        )
        return cast(pd.DataFrame, pd.concat([real, selected], ignore_index=True))
    if include_synthetic:
        return frame
    return cast(pd.DataFrame, frame[~frame["is_synthetic"].astype(bool)].copy())
