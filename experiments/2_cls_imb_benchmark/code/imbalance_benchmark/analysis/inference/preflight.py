from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.bootstrap import (
    _class_preflight,
    _preflight_row_weights,
)


def run_preflight(
    identity: pd.DataFrame,
    n_replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per split-frame, by-class preflight: unique resampled patients, Kish, max weight.

    Kish and max-weight-fraction are averaged across replicates and flagged
    descriptive-only when the mean Kish count is below five, or when one
    patient supplies more than 50% of a class's weight in more than 5% of
    replicates, per report §"Imbalance deficit, recovery, and inference".
    """
    row_weights = _preflight_row_weights(identity, n_replicates, seed)
    class_col = identity["cancer_type"].to_numpy()
    by_class = {
        str(cls): _class_preflight(
            rows["case_id"].to_numpy(), row_weights[class_col == cls, :], n_replicates
        )
        for cls, rows in identity.groupby("cancer_type")
    }
    split_col = "patient_split" if "patient_split" in identity else None
    by_split_class: dict[str, dict[str, Any]] = {}
    split_values = identity[split_col].astype(str).unique() if split_col else ["0"]
    for split in sorted(split_values):
        split_mask = (
            identity[split_col].astype(str).to_numpy() == split
            if split_col
            else np.ones(len(identity), dtype=bool)
        )
        by_split_class[str(split)] = {}
        for cls in sorted(identity.loc[split_mask, "cancer_type"].astype(str).unique()):
            mask = split_mask & (class_col == cls)
            diagnostic = _class_preflight(
                identity.loc[mask, "case_id"].to_numpy(),
                row_weights[mask, :],
                n_replicates,
            )
            # Every original class has positive total weight in every replicate
            # because resampling occurs within complete contribution strata.
            diagnostic["all_replicates_represented"] = bool(
                (row_weights[mask, :].sum(axis=0) > 0).all()
            )
            diagnostic["metric_computable"] = diagnostic["all_replicates_represented"]
            by_split_class[str(split)][cls] = diagnostic
    multiplicity_consistent = bool(
        all(
            np.all(
                row_weights[identity["case_id"].to_numpy() == case, :]
                == row_weights[
                    np.flatnonzero(identity["case_id"].to_numpy() == case)[0], :
                ]
            )
            for case in identity["case_id"].astype(str).unique()
        )
    )
    metrics_computable = all(
        value["metric_computable"]
        for split in by_split_class.values()
        for value in split.values()
    )
    return {
        "n_replicates": n_replicates,
        "seed": seed,
        "by_class": by_class,
        "by_split_class": by_split_class,
        "all_split_level_metrics_computable": metrics_computable,
        "identical_multiplicities_across_split_appearances": multiplicity_consistent,
        "is_descriptive_only": any(v["is_descriptive_only"] for v in by_class.values()),
    }
