from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.db import connect_db
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.analysis.query import load_eval_details, load_seed_predictions
from imbalance_benchmark.common import split_paths


def crossed_ece_distribution(
    split_predictions: list[tuple[np.ndarray, np.ndarray]],
    contexts: list[BootstrapContext],
) -> list[float]:
    """Average split ECE within each shared crossed-bootstrap replicate."""
    if len(split_predictions) != len(contexts) or not contexts:
        raise ValueError(
            "ECE aggregation requires one bootstrap context per patient split"
        )
    distributions = [
        context.ece_distribution(labels, probabilities)
        for (labels, probabilities), context in zip(
            split_predictions, contexts, strict=True
        )
    ]
    return np.mean(np.stack(distributions), axis=0).tolist()


def write_crossed_calibration_table(
    base_paths: dict[str, Path], config: dict[str, Any], seed: int
) -> None:
    """Write the final ECE table using the shared three-split crossed bootstrap."""
    rows = _crossed_calibration_rows(base_paths, config, seed)
    table = pd.DataFrame(rows)
    body = (
        "\\multicolumn{1}{c}{No confirmed runs ingested yet.}"
        if table.empty
        else table.to_latex(index=False, float_format="%.3f", escape=True)
    )
    text = (
        "% Raw and temperature-scaled calibration summary with crossed ECE intervals\n"
        "\\begin{table}[ht]\n\\centering\n"
        f"{body}\n"
        "\\caption{Calibration summary with crossed patient-bootstrap ECE intervals}\n"
        "\\label{tab:calibration}\n\\end{table}\n"
    )
    base_paths["tables"].mkdir(parents=True, exist_ok=True)
    (base_paths["tables"] / "calibration_table.tex").write_text(text, encoding="utf-8")


def _crossed_calibration_rows(
    base_paths: dict[str, Path], config: dict[str, Any], seed: int
) -> list[dict[str, object]]:
    keys = _complete_result_keys(base_paths)
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    n_replicates = int(config.get("analysis", {}).get("bootstrap_replicates", 10_000))
    rows = []
    for assignment, condition, method in sorted(keys):
        distributions = _ece_distributions(
            base_paths, is_mil, n_replicates, seed, assignment, condition, method
        )
        rows.append(
            {
                "assignment": assignment,
                "condition": condition,
                "method": method,
                **_distribution_summary(distributions["probs"], "ECE"),
                **_distribution_summary(
                    distributions["temperature_scaled_probs"], "Temperature ECE"
                ),
            }
        )
    return rows


def _complete_result_keys(base_paths: dict[str, Path]) -> set[tuple[str, str, str]]:
    keys_by_split = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        conn = connect_db(paths["db"])
        try:
            details = load_eval_details(conn)
        finally:
            conn.close()
        test = cast(pd.DataFrame, details[details["split"] == "test"])
        key_frame = cast(pd.DataFrame, test[["assignment", "condition", "method"]])
        keys_by_split.append(set(key_frame.itertuples(index=False, name=None)))
    return set.intersection(*keys_by_split) if keys_by_split else set()


def _ece_distributions(
    base_paths: dict[str, Path],
    is_mil: bool,
    n_replicates: int,
    seed: int,
    assignment: str,
    condition: str,
    method: str,
) -> dict[str, list[float]]:
    records = []
    contexts = []
    for index in range(3):
        paths = split_paths(base_paths, index)
        predictions = load_seed_predictions(paths, condition, method, assignment)
        if predictions is None:
            raise RuntimeError(
                f"Missing confirmed predictions for {assignment}/{condition}/{method}"
            )
        records.append(predictions)
        contexts.append(BootstrapContext(paths, is_mil, n_replicates, seed))
    labels = [np.asarray(record["labels"]) for record in records]
    return {
        key: crossed_ece_distribution(
            list(
                zip(
                    labels, [np.asarray(record[key]) for record in records], strict=True
                )
            ),
            contexts,
        )
        for key in ("probs", "temperature_scaled_probs")
    }


def _distribution_summary(values: list[float], name: str) -> dict[str, object]:
    array = np.asarray(values)
    replicates = array[1:] if len(array) > 1 else array
    low, high = np.percentile(replicates, [2.5, 97.5])
    return {name: float(array[0]), f"{name} 95% CI": f"[{low:.3f}, {high:.3f}]"}
