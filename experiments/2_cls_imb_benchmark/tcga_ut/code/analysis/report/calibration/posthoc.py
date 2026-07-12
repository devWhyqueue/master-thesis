"""Fit post-hoc calibrators on validation logits and report held-out test metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.common import ensure_dirs, load_config
from code.analysis.results import (
    connect,
    init_schema,
    load_split_payload,
    replace_table,
)
from code.analysis.report.calibration.utils import (
    calibrated_probabilities,
    fit_dirichlet,
    fit_temperature,
    fit_vector_scaling,
    metric_bundle,
    probabilities_to_logits,
    reliability_curve,
)
from code.analysis.report.figures.labels import METHOD_LABELS

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATORS = ("temperature",)
ALL_CALIBRATORS = ("temperature", "vector", "dirichlet")
PATCH_METHODS = ("patch_feature_ce", "patch_feature_cfal")


def parse_args() -> argparse.Namespace:
    """Parse post-hoc calibration arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibrators",
        nargs="*",
        default=list(DEFAULT_CALIBRATORS),
        choices=list(ALL_CALIBRATORS),
        help="Calibrators to fit.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(PATCH_METHODS),
        help="Methods to calibrate.",
    )
    return parser.parse_args()


def _load_split(connection, method: str, seed: int, split: str) -> dict[str, Any]:
    if (
        payload := load_split_payload(connection, "patch", method, seed, split)
    ) is None:
        raise FileNotFoundError(
            f"Missing patch result for {method} seed={seed} split={split}"
        )
    return payload


def _arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    return (
        np.asarray(payload["labels"], dtype=np.int64),
        np.asarray(payload["probabilities"], dtype=np.float64),
        len(payload["class_names"]),
    )


def _fit_and_evaluate(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
    calibrator: str,
) -> dict[str, dict[str, float]]:
    fit_fns = {
        "temperature": fit_temperature,
        "vector": fit_vector_scaling,
        "dirichlet": fit_dirichlet,
    }
    if calibrator not in fit_fns:
        raise ValueError(f"Unsupported calibrator: {calibrator}")
    fit = fit_fns[calibrator](val_logits, val_labels)
    val_probs = calibrated_probabilities(val_logits, fit)
    test_probs = calibrated_probabilities(test_logits, fit)
    return {
        "val": metric_bundle(val_probs, val_labels, n_classes),
        "test": metric_bundle(test_probs, test_labels, n_classes),
    }


def _plot_reliability(
    method_payloads: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.0, label="Perfect"
    )
    for method, (probabilities, labels) in method_payloads.items():
        _, mean_confidence, accuracy = reliability_curve(probabilities, labels)
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            mean_confidence,
            accuracy,
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Patch reliability diagram (seed {seed}, test split)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_latex_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Method & Calibrator & Split & NLL & Brier & ECE\\\\",
        "\\midrule",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"{METHOD_LABELS.get(row['method'], row['method'])} & "
            f"{row['calibrator']} & {row['split']} & "
            f"\\num{{{row['negative_log_likelihood']:.3f}}} & "
            f"\\num{{{row['brier_score']:.3f}}} & "
            f"\\num{{{row['expected_calibration_error']:.3f}}}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _rows_for_method(
    connection,
    method: str,
    seed: int,
    calibrators: list[str],
) -> tuple[list[dict[str, object]], tuple[np.ndarray, np.ndarray]]:
    """Return calibration metric rows and raw test probabilities for one method."""
    val_payload = _load_split(connection, method, seed, "val")
    test_payload = _load_split(connection, method, seed, "test")
    val_labels, val_probs, n_classes = _arrays(val_payload)
    test_labels, test_probs, _ = _arrays(test_payload)
    val_logits = probabilities_to_logits(val_probs)
    test_logits = probabilities_to_logits(test_probs)
    rows: list[dict[str, object]] = []
    for calibrator in ("raw", *calibrators):
        if calibrator == "raw":
            metrics = {
                "val": metric_bundle(val_probs, val_labels, n_classes),
                "test": metric_bundle(test_probs, test_labels, n_classes),
            }
        else:
            metrics = _fit_and_evaluate(
                val_logits,
                val_labels,
                test_logits,
                test_labels,
                n_classes,
                calibrator,
            )
        for split, values in metrics.items():
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "calibrator": calibrator,
                    "split": split,
                    **values,
                }
            )
    return rows, (test_probs, test_labels)


def _collect_rows(
    connection, methods: list[str], seed: int, calibrators: list[str]
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Build the calibration table and reliability-diagram inputs."""
    rows: list[dict[str, object]] = []
    reliability_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in methods:
        method_rows, reliability_inputs[method] = _rows_for_method(
            connection, method, seed, calibrators
        )
        rows.extend(method_rows)
    return pd.DataFrame(rows), reliability_inputs


def _publish_artifacts(
    frame: pd.DataFrame,
    reliability_inputs: dict[str, tuple[np.ndarray, np.ndarray]],
    paths: dict[str, Path],
    seed: int,
) -> None:
    """Write CSV, LaTeX, and reliability-diagram outputs."""
    paths["tables"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    connection = connect(paths["db"])
    init_schema(connection)
    replace_table(connection, "posthoc_calibration", frame)
    connection.close()
    test_rows = cast(
        pd.DataFrame, frame[(frame["seed"] == seed) & (frame["split"] == "test")]
    )
    _write_latex_table(
        test_rows, paths["tables"] / "result_posthoc_calibration_test.tex"
    )
    fig_path = paths["figures"] / f"reliability_patch_ce_cfal_seed{seed}.png"
    _plot_reliability(reliability_inputs, seed, fig_path)


def main() -> None:
    """Fit post-hoc calibrators and write report artifacts."""
    args = parse_args()
    paths = ensure_dirs(load_config(args.config))
    connection = connect(paths["db"])
    init_schema(connection)
    methods, calibs = list(args.methods), list(args.calibrators)
    frame, reliability_inputs = _collect_rows(connection, methods, args.seed, calibs)
    connection.close()
    _publish_artifacts(frame, reliability_inputs, paths, args.seed)


if __name__ == "__main__":
    main()
