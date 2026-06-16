from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


_REFERENCE_FINAL_DEPTH_EPOCHS = 25


def summary_path(paths: dict[str, Path], seed: int, variant: int | None = None) -> Path:
    """Return the stored ProGAN summary JSON for one benchmark seed."""
    ref = variant if variant is not None else _REFERENCE_FINAL_DEPTH_EPOCHS
    candidates = [
        paths["root"]
        / "synthetic_patch_images"
        / f"seed={seed}"
        / f"epochs={ref}"
        / "synthetic_patch_summary.json",
        paths["patch_results"]
        / "patch_progan_aug"
        / f"seed={seed}"
        / "synthetic_patch_summary.json",
        paths["root"]
        / "synthetic_patch_images"
        / f"seed={seed}"
        / "synthetic_patch_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No ProGAN summary found for seed={seed}")


def load_summary(path: Path) -> dict[str, Any]:
    """Load one ProGAN summary JSON payload."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def pretty_class_name(name: str) -> str:
    """Render a class identifier for tables and figure labels."""
    return name.replace("_", " ")


def _summary_rows(summary: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary["per_class"]:
        fid = item.get("fid", {})
        rows.append(
            {
                "seed": seed,
                "class_name": item["class_name"],
                "real_train_patches": int(item["real_train_patches"]),
                "generated_patches": int(item["generated_patches"]),
                "balance_target": int(item["balance_target"]),
                "inception_fid": fid.get("value"),
                "fid_status": fid.get("status"),
            }
        )
    return rows


def _feature_cache_frame(cache_dir: Path) -> tuple[pd.DataFrame, np.memmap]:
    manifest = pd.read_csv(cache_dir / "manifest.csv", low_memory=False)
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    return manifest, features


def _class_feature_groups(
    manifest: pd.DataFrame,
    features: np.memmap,
    class_name: str,
    variant: int = _REFERENCE_FINAL_DEPTH_EPOCHS,
) -> tuple[np.ndarray, np.ndarray]:
    frame = cast(pd.DataFrame, manifest[manifest["cancer_type"] == class_name])
    real = frame[~frame["is_synthetic"].astype(bool)]
    synthetic_all = frame[frame["is_synthetic"].astype(bool)]
    if "final_depth_epochs" in synthetic_all.columns:
        synthetic = cast(
            pd.DataFrame,
            synthetic_all[synthetic_all["final_depth_epochs"].astype(int) == variant],
        )
    else:
        synthetic = synthetic_all
    real_idx = np.asarray(real["feature_index"], dtype=np.int64)
    synthetic_idx = np.asarray(synthetic["feature_index"], dtype=np.int64)
    real_features = np.asarray(features[real_idx], dtype=np.float32)
    synthetic_features = np.asarray(features[synthetic_idx], dtype=np.float32)
    return real_features, synthetic_features


def _mean_nearest_neighbor_distance(
    real_features: np.ndarray, synthetic_features: np.ndarray, sample_size: int = 64
) -> float | None:
    if len(real_features) == 0 or len(synthetic_features) == 0:
        return None
    rng = np.random.default_rng(0)
    if len(real_features) > 512:
        real_features = real_features[
            rng.choice(len(real_features), size=512, replace=False)
        ]
    if len(synthetic_features) > sample_size:
        synthetic_features = synthetic_features[
            rng.choice(len(synthetic_features), size=sample_size, replace=False)
        ]
    min_distances: list[float] = []
    for start in range(0, len(synthetic_features), 16):
        chunk = synthetic_features[start : start + 16]
        distances = np.linalg.norm(
            chunk[:, None, :] - real_features[None, :, :], axis=2
        )
        min_distances.extend(distances.min(axis=1).tolist())
    return float(np.mean(min_distances))


def build_metrics_frame(
    paths: dict[str, Path],
    seed: int,
    cache_dir: Path | None,
    variant: int = _REFERENCE_FINAL_DEPTH_EPOCHS,
) -> pd.DataFrame:
    """Combine stored ProGAN summaries with Virchow2 nearest-neighbor distances.

    variant selects which epoch snapshot to report; defaults to the reference (25 epochs).
    """
    summary = load_summary(summary_path(paths, seed, variant))
    frame = pd.DataFrame(_summary_rows(summary, seed))
    if cache_dir is None or not (cache_dir / "manifest.csv").exists():
        frame["virchow_mean_nn_distance"] = np.nan
        return frame
    manifest, features = _feature_cache_frame(cache_dir)
    distances = []
    for class_name in frame["class_name"]:
        real_features, synthetic_features = _class_feature_groups(
            manifest, features, str(class_name), variant=variant
        )
        distances.append(
            _mean_nearest_neighbor_distance(real_features, synthetic_features)
        )
    frame["virchow_mean_nn_distance"] = distances
    return frame


def _format_metric_row(row: dict[str, Any]) -> str:
    fid_value = row["inception_fid"]
    nn_value = row["virchow_mean_nn_distance"]
    fid_text = "--" if pd.isna(fid_value) else rf"\num{{{float(fid_value):.1f}}}"
    nn_text = "--" if pd.isna(nn_value) else rf"\num{{{float(nn_value):.2f}}}"
    return (
        f"{pretty_class_name(str(row['class_name']))} & "
        rf"\num{{{int(row['real_train_patches'])}}} & "
        rf"\num{{{int(row['generated_patches'])}}} & "
        f"{fid_text} & {nn_text}\\\\"
    )


def _latex_summary_footer(
    frame: pd.DataFrame, fid: pd.Series, nn: pd.Series
) -> list[str]:
    return [
        "\\midrule",
        f"All augmented classes ({len(frame)} total) & "
        rf"\num{{{int(frame['real_train_patches'].sum())}}} & "
        rf"\num{{{int(frame['generated_patches'].sum())}}} & "
        rf"\num{{{float(fid.median()):.1f}}} median & "
        rf"\num{{{float(nn.median()):.2f}}} median\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "",
    ]


def write_summary_latex(frame: pd.DataFrame, path: Path) -> None:
    """Write the paper table for per-class ProGAN quality diagnostics."""
    fid = cast(pd.Series, frame["inception_fid"].dropna())
    nn = cast(pd.Series, frame["virchow_mean_nn_distance"].dropna())
    lines = [
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xrrrr@{}}",
        "\\toprule",
        "Class & Real train & Generated & Inception FID & Virchow2 mean NN\\\\",
        "\\midrule",
    ]
    display = frame.sort_values("generated_patches", ascending=False).head(8)
    for row in display.to_dict("records"):
        lines.append(_format_metric_row(row))
    lines.extend(_latex_summary_footer(frame, fid, nn))
    path.write_text("\n".join(lines), encoding="utf-8")
