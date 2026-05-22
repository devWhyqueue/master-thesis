from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def summary_path(paths: dict[str, Path], seed: int) -> Path:
    """Return the stored ProGAN summary JSON for one benchmark seed."""
    candidates = [
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
    manifest: pd.DataFrame, features: np.memmap, class_name: str
) -> tuple[np.ndarray, np.ndarray]:
    frame = cast(pd.DataFrame, manifest[manifest["cancer_type"] == class_name])
    real = frame[~frame["is_synthetic"].astype(bool)]
    synthetic = frame[frame["is_synthetic"].astype(bool)]
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
    paths: dict[str, Path], seed: int, cache_dir: Path | None
) -> pd.DataFrame:
    """Combine stored ProGAN summaries with Virchow2 nearest-neighbor distances."""
    summary = load_summary(summary_path(paths, seed))
    frame = pd.DataFrame(_summary_rows(summary, seed))
    if cache_dir is None or not (cache_dir / "manifest.csv").exists():
        frame["virchow_mean_nn_distance"] = np.nan
        return frame
    manifest, features = _feature_cache_frame(cache_dir)
    distances = []
    for class_name in frame["class_name"]:
        real_features, synthetic_features = _class_feature_groups(
            manifest, features, str(class_name)
        )
        distances.append(
            _mean_nearest_neighbor_distance(real_features, synthetic_features)
        )
    frame["virchow_mean_nn_distance"] = distances
    return frame


def _format_metric_row(row: dict[str, Any]) -> str:
    fid_value = row["inception_fid"]
    nn_value = row["virchow_mean_nn_distance"]
    fid_text = "--" if pd.isna(fid_value) else f"{float(fid_value):.1f}"
    nn_text = "--" if pd.isna(nn_value) else f"{float(nn_value):.2f}"
    return (
        f"{pretty_class_name(str(row['class_name']))} & "
        f"{int(row['real_train_patches'])} & "
        f"{int(row['generated_patches'])} & "
        f"{fid_text} & {nn_text}\\\\"
    )


def _latex_summary_footer(
    frame: pd.DataFrame, fid: pd.Series, nn: pd.Series
) -> list[str]:
    return [
        "\\midrule",
        f"All augmented classes ({len(frame)} total) & "
        f"{int(frame['real_train_patches'].sum())} & "
        f"{int(frame['generated_patches'].sum())} & "
        f"{float(fid.median()):.1f} median & "
        f"{float(nn.median()):.2f} median\\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]


def write_summary_latex(frame: pd.DataFrame, path: Path) -> None:
    """Write the paper table for per-class ProGAN quality diagnostics."""
    fid = frame["inception_fid"].dropna()
    nn = frame["virchow_mean_nn_distance"].dropna()
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Class & Real train & Generated & Inception FID & Virchow2 mean NN\\\\",
        "\\midrule",
    ]
    display = frame.sort_values("generated_patches", ascending=False).head(8)
    for row in display.to_dict("records"):
        lines.append(_format_metric_row(row))
    lines.extend(_latex_summary_footer(frame, fid, nn))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(frame: pd.DataFrame, path: Path, seed: int) -> None:
    """Write machine-readable ProGAN diagnostic summaries."""
    payload = {
        "seed": seed,
        "n_augmented_classes": int(len(frame)),
        "total_generated_patches": int(frame["generated_patches"].sum()),
        "total_real_train_patches_in_augmented_classes": int(
            frame["real_train_patches"].sum()
        ),
        "inception_fid_median": float(frame["inception_fid"].median()),
        "inception_fid_min": float(frame["inception_fid"].min()),
        "inception_fid_max": float(frame["inception_fid"].max()),
        "virchow_mean_nn_median": float(frame["virchow_mean_nn_distance"].median()),
        "virchow_mean_nn_min": float(frame["virchow_mean_nn_distance"].min()),
        "virchow_mean_nn_max": float(frame["virchow_mean_nn_distance"].max()),
        "per_class": frame.to_dict("records"),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
