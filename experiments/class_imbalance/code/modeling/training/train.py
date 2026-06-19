from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_progress, write_run_record
from scripts.modeling.mil.bag.trainer import _train_bag_method
from scripts.modeling.mil.metadata import BAG_METHODS, method_metadata
from scripts.analysis.tuning.grid import validate_tuning_params
from scripts.analysis.tuning.paths import tuning_result_dir
from scripts.modeling.training.split import _slice_split_rows

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for method training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tuning-id", default=None)
    parser.add_argument("--tuning-params", default=None)
    return parser.parse_args()


def _train_selected_method(
    method: str,
    frame: pd.DataFrame,
    class_names: list[str],
    config: dict,
    seed: int,
    result_dir: Path,
    smoke: bool = False,
) -> tuple[dict[str, dict[str, object]], dict[str, int] | None]:
    if method in BAG_METHODS:
        return _train_bag_method(
            method, frame, class_names, config, seed, result_dir, smoke
        )
    raise ValueError(f"Unknown WSI-bag benchmark method: {method}")


def _load_split(
    paths: dict[str, Path], seed: int, smoke: bool, config: dict
) -> pd.DataFrame:
    """Load the shared slide-level split manifest for WSI-bag training."""
    frame = pd.read_csv(paths["data"] / f"manifest_splits_seed={seed}.csv")
    max_train = config["wsi_training"].get("max_train_rows")
    max_eval = config["wsi_training"].get("max_eval_rows")
    if smoke:
        max_train = min(int(max_train or 8), 8)
        max_eval = min(int(max_eval or 4), 4)
    return _slice_split_rows(frame, max_train, max_eval)


def main() -> None:
    """Train one mitigation method and write result artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    _run(args)


def _run(args: argparse.Namespace) -> None:
    config, tuning_params = _load_config_with_tuning(args)
    paths = ensure_dirs(config)
    result_dir = _result_dir(paths, args.method, args.seed, args.tuning_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_split(paths, args.seed, args.smoke, config)
    class_names = sorted(frame["cancer_type"].unique().tolist())
    _write_status(result_dir, args.method, args.seed, "started")
    results, diagnostics = _train_selected_method(
        args.method, frame, class_names, config, args.seed, result_dir, args.smoke
    )
    _write_outputs(
        result_dir,
        args.method,
        args.seed,
        args.smoke,
        results,
        args.tuning_id,
        tuning_params,
        diagnostics,
    )
    _write_status(result_dir, args.method, args.seed, "completed")
    logger.info(f"Wrote results to {result_dir}")


def _load_config_with_tuning(args: argparse.Namespace) -> tuple[dict, dict[str, float]]:
    config = load_config(args.config)
    tuning_params = _load_tuning_params(args.method, args.tuning_params)
    return _with_tuning_params(config, tuning_params), tuning_params


def _write_outputs(
    result_dir: Path,
    method: str,
    seed: int,
    smoke: bool,
    results: dict[str, dict[str, object]],
    tuning_id: str | None,
    tuning_params: dict[str, float],
    diagnostics: dict[str, int] | None,
) -> None:
    """Write the consolidated per-run record."""
    write_run_record(
        result_dir,
        {
            "benchmark": "wsi_bag",
            "method": method,
            "seed": seed,
            "smoke": smoke,
            "tuning_id": tuning_id,
            "tuning_params": tuning_params,
            "model_path": "model.pt",
            "method_metadata": method_metadata(method),
            "diagnostics": diagnostics,
            "splits": results,
        },
    )


def _write_status(result_dir: Path, method: str, seed: int, status: str) -> None:
    """Write coarse progress for methods without epoch-level training."""
    write_progress(
        result_dir / "progress.json",
        {
            "method": method,
            "seed": seed,
            "status": status,
        },
    )
    logger.info("progress method=%s seed=%s status=%s", method, seed, status)


def _load_tuning_params(method: str, raw: str | None) -> dict[str, float]:
    if raw is None:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--tuning-params must be a JSON object")
    validate_tuning_params("wsi_bag", method, payload)
    return {str(key): float(value) for key, value in payload.items()}


def _with_tuning_params(
    config: dict[str, Any], params: dict[str, float]
) -> dict[str, Any]:
    copied = dict(config)
    copied["wsi_training"] = dict(config["wsi_training"])
    copied["wsi_training"].update(params)
    return copied


def _result_dir(
    paths: dict[str, Path], method: str, seed: int, tuning_id: str | None
) -> Path:
    if tuning_id is None:
        return paths["wsi_results"] / method / f"seed={seed}"
    return tuning_result_dir(paths, "wsi_bag", method, tuning_id, seed)


if __name__ == "__main__":
    main()
