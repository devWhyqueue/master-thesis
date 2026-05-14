from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.common import ensure_dirs, load_config, write_json, write_progress
from scripts.mil.bag_trainer import _train_bag_method
from scripts.mil.metadata import BAG_METHODS, method_metadata
from scripts.training.eval import _train_sklearn
from scripts.training.trainer import _load_split, _train_mlp

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for method training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _train_selected_method(
    method: str,
    frame,
    class_names: list[str],
    config: dict,
    seed: int,
    result_dir,
) -> dict[str, dict[str, object]]:
    if method in BAG_METHODS:
        return _train_bag_method(method, frame, class_names, config, seed, result_dir)
    if method in {"knn", "ncc"}:
        return _train_sklearn(method, frame, class_names, config)
    return _train_mlp(method, frame, class_names, config, seed, result_dir)


def main() -> None:
    """Train one mitigation method and write result artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = _load_split(paths, args.seed, args.smoke, config)
    class_names = sorted(frame["cancer_type"].unique().tolist())
    result_dir = paths["results"] / args.method / f"seed={args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_status(result_dir, args.method, args.seed, "started")
    results = _train_selected_method(
        args.method, frame, class_names, config, args.seed, result_dir
    )
    _write_outputs(result_dir, args.method, args.seed, args.smoke, results)
    _write_status(result_dir, args.method, args.seed, "completed")
    logger.info(f"Wrote results to {result_dir}")


def _write_outputs(
    result_dir: Path,
    method: str,
    seed: int,
    smoke: bool,
    results: dict[str, dict[str, object]],
) -> None:
    """Write config and split-level result payloads."""
    write_json(
        result_dir / "config.json",
        {
            "method": method,
            "method_metadata": method_metadata(method),
            "seed": seed,
            "smoke": smoke,
        },
    )
    write_json(result_dir / "val_results.json", results["val"])
    write_json(result_dir / "test_results.json", results["test"])


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


if __name__ == "__main__":
    main()
