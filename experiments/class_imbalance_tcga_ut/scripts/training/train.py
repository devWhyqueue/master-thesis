from __future__ import annotations

import logging
import argparse

from scripts.common import ensure_dirs, load_config, write_json
from scripts.training.support import _write_config_json
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
    if method in {"knn", "ncc"}:
        return _train_sklearn(method, frame, class_names, config)
    return _train_mlp(method, frame, class_names, config, seed, result_dir)


def main() -> None:
    """Train one mitigation method and write result artifacts."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = _load_split(paths, args.seed, args.smoke, config)
    class_names = sorted(frame["cancer_type"].unique().tolist())
    result_dir = paths["results"] / args.method / f"seed={args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    results = _train_selected_method(
        args.method, frame, class_names, config, args.seed, result_dir
    )
    _write_config_json(result_dir, args.method, args.seed, args.smoke)
    write_json(result_dir / "validation_results.json", results["val"])
    write_json(result_dir / "test_results.json", results["test"])
    logger.info(f"Wrote results to {result_dir}")


if __name__ == "__main__":
    main()
