"""WSI bag trainer shim injecting experiment output paths."""

from __future__ import annotations

from pathlib import Path

from common_code.wsi import trainer as _trainer
from scripts.common import output_root

_build_model = _trainer._build_model
_build_mde_model = _trainer._build_mde_model
_loader = _trainer._loader
_run_training = _trainer._run_training
_run_mde_training = _trainer._run_mde_training


def _train_bag_method(
    method: str,
    frame,
    class_names: list[str],
    config: dict,
    seed: int,
    result_dir: Path,
    smoke: bool = False,
):
    enriched = dict(config)
    enriched.setdefault("output_root", str(output_root(config)))
    return _trainer._train_bag_method(
        method, frame, class_names, enriched, seed, result_dir, smoke
    )


__all__ = [
    "_build_mde_model",
    "_build_model",
    "_loader",
    "_run_mde_training",
    "_run_training",
    "_train_bag_method",
]
