from __future__ import annotations

import argparse
import logging

import yaml

from imbalance_benchmark.commands.analyze import cmd_analyze
from imbalance_benchmark.commands.confirm import cmd_confirm
from imbalance_benchmark.commands.freeze import cmd_freeze
from imbalance_benchmark.commands.pilot import cmd_pilot
from imbalance_benchmark.commands.prepare import cmd_prepare
from imbalance_benchmark.commands.tuning import cmd_tune
from imbalance_benchmark.common import REPO_ROOT
from imbalance_benchmark.hydra import cmd_submit

__all__ = ["cmd_smoke"]

logger = logging.getLogger(__name__)


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run local end-to-end smoke test."""
    logger.info("=== Running End-to-End Smoke Test ===")
    mock_config = {
        "paths": {"outputs": "experiments/2_cls_imb_benchmark/smoke_outputs"},
        "slurm": {"partition": "cpu-test", "container": "./environment.sif"},
    }
    config_path = (
        REPO_ROOT / "experiments/2_cls_imb_benchmark/smoke_outputs/configs/default.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(mock_config, f)
    ns = argparse.Namespace(
        config=str(config_path), seed=0, dry_run=True, split_index=None
    )
    cmd_prepare(ns)
    cmd_pilot(ns)
    cmd_freeze(ns)
    cmd_tune(ns)
    cmd_confirm(ns)
    cmd_analyze(ns)
    cmd_submit(ns)
    logger.info("=== Smoke Test Finished Successfully! ===")
