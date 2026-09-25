"""Run exp-39 analysis with a fresh feature bank for each encoder."""

from __future__ import annotations

import argparse
import importlib
import logging

import _bootstrap  # noqa: F401
from imbalance_benchmark.common import load_config
from imbalance_benchmark.datasets.features.cache import reset_feature_bank

logger = logging.getLogger(__name__)


def main() -> None:
    """Audit the fits and run analysis with per-encoder cache resets."""
    analyze = importlib.import_module("transfer.analyze")
    sensitivity = importlib.import_module("transfer.analyze.sensitivity")
    fit = importlib.import_module("transfer.fit")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    fit.audit_fits(config)
    original = getattr(sensitivity, "_encoder_sensitivity")

    def _fresh_bank(*encoder_args):
        reset_feature_bank()
        return original(*encoder_args)

    setattr(sensitivity, "_encoder_sensitivity", _fresh_bank)
    logger.info(analyze.run_analyze(config))


if __name__ == "__main__":
    main()
