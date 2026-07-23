from __future__ import annotations

import argparse
import functools
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import yaml

import imbalance_benchmark.analysis.inference.recovery as _recovery

from imbalance_benchmark.analysis import query as _query
from imbalance_benchmark.analysis.inference import crossed_permutation as _crossed
from imbalance_benchmark.commands import confirm as _confirm_pkg
from imbalance_benchmark.commands import tuning as _tuning_module
from imbalance_benchmark.commands.analyze import cmd_analyze
from imbalance_benchmark.commands.confirm import cmd_confirm
from imbalance_benchmark.commands.freeze import cmd_freeze
from imbalance_benchmark.commands.pilot import cmd_pilot
from imbalance_benchmark.commands.prepare import cmd_prepare
from imbalance_benchmark.commands.tuning import cmd_tune
from imbalance_benchmark.common import REPO_ROOT
from imbalance_benchmark.hydra import cmd_submit
from imbalance_benchmark.modeling import context as _context
from imbalance_benchmark.modeling import training as _training

__all__ = ["cmd_smoke"]

logger = logging.getLogger(__name__)

_SMOKE_CROSSED_PERMUTATIONS = 2000

Patch = tuple[object, str, Any]


def _fast_tuning_seeds(orig: Any) -> Any:
    """Wrap `_tuning_seeds` to keep only its first (of two) tuning-init seeds."""
    return lambda freeze: orig(freeze)[:1]


def _fast_balanced_baseline(orig: Any) -> Any:
    """Wrap `balanced_baseline` to shrink its permutation-test sample count."""

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        baseline = orig(*args, **kwargs)
        if baseline is not None:
            baseline.n_perm = _SMOKE_CROSSED_PERMUTATIONS
        return baseline

    return _wrapped


def _smoke_patches() -> list[Patch]:
    """(module, attribute, new_value) triples the smoke run temporarily applies.

    Grid/seed/step-budget knobs keep tune+confirm from running the full
    protocol search on toy data; the permutation-count patches shrink the
    cross-split gate tests.
    """
    return [
        (_context, "LEARNING_RATE_GRID", _context.LEARNING_RATE_GRID[:1]),
        (_context, "GRIDS", {m: v[:1] for m, v in _context.GRIDS.items()}),
        (_context, "REFERENCE_PASSES", 2),
        (_training, "REFERENCE_PASSES", 2),
        (
            _confirm_pkg,
            "CONFIRMATION_SEED_ROLES",
            _confirm_pkg.CONFIRMATION_SEED_ROLES[:1],
        ),
        (_query, "EXPECTED_CONFIRMATION_SEEDS", 1),
        (
            _tuning_module,
            "_tuning_seeds",
            _fast_tuning_seeds(_tuning_module._tuning_seeds),
        ),
        (
            _crossed,
            "crossed_block_permutation_ba",
            functools.partial(
                _crossed.crossed_block_permutation_ba,
                n_permutations=_SMOKE_CROSSED_PERMUTATIONS,
            ),
        ),
        (
            _crossed,
            "crossed_block_permutation_tail_nll",
            functools.partial(
                _crossed.crossed_block_permutation_tail_nll,
                n_permutations=_SMOKE_CROSSED_PERMUTATIONS,
            ),
        ),
        (
            _recovery,
            "balanced_baseline",
            _fast_balanced_baseline(_recovery.balanced_baseline),
        ),
    ]


@contextmanager
def _scaled_down_search() -> Iterator[None]:
    """Apply `_smoke_patches` for the duration of the block, then restore every value.

    Every stage below runs the real algorithm on the synthetic dataset with
    the protocol's full config, which takes 20+ minutes on fake data. These
    are pure scale knobs (not algorithmic changes); real (non-smoke) runs are
    never affected since everything is restored on exit.
    Consistency note: `CONFIRMATION_SEED_ROLES` and `EXPECTED_CONFIRMATION_SEEDS`
    must match, or analysis will reject the confirmation block as incomplete.
    """
    patches = _smoke_patches()
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patches]
    for module, attr, value in patches:
        setattr(module, attr, value)
    try:
        yield
    finally:
        for module, attr, value in originals:
            setattr(module, attr, value)


def _mock_config() -> dict[str, object]:
    """Synthetic-dataset config for the local end-to-end smoke path."""
    return {
        "paths": {"outputs": "experiments/2_cls_imb_benchmark/smoke_outputs"},
        "slurm": {"partition": "cpu-test", "container": "./environment.sif"},
        "dataset": {
            "name": "synthetic",
            "regime": "patch",
            "target": "cancer_type",
            "version": "smoke-test-v1",
            "eligibility_rules": {"fixture": True},
        },
        "analysis": {"bootstrap_replicates": 5},
    }


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run local end-to-end smoke test."""
    logger.info("=== Running End-to-End Smoke Test ===")
    config_path = (
        REPO_ROOT / "experiments/2_cls_imb_benchmark/smoke_outputs/configs/default.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_mock_config(), f)
    ns = argparse.Namespace(
        config=str(config_path), seed=0, dry_run=True, split_index=None
    )
    with _scaled_down_search():
        cmd_prepare(ns)
        cmd_pilot(ns)
        cmd_freeze(ns)
        cmd_tune(ns)
        cmd_confirm(ns)
        cmd_analyze(ns)
    cmd_submit(ns)
    logger.info("=== Smoke Test Finished Successfully! ===")
