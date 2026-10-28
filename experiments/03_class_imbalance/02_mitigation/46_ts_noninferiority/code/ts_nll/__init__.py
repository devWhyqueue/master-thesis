"""Temperature-scaled NLL noninferiority test of the mitigation methods against CE (exp-46)."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from pathlib import Path
from typing import Any

from imbalance_benchmark.common import REPO_ROOT, load_config

__all__ = ["load_protocol", "load_source", "source_report"]


def load_protocol(path: str | Path) -> dict[str, Any]:
    """The exp-46 config (``sources`` and ``protocol``)."""
    return load_config(path)


def load_source(source: str | Path) -> dict[str, Any]:
    """One exp-41..44 config, given relative to the repository root."""
    return load_config(REPO_ROOT / source)


def source_report(source: str | Path) -> Path:
    """The ``report/analysis.json`` of the experiment that owns a source config."""
    return REPO_ROOT / Path(source).parents[1] / "report" / "analysis.json"
