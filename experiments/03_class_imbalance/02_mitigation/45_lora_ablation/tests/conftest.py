"""Pytest fixtures and environment configuration for the LoRA-omission tests."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
for _path in (_CODE, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _bootstrap  # noqa: E402,F401
