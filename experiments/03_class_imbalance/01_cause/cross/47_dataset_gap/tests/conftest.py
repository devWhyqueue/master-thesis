"""Put the exp-47 code directory and its earlier-experiment dependencies on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import _bootstrap  # noqa: E402,F401
