"""Put exp-48's code (and, through its bootstrap, exp-40's) on the import path.

The bootstrap is executed by path: in a repository-wide session another experiment's
``_bootstrap`` module may already be cached under the same name.
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "code" / "_bootstrap.py"))
