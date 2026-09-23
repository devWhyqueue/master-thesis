"""Pytest fixtures and environment configuration for the classprops tests.

Every experiment in this repo names its sys.path-setup module ``code/_bootstrap.py``, a plain
top-level module rather than a package member. When pytest collects more than one experiment's
tests in one process (e.g. clean-code's repo-wide gate), the first one imported wins the
``sys.modules["_bootstrap"]`` cache slot and every later experiment's own ``import _bootstrap``
silently reuses it -- with whichever *other* experiment's code-dir list that first import brought,
not this experiment's own (exp-29 *and* exp-30's dirs, both needed here). Loading this
experiment's ``_bootstrap.py`` under its own unique module name sidesteps that collision instead
of relying on import order.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

_SPEC = importlib.util.spec_from_file_location(
    "classprops_bootstrap", _CODE / "_bootstrap.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_bootstrap_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_bootstrap_module)
