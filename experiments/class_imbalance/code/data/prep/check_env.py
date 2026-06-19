from __future__ import annotations

import logging
import importlib

logger = logging.getLogger(__name__)


REQUIRED_MODULES = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "torch": "torch",
}


def main() -> None:
    """Verify that all required Python dependencies are installed."""
    missing = []
    for module_name, package_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            missing.append(f"{package_name} ({type(exc).__name__}: {exc})")
            continue
        logger.info(f"{package_name}: {getattr(module, '__version__', 'installed')}")
    if missing:
        joined = "\n  - ".join(missing)
        raise SystemExit(f"Missing experiment dependencies:\n  - {joined}")


if __name__ == "__main__":
    main()
