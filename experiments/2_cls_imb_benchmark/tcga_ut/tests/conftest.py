import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = next(p for p in CODE_ROOT.parents if p.name == "experiments")
sys.path.insert(0, str(EXPERIMENTS_ROOT / "shared"))
sys.path.insert(0, str(CODE_ROOT))

# Remove the stdlib 'code' module from sys.modules if preloaded,
# forcing Python to resolve 'code' to the local package on sys.path.
sys.modules.pop("code", None)
