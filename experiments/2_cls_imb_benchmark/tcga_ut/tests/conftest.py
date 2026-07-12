import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "shared"))
sys.path.insert(0, str(EXPERIMENTS_ROOT / "2_cls_imb_benchmark/tcga_ut"))

# Remove the stdlib 'code' module from sys.modules if preloaded,
# forcing Python to resolve 'code' to the local package on sys.path.
sys.modules.pop("code", None)
