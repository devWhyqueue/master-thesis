import sys
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS_ROOT / "common_code"))
sys.path.insert(0, str(EXPERIMENTS_ROOT / "class_imbalance"))
