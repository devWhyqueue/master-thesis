import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = CODE_ROOT.parent.parent
sys.path.insert(0, str(EXPERIMENTS_ROOT / "common_code"))
sys.path.insert(0, str(CODE_ROOT))
