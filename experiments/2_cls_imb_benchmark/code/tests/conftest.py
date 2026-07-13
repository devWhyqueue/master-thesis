import sys
from pathlib import Path

# Add experiments/2_cls_imb_benchmark/code to sys.path so imbalance_benchmark is importable
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
