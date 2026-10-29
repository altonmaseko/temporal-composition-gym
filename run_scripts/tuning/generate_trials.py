"""Generate/check the frozen trial list without running training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.common.tuning_plan import main

if __name__ == '__main__':
    raise SystemExit(main())
