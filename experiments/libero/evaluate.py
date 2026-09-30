"""Public single-checkpoint LIBERO evaluation entry point."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "vla_adapter"))

from vla_adapter.rollout import main


if __name__ == "__main__":
    main()
