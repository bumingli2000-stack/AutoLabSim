from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim.tasks.cli import main


if __name__ == "__main__":
    defaults = [
        "--scene",
        "fast_tubes_pipette",
        "--task",
        "pipette_grasp",
        "--out-root",
        "data/episodes/pipette_grasp_batch",
    ]
    raise SystemExit(main([*defaults, *sys.argv[1:]]))
