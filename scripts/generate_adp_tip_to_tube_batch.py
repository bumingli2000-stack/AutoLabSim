#!/usr/bin/env python3
"""Batch generation entry point for the ADP tip-to-tube task.

This script wraps the common batch CLI and supplies default arguments for
the ADP task (model, reset config, cameras, etc.). Command-line overrides
are allowed via additional sys.argv arguments.
"""

from __future__ import annotations

from pathlib import Path
from random import SystemRandom
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim_Task.tasks.cli import main


def _has_seed_override(args: list[str]) -> bool:
    return any(
        arg in {"--seeds", "--seed-start"}
        or arg.startswith("--seeds=")
        or arg.startswith("--seed-start=")
        for arg in args
    )


if __name__ == "__main__":
    # Default arguments for the ADP tip-to-tube batch generation.
    # Episode image recording is intentionally off by default; use
    # export_episode_images.py after generation, or pass --with-images here.
    defaults = [
        "--model",
        "model/scenes/scene_mujoco_fast_tubes_adp_pipette_grasped.xml",
        "--reset-config",
        "configs/reset_adp_pipette_grasped.json",
        "--task",
        "adp_tip_to_tube",
        "--out-root",
        "data/episodes/adp_pipette_tip_to_tube_batch",
        "--cameras",
        "overview_camera,wrist_cam,wrist_cam1",
        "--initial-static-steps",
        "20",
        "--settle-steps",
        "8",
        "--tool-stabilize-steps",
        "12",
        "--steps-per-segment",
        "12",
        "--tip-hover-steps",
        "36",
        "--close-steps",
        "2",
        "--tip-mount-settle-steps",
        "8",
        "--hold-steps",
        "60",
        "--release-wait-steps",
        "8",
        "--waypoint-settle-steps",
        "1",
        "--visual-servo",
        "--visual-servo-max-iters",
        "14",
        "--visual-servo-steps",
        "6",
        "--visual-servo-pos-tol",
        "0.00015",
        "--visual-servo-rot-tol",
        "0.03",
        "--visual-servo-gain",
        "0.85",
        "--visual-servo-integral-gain",
        "0.2",
        "--visual-servo-max-correction",
        "0.012",
        "--tube-top-offset",
        "0.115",
        "--tube-near-height",
        "0.010",
        "--tube-target-offset",
        "0 0 0",
        "--tip-hover-height",
        "0.020",
        "--tip-retract-height",
        "0.100",
        "--tip-mount-offset",
        "0 0 -0.012",
    ]
    user_args = sys.argv[1:]
    if not _has_seed_override(user_args):
        defaults.extend(
            [
                "--seed-start",
                str(SystemRandom().randrange(0, 1_000_000)),
            ]
        )

    # Combine defaults with any command-line arguments provided by the user
    sys.exit(main([*defaults, *user_args]))
