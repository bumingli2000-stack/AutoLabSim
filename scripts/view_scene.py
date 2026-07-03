from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim.reset_config import apply_reset_config, load_reset_config


def parse_group_list(raw_value: str) -> list[int]:
    groups: list[int] = []
    for part in raw_value.replace(",", " ").split():
        group = int(part)
        if group < 0 or group > 5:
            raise argparse.ArgumentTypeError("geom groups must be between 0 and 5")
        groups.append(group)
    return groups


def resolve_scene_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = ROOT / path
    if project_candidate.exists():
        return project_candidate

    scene_candidate = ROOT / "model" / "scenes" / path
    if scene_candidate.exists():
        return scene_candidate

    return cwd_candidate


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = ROOT / path
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a MuJoCo scene XML file in the interactive viewer.")
    parser.add_argument("xml", help="Path to the scene XML file, e.g. model/scenes/scene_mujoco_fast_tubes.xml")
    parser.add_argument("--reset-config", default=None, help="Optional reset config JSON to apply before opening the viewer.")
    parser.add_argument("--reset-seed", type=int, default=None, help="Optional random seed used by --reset-config.")
    parser.add_argument(
        "--geom-groups",
        type=parse_group_list,
        default=None,
        help="Optional visible geom groups, e.g. '0 5' to show visuals plus debug collision geoms.",
    )
    parser.add_argument(
        "--show-cap-collision",
        action="store_true",
        help="Convenience view for the cap: show visual geoms plus group-5 cap collision/debug geoms.",
    )
    parser.add_argument(
        "--site-groups",
        type=parse_group_list,
        default=None,
        help="Optional visible site groups, e.g. '5' to show pinch/debug sites.",
    )
    parser.add_argument(
        "--show-gripper-collision",
        action="store_true",
        help="Convenience view for gripper debugging: show visuals, collision geoms, and pinch/debug sites.",
    )
    parser.add_argument("--show-contacts", action="store_true", help="Show MuJoCo contact points in the viewer.")
    args = parser.parse_args()

    scene_path = resolve_scene_path(args.xml)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {scene_path}")
    if scene_path.suffix.lower() != ".xml":
        raise ValueError(f"Expected an XML scene file, got: {scene_path}")

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    reset_info = {}
    if args.reset_config:
        reset_config_path = resolve_project_path(args.reset_config)
        if not reset_config_path.exists():
            raise FileNotFoundError(f"Reset config not found: {reset_config_path}")
        reset_info = apply_reset_config(
            model,
            data,
            mujoco,
            load_reset_config(reset_config_path),
            np.random.default_rng(args.reset_seed),
        )

    mujoco.mj_forward(model, data)

    print(f"scene: {scene_path}")
    if args.reset_config:
        print(f"reset_config: {resolve_project_path(args.reset_config)}")
        if reset_info:
            print(f"reset_info: {reset_info}")
    print("viewer: close the MuJoCo window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        visible_groups = args.geom_groups
        if args.show_cap_collision and visible_groups is None:
            visible_groups = [0, 5]
        if args.show_gripper_collision:
            visible_groups = sorted(set((visible_groups or [0]) + [2, 3, 5]))
        visible_site_groups = args.site_groups
        if args.show_gripper_collision and visible_site_groups is None:
            visible_site_groups = [5]
        if visible_groups is not None:
            with viewer.lock():
                viewer.opt.geomgroup[:] = 0
                for group in visible_groups:
                    viewer.opt.geomgroup[group] = 1
        if visible_site_groups is not None:
            with viewer.lock():
                viewer.opt.sitegroup[:] = 0
                for group in visible_site_groups:
                    viewer.opt.sitegroup[group] = 1
        if args.show_contacts:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
