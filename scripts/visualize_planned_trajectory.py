from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autolabsim_matplotlib")

from autolabsim.scene import body_pos, joint_qpos_ids, site_pose
from autolabsim.tasks import TaskRequest, create_task
from autolabsim.tasks.common import cap_body_from_tube_joint, random_reset_info
from autolabsim.topp import Topp, ToppConfig


def add_sphere(mujoco: Any, viewer: Any, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        pos,
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    viewer.user_scn.ngeom += 1


def add_capsule(mujoco: Any, viewer: Any, start: np.ndarray, end: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
    viewer.user_scn.ngeom += 1


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


def build_task(args: argparse.Namespace):
    return create_task(
        TaskRequest(
            task="tube_then_cap_grasp",
            seed=args.reset_seed,
            episode_index=0,
            out_dir=Path("/tmp/autolabsim_visualize_planned_trajectory"),
            model=str(resolve_project_path(args.model)),
            reset_config=str(resolve_project_path(args.reset_config)) if args.reset_config else None,
            cameras=("overview_camera",),
            with_images=False,
            control_dt=args.control_dt,
            frame_skip=args.frame_skip,
            gl_backend="glfw" if not args.no_viewer else None,
            params={},
        )
    )


def plan_phase(task: Any, phase: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task.reset()
    reset_info = dict(task.env.last_reset_info)
    random_info = random_reset_info(reset_info)
    active_joint = str(random_info["active_joint"]) if random_info else "centrifuge_50ml_screw_joint_1"
    active_cap_body = cap_body_from_tube_joint(active_joint)
    cap_pos = body_pos(task.env.model, task.env.data, task.env.mujoco, active_cap_body)

    if phase == "cap_grasp":
        sparse_waypoints = task._make_cap_waypoints(cap_pos)
        waypoints = task._densify_waypoints_from_current_site(task.cap_arm, sparse_waypoints)
        plan = task._plan_arm(task.cap_arm, waypoints, task.runtime.open_gripper)
        arm = task.cap_arm
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    metadata = {
        "phase": phase,
        "active_joint": active_joint,
        "active_cap_body": active_cap_body,
        "slot_index": random_info.get("slot_index") if random_info else None,
        "slot_name": random_info.get("slot_name") if random_info else None,
        "site": str(arm["gripper_site"]),
        "joint_names": list(arm["joint_names"]),
        "runtime": {
            "cap_pregrasp_distance": task.runtime.cap_pregrasp_distance,
            "cap_tool_roll": task.runtime.cap_tool_roll,
            "ik_damping": task.runtime.ik_damping,
            "ik_pos_tol": task.runtime.ik_pos_tol,
            "ik_rot_tol": task.runtime.ik_rot_tol,
            "cartesian_step_size": task.runtime.cartesian_step_size,
            "cartesian_min_steps": task.runtime.cartesian_min_steps,
            "use_topp": task.runtime.use_topp,
            "topp_vel": task.runtime.topp_vel,
            "topp_acc": task.runtime.topp_acc,
        },
        "waypoints": [
            {
                "name": item["name"],
                "target_pos": item["target_pos"],
                "ik_success": item["ik_success"],
                "ik_pos_error": item["ik_pos_error"],
                "ik_rot_error": item["ik_rot_error"],
                "arm_qpos": item["arm_qpos"],
            }
            for item in plan
        ],
    }
    return plan, waypoints, metadata


def sample_joint_path(task: Any, plan: list[dict[str, Any]], samples: int) -> tuple[np.ndarray, np.ndarray]:
    joint_names = tuple(plan[0]["arm_joint_names"])
    qpos_ids = joint_qpos_ids(task.env.model, task.env.mujoco, joint_names)
    start_q = np.asarray([task.env.data.qpos[qpos_id] for qpos_id in qpos_ids], dtype=np.float64)
    q_waypoints = np.vstack([start_q, *(np.asarray(item["arm_qpos"], dtype=np.float64) for item in plan)])

    if task.runtime.use_topp:
        planner = Topp(ToppConfig(dof=len(joint_names), qc_vel=task.runtime.topp_vel, qc_acc=task.runtime.topp_acc))
        trajectory = planner.jnt_traj(q_waypoints)
        times = np.linspace(0.0, float(trajectory.duration), max(2, samples))
        q_samples = np.asarray([planner.query(trajectory, t) for t in times], dtype=np.float64)
    else:
        pieces = []
        per_segment = max(2, samples // max(1, len(q_waypoints) - 1))
        for start, end in zip(q_waypoints[:-1], q_waypoints[1:], strict=True):
            alpha = np.linspace(0.0, 1.0, per_segment, endpoint=False)
            pieces.extend((1.0 - value) * start + value * end for value in alpha)
        pieces.append(q_waypoints[-1])
        q_samples = np.asarray(pieces, dtype=np.float64)

    site_name = str(plan[0]["arm_joint_names"] and task.cap_arm["gripper_site"])
    original_qpos = task.env.data.qpos.copy()
    original_qvel = task.env.data.qvel.copy()
    positions = []
    for q in q_samples:
        task.env.data.qpos[qpos_ids] = q
        task.env.data.qvel[:] = 0.0
        task.env.mujoco.mj_forward(task.env.model, task.env.data)
        pos, _ = site_pose(task.env.model, task.env.data, task.env.mujoco, site_name)
        positions.append(pos)
    task.env.data.qpos[:] = original_qpos
    task.env.data.qvel[:] = original_qvel
    task.env.mujoco.mj_forward(task.env.model, task.env.data)
    return q_samples, np.asarray(positions, dtype=np.float64)


def draw_path(mujoco: Any, viewer: Any, positions: np.ndarray, waypoint_positions: list[np.ndarray]) -> None:
    path_rgba = np.asarray([1.0, 0.45, 0.05, 0.85], dtype=np.float32)
    line_rgba = np.asarray([1.0, 0.20, 0.05, 0.65], dtype=np.float32)
    for start, end in zip(positions[:-1], positions[1:], strict=True):
        add_capsule(mujoco, viewer, start, end, 0.004, line_rgba)
    stride = max(1, len(positions) // 40)
    for pos in positions[::stride]:
        add_sphere(mujoco, viewer, pos, 0.008, path_rgba)

    colors = [
        np.asarray([1.0, 0.85, 0.05, 0.95], dtype=np.float32),
        np.asarray([0.05, 1.0, 0.20, 0.95], dtype=np.float32),
        np.asarray([0.05, 0.35, 1.0, 0.95], dtype=np.float32),
    ]
    for index, pos in enumerate(waypoint_positions):
        add_sphere(mujoco, viewer, pos, 0.018, colors[index % len(colors)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the planned gripper-site trajectory for a scripted task phase.")
    parser.add_argument("--model", default="model/scenes/scene_mujoco_fast_tubes.xml", help="MuJoCo scene XML.")
    parser.add_argument("--reset-config", default="configs/reset_single_tube_random.json", help="Reset config JSON.")
    parser.add_argument("--reset-seed", type=int, default=40, help="Reset seed used to choose the active tube slot.")
    parser.add_argument("--phase", choices=("cap_grasp",), default="cap_grasp", help="Task phase to visualize.")
    parser.add_argument("--samples", type=int, default=120, help="Number of trajectory samples.")
    parser.add_argument("--control-dt", type=float, default=0.05, help="Control timestep used by the task env.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Optional frame skip override.")
    parser.add_argument("--no-viewer", action="store_true", help="Only print trajectory diagnostics.")
    parser.add_argument("--static", action="store_true", help="Do not animate the arm along the sampled trajectory.")
    args = parser.parse_args()

    task = build_task(args)
    try:
        plan, waypoints, metadata = plan_phase(task, args.phase)
        q_samples, positions = sample_joint_path(task, plan, args.samples)
        metadata["sample_count"] = int(len(positions))
        metadata["path_start"] = positions[0].tolist()
        metadata["path_end"] = positions[-1].tolist()
        metadata["path_bbox_min"] = positions.min(axis=0).tolist()
        metadata["path_bbox_max"] = positions.max(axis=0).tolist()
        print(json.dumps(metadata, indent=2))

        if args.no_viewer:
            return

        import mujoco.viewer

        model = task.env.model
        data = task.env.data
        mujoco = task.env.mujoco
        joint_names = tuple(plan[0]["arm_joint_names"])
        qpos_ids = joint_qpos_ids(model, mujoco, joint_names)
        waypoint_positions = [np.asarray(item["pos"], dtype=np.float64) for item in waypoints]

        print("viewer: orange path=sampled gripper-site trajectory, yellow/green=target waypoints")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame = 0
            while viewer.is_running():
                step_start = time.time()
                if not args.static:
                    q = q_samples[frame % len(q_samples)]
                    data.qpos[qpos_ids] = q
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    frame += 1
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_path(mujoco, viewer, positions, waypoint_positions)
                viewer.sync()
                elapsed = time.time() - step_start
                target = model.opt.timestep * 10
                if elapsed < target:
                    time.sleep(target - elapsed)
    finally:
        task.finish()


if __name__ == "__main__":
    main()
