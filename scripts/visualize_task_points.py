from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim.math3d import gripper_quat_from_axes, mat_to_quat, quat_to_mat, unit
from autolabsim.grasp_pose import LocalGraspPose, local_grasp_to_world_target
from autolabsim.scene import (
    actuator_id,
    body_pos,
    capture_free_joint_state,
    free_joint_pos,
    free_joint_addresses,
    restore_free_joint_state,
)
from autolabsim.reset_config import apply_reset_config, load_reset_config
from autolabsim.tasks.common import ARM_DEFAULTS, cap_body_from_tube_joint, random_reset_info
from autolabsim.tasks.pipette_grasp import PipetteGraspTaskConfig

TASK_DEFAULTS = {
    "tube_grasp": {
        "arm": "second",
        "approach_axis": None,
        "closing_axis": None,
        "tool_roll": float(np.pi),
        "grasp_outward_offset": 0.02,
    },
    "cap_grasp": {
        "arm": "first",
        "approach_axis": "0 0 -1",
        "closing_axis": "1 0 0",
        "tool_roll": 0.0,
        "grasp_outward_offset": 0.0,
    },
    "tube_then_cap_grasp": {
        "arm": "second",
        "approach_axis": None,
        "closing_axis": None,
        "tool_roll": float(np.pi),
        "grasp_outward_offset": 0.02,
    },
    "bimanual_unscrew_cap": {
        "arm": "second",
        "approach_axis": None,
        "closing_axis": None,
        "tool_roll": float(np.pi),
        "grasp_outward_offset": 0.0,
    },
    "pipette_grasp": {
        "arm": "first",
        "approach_axis": None,
        "closing_axis": None,
        "tool_roll": 0.0,
        "grasp_outward_offset": 0.0,
    },
}


Planner = Callable[[Any, Any, Any, argparse.Namespace, str, dict[str, Any] | None], dict[str, Any]]


def parse_vec3(value: str, name: str) -> np.ndarray:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must have exactly 3 numbers")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def parse_optional_vec3(value: str | None, name: str) -> np.ndarray | None:
    if value is None or value.strip().lower() in ("", "none"):
        return None
    return parse_vec3(value, name)


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


def add_connector(
    mujoco: Any,
    viewer: Any,
    geom_type: Any,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(geom, geom_type, radius, start, end)
    viewer.user_scn.ngeom += 1


def add_frame_axes(
    mujoco: Any,
    viewer: Any,
    pos: np.ndarray,
    mat: np.ndarray,
    scale: float,
    radius: float,
    alpha: float = 0.95,
) -> None:
    colors = (
        np.asarray([1.0, 0.05, 0.05, alpha], dtype=np.float32),
        np.asarray([0.05, 0.9, 0.15, alpha], dtype=np.float32),
        np.asarray([0.1, 0.25, 1.0, alpha], dtype=np.float32),
    )
    for axis_id, color in enumerate(colors):
        start = np.asarray(pos, dtype=np.float64)
        end = start + float(scale) * np.asarray(mat, dtype=np.float64).reshape(3, 3)[:, axis_id]
        add_connector(mujoco, viewer, mujoco.mjtGeom.mjGEOM_ARROW, start, end, radius, color)


def frame_pose(mujoco: Any, name: str, pos: np.ndarray, mat: np.ndarray) -> dict[str, Any]:
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    return {
        "name": name,
        "pos": np.asarray(pos, dtype=np.float64).tolist(),
        "quat_wxyz": mat_to_quat(mujoco, mat).tolist(),
        "x_axis": mat[:, 0].tolist(),
        "y_axis": mat[:, 1].tolist(),
        "z_axis": mat[:, 2].tolist(),
    }


def resolve_arm(args: argparse.Namespace) -> str:
    return args.arm or str(TASK_DEFAULTS[args.task]["arm"])


def resolve_task_arm(args: argparse.Namespace, task: str, arm_override: str | None = None) -> str:
    return arm_override or str(TASK_DEFAULTS[task]["arm"])


def resolve_axes_for_task(
    args: argparse.Namespace,
    task: str,
    arm: str,
    approach_override: str | None = None,
    closing_override: str | None = None,
    tool_roll_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    def coerce_axis(value: str | np.ndarray, axis_name: str) -> np.ndarray:
        if isinstance(value, str):
            return unit(parse_vec3(value, axis_name), axis_name)
        return unit(np.asarray(value, dtype=np.float64), axis_name)

    task_defaults = TASK_DEFAULTS[task]
    default_approach = task_defaults["approach_axis"] or ARM_DEFAULTS[arm]["approach_axis"]
    default_closing = task_defaults["closing_axis"] or ARM_DEFAULTS[arm]["closing_axis"]
    if approach_override is not None:
        approach_axis = unit(parse_vec3(approach_override, "approach_axis"), "approach_axis")
    else:
        approach_axis = coerce_axis(default_approach, "approach_axis")
    if closing_override is not None:
        closing_axis = unit(parse_vec3(closing_override, "closing_axis"), "closing_axis")
    else:
        closing_axis = coerce_axis(default_closing, "closing_axis")
    tool_roll = float(tool_roll_override if tool_roll_override is not None else task_defaults["tool_roll"])
    return approach_axis, closing_axis, tool_roll


def resolve_axes(args: argparse.Namespace, arm: str) -> tuple[np.ndarray, np.ndarray, float]:
    return resolve_axes_for_task(args, args.task, arm, args.approach_axis, args.closing_axis, args.tool_roll)


def resolve_grasp_outward_offset(args: argparse.Namespace, task: str | None = None) -> float:
    if args.grasp_outward_offset is not None:
        return float(args.grasp_outward_offset)
    return float(TASK_DEFAULTS[task or args.task]["grasp_outward_offset"])


def marker_plan(
    mujoco: Any,
    approach_axis: np.ndarray,
    closing_axis: np.ndarray,
    tool_roll: float,
    pregrasp_pos: np.ndarray,
    grasp_pos: np.ndarray,
    post_pos: np.ndarray,
) -> dict[str, Any]:
    quat = gripper_quat_from_axes(mujoco, approach_axis, closing_axis, tool_roll)
    grasp_mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(grasp_mat, quat)
    local_y = grasp_mat.reshape(3, 3)[:, 1]
    return {
        "poses": {
            "pregrasp": {"pos": pregrasp_pos.tolist(), "quat_wxyz": quat.tolist()},
            "grasp": {"pos": grasp_pos.tolist(), "quat_wxyz": quat.tolist()},
            "post": {"pos": post_pos.tolist(), "quat_wxyz": quat.tolist()},
        },
        "markers": [
            {"name": "pregrasp", "pos": pregrasp_pos, "radius": 0.016, "rgba": np.asarray([1.0, 0.75, 0.05, 0.9], dtype=np.float32)},
            {"name": "grasp", "pos": grasp_pos, "radius": 0.016, "rgba": np.asarray([0.1, 1.0, 0.25, 0.9], dtype=np.float32)},
            {"name": "post", "pos": post_pos, "radius": 0.016, "rgba": np.asarray([0.1, 0.35, 1.0, 0.9], dtype=np.float32)},
        ],
        "connectors": [
            {
                "name": "approach",
                "type": "arrow",
                "start": pregrasp_pos,
                "end": grasp_pos,
                "radius": 0.007,
                "rgba": np.asarray([1.0, 0.15, 0.1, 0.9], dtype=np.float32),
            },
            {
                "name": "gripper_local_y",
                "type": "capsule",
                "start": grasp_pos,
                "end": grasp_pos + 0.065 * local_y,
                "radius": 0.005,
                "rgba": np.asarray([0.9, 0.1, 1.0, 0.9], dtype=np.float32),
            },
        ],
    }


def prefix_plan(plan: dict[str, Any], prefix: str) -> dict[str, Any]:
    prefixed = {
        "poses": {f"{prefix}_{name}": pose for name, pose in plan["poses"].items()},
        "frames": [],
        "markers": [],
        "connectors": [],
    }
    for frame in plan.get("frames", []):
        prefixed["frames"].append({**frame, "name": f"{prefix}_{frame['name']}"})
    for marker in plan["markers"]:
        prefixed["markers"].append({**marker, "name": f"{prefix}_{marker['name']}"})
    for connector in plan["connectors"]:
        prefixed["connectors"].append({**connector, "name": f"{prefix}_{connector['name']}"})
    return prefixed


def body_frame(model: Any, data: Any, mujoco: Any, body_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Unknown body: {body_name}")
    pos = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    mat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, mat_to_quat(mujoco, mat), mat


def site_frame(model: Any, data: Any, mujoco: Any, site_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return None
    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, mat_to_quat(mujoco, mat), mat


def tip_site_name(joint_name: str, joint_prefix: str, site_prefix: str, suffix: str) -> str:
    if not joint_name.startswith(joint_prefix):
        raise ValueError(f"Tip joint does not match prefix {joint_prefix!r}: {joint_name}")
    return f"{site_prefix}{joint_name[len(joint_prefix):]}{suffix}"


def nearest_active_tip_site_frames(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    pipette_tip_pos: np.ndarray,
    reset_info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    subset_info = (reset_info or {}).get("random_free_joint_subset", {})
    active = subset_info.get("active", []) if isinstance(subset_info, dict) else []
    candidates = [str(item["joint"]) for item in active if isinstance(item, dict) and "joint" in item]
    if not candidates:
        candidates = []
        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name and joint_name.startswith(args.tip_joint_prefix):
                candidates.append(joint_name)

    best: dict[str, Any] | None = None
    for joint_name in candidates:
        mount_site = tip_site_name(joint_name, args.tip_joint_prefix, args.tip_site_prefix, args.tip_mount_site_suffix)
        end_site = tip_site_name(joint_name, args.tip_joint_prefix, args.tip_site_prefix, args.tip_end_site_suffix)
        mount_frame = site_frame(model, data, mujoco, mount_site)
        if mount_frame is None:
            continue
        mount_pos, mount_quat, mount_mat = mount_frame
        end_frame = site_frame(model, data, mujoco, end_site)
        distance = float(np.linalg.norm(mount_pos[:2] - pipette_tip_pos[:2]))
        item = {
            "joint": joint_name,
            "mount_site": mount_site,
            "mount_pos": mount_pos,
            "mount_quat": mount_quat,
            "mount_mat": mount_mat,
            "end_site": end_site if end_frame is not None else None,
            "end_frame": end_frame,
            "xy_distance": distance,
        }
        if best is None or distance < best["xy_distance"]:
            best = item
    return best


def plan_tube_grasp_points(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    active_joint: str,
    random_info: dict[str, Any] | None,
) -> dict[str, Any]:
    arm = resolve_arm(args)
    approach_axis, closing_axis, tool_roll = resolve_axes(args, arm)
    tube_pos = free_joint_pos(model, data, mujoco, active_joint)
    lift_offset = args.lift_offset
    if lift_offset is None:
        lift_offset = np.asarray([0.0, 0.0, args.lift_distance], dtype=np.float64)
    lift_offset = np.asarray(lift_offset, dtype=np.float64)

    grasp_pos = (
        tube_pos
        + np.asarray([0.0, 0.0, args.grasp_height], dtype=np.float64)
        + approach_axis * float(args.pinch_forward_offset)
        - approach_axis * resolve_grasp_outward_offset(args)
    )
    pregrasp_pos = grasp_pos - approach_axis * float(args.pregrasp_distance)
    post_pos = grasp_pos + lift_offset
    plan = marker_plan(mujoco, approach_axis, closing_axis, tool_roll, pregrasp_pos, grasp_pos, post_pos)
    plan["metadata"] = {
        "task": args.task,
        "arm": arm,
        "active_joint": active_joint,
        "slot_index": random_info.get("slot_index") if random_info else None,
        "slot_name": random_info.get("slot_name") if random_info else None,
        "tube_origin_pos": tube_pos.tolist(),
        "grasp_height": args.grasp_height,
        "pinch_forward_offset": args.pinch_forward_offset,
        "grasp_outward_offset": resolve_grasp_outward_offset(args),
        "lift_offset": lift_offset.tolist(),
        "tool_roll": tool_roll,
    }
    return plan


def plan_cap_grasp_points(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    active_joint: str,
    random_info: dict[str, Any] | None,
) -> dict[str, Any]:
    arm = resolve_arm(args)
    approach_axis, closing_axis, tool_roll = resolve_axes(args, arm)
    cap_body = args.cap_body or cap_body_from_tube_joint(active_joint)
    cap_pos = body_pos(model, data, mujoco, cap_body)
    return plan_cap_grasp_at_pos(
        mujoco,
        args,
        arm,
        active_joint,
        random_info,
        cap_body,
        cap_pos,
        approach_axis,
        closing_axis,
        tool_roll,
    )


def plan_cap_grasp_at_pos(
    mujoco: Any,
    args: argparse.Namespace,
    arm: str,
    active_joint: str,
    random_info: dict[str, Any] | None,
    cap_body: str,
    cap_pos: np.ndarray,
    approach_axis: np.ndarray,
    closing_axis: np.ndarray,
    tool_roll: float,
) -> dict[str, Any]:
    cap_offset = np.asarray(args.cap_offset, dtype=np.float64)
    post_offset = args.post_offset
    if post_offset is None:
        post_offset = np.asarray([0.0, 0.0, args.post_distance], dtype=np.float64)
    post_offset = np.asarray(post_offset, dtype=np.float64)

    grasp_pos = cap_pos + cap_offset - approach_axis * resolve_grasp_outward_offset(args)
    pregrasp_pos = grasp_pos - approach_axis * float(args.pregrasp_distance)
    post_pos = grasp_pos + post_offset
    plan = marker_plan(mujoco, approach_axis, closing_axis, tool_roll, pregrasp_pos, grasp_pos, post_pos)
    plan["metadata"] = {
        "task": args.task,
        "arm": arm,
        "active_joint": active_joint,
        "cap_body": cap_body,
        "slot_index": random_info.get("slot_index") if random_info else None,
        "slot_name": random_info.get("slot_name") if random_info else None,
        "cap_body_pos": cap_pos.tolist(),
        "cap_offset": cap_offset.tolist(),
        "grasp_outward_offset": resolve_grasp_outward_offset(args),
        "post_offset": post_offset.tolist(),
        "tool_roll": tool_roll,
    }
    return plan


def namespace_with(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def plan_tube_then_cap_grasp_points(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    active_joint: str,
    random_info: dict[str, Any] | None,
) -> dict[str, Any]:
    tube_arm = args.tube_arm or "second"
    cap_arm = args.cap_arm or "first"

    tube_args = namespace_with(
        args,
        task="tube_grasp",
        arm=tube_arm,
        approach_axis=args.tube_approach_axis,
        closing_axis=args.tube_closing_axis,
        tool_roll=args.tube_tool_roll,
        grasp_outward_offset=args.tube_grasp_outward_offset if args.tube_grasp_outward_offset is not None else args.grasp_outward_offset,
    )
    cap_args = namespace_with(
        args,
        task="cap_grasp",
        arm=cap_arm,
        approach_axis=args.cap_approach_axis,
        closing_axis=args.cap_closing_axis,
        tool_roll=args.cap_tool_roll,
        grasp_outward_offset=args.cap_grasp_outward_offset if args.cap_grasp_outward_offset is not None else args.grasp_outward_offset,
    )

    tube_plan = plan_tube_grasp_points(model, data, mujoco, tube_args, active_joint, random_info)
    cap_body = args.cap_body or cap_body_from_tube_joint(active_joint)
    cap_start_pos = body_pos(model, data, mujoco, cap_body)
    tube_grasp_pos = np.asarray(tube_plan["poses"]["grasp"]["pos"], dtype=np.float64)
    tube_post_pos = np.asarray(tube_plan["poses"]["post"]["pos"], dtype=np.float64)
    tube_motion_delta = tube_post_pos - tube_grasp_pos
    cap_at_tube_post = cap_start_pos + tube_motion_delta

    cap_approach_axis, cap_closing_axis, cap_tool_roll = resolve_axes_for_task(
        cap_args,
        "cap_grasp",
        cap_arm,
        cap_args.approach_axis,
        cap_args.closing_axis,
        cap_args.tool_roll,
    )
    cap_plan = plan_cap_grasp_at_pos(
        mujoco,
        cap_args,
        cap_arm,
        active_joint,
        random_info,
        cap_body,
        cap_at_tube_post,
        cap_approach_axis,
        cap_closing_axis,
        cap_tool_roll,
    )

    tube_prefixed = prefix_plan(tube_plan, "tube")
    cap_prefixed = prefix_plan(cap_plan, "cap")
    return {
        "poses": {**tube_prefixed["poses"], **cap_prefixed["poses"]},
        "frames": [*tube_prefixed.get("frames", []), *cap_prefixed.get("frames", [])],
        "markers": [*tube_prefixed["markers"], *cap_prefixed["markers"]],
        "connectors": [*tube_prefixed["connectors"], *cap_prefixed["connectors"]],
        "metadata": {
            "task": args.task,
            "active_joint": active_joint,
            "slot_index": random_info.get("slot_index") if random_info else None,
            "slot_name": random_info.get("slot_name") if random_info else None,
            "tube_arm": tube_arm,
            "cap_arm": cap_arm,
            "cap_body": cap_body,
            "cap_start_pos": cap_start_pos.tolist(),
            "tube_motion_delta": tube_motion_delta.tolist(),
            "cap_pos_at_tube_post": cap_at_tube_post.tolist(),
            "tube": tube_plan["metadata"],
            "cap": cap_plan["metadata"],
        },
    }


def plan_bimanual_unscrew_cap_points(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    active_joint: str,
    random_info: dict[str, Any] | None,
) -> dict[str, Any]:
    tube_arm = args.tube_arm or "second"
    cap_arm = args.cap_arm or "first"

    cap_args = namespace_with(
        args,
        task="cap_grasp",
        arm=cap_arm,
        approach_axis=args.cap_approach_axis,
        closing_axis=args.cap_closing_axis,
        tool_roll=args.cap_tool_roll,
        pregrasp_distance=args.cap_pregrasp_distance,
        grasp_outward_offset=args.cap_grasp_outward_offset if args.cap_grasp_outward_offset is not None else 0.0,
    )
    tube_args = namespace_with(
        args,
        task="tube_grasp",
        arm=tube_arm,
        approach_axis=args.tube_approach_axis,
        closing_axis=args.tube_closing_axis,
        tool_roll=args.tube_tool_roll,
        grasp_outward_offset=args.tube_grasp_outward_offset if args.tube_grasp_outward_offset is not None else 0.0,
    )

    cap_body = args.cap_body or cap_body_from_tube_joint(active_joint)
    cap_start_pos = body_pos(model, data, mujoco, cap_body)
    cap_approach_axis, cap_closing_axis, cap_tool_roll = resolve_axes_for_task(
        cap_args,
        "cap_grasp",
        cap_arm,
        cap_args.approach_axis,
        cap_args.closing_axis,
        cap_args.tool_roll,
    )
    cap_plan = plan_cap_grasp_at_pos(
        mujoco,
        cap_args,
        cap_arm,
        active_joint,
        random_info,
        cap_body,
        cap_start_pos,
        cap_approach_axis,
        cap_closing_axis,
        cap_tool_roll,
    )

    # screw_cap.py 里是瓶盖臂先抓盖并整体抬起管子，再由管身臂横向夹管。
    tube_start_pos = free_joint_pos(model, data, mujoco, active_joint)
    tube_lifted_pos = tube_start_pos + np.asarray(args.post_offset, dtype=np.float64)
    tube_plan = plan_tube_grasp_points(model, data, mujoco, tube_args, active_joint, random_info)
    lifted_tube_plan = plan_tube_grasp_points_at_pos(
        model,
        data,
        mujoco,
        tube_args,
        active_joint,
        random_info,
        tube_lifted_pos,
    )

    cap_prefixed = prefix_plan(cap_plan, "cap")
    tube_prefixed = prefix_plan(lifted_tube_plan, "tube_after_cap_lift")
    return {
        "poses": {**cap_prefixed["poses"], **tube_prefixed["poses"]},
        "frames": [*cap_prefixed.get("frames", []), *tube_prefixed.get("frames", [])],
        "markers": [*cap_prefixed["markers"], *tube_prefixed["markers"]],
        "connectors": [*cap_prefixed["connectors"], *tube_prefixed["connectors"]],
        "metadata": {
            "task": args.task,
            "active_joint": active_joint,
            "slot_index": random_info.get("slot_index") if random_info else None,
            "slot_name": random_info.get("slot_name") if random_info else None,
            "tube_arm": tube_arm,
            "cap_arm": cap_arm,
            "cap_body": cap_body,
            "cap_start_pos": cap_start_pos.tolist(),
            "tube_start_pos": tube_start_pos.tolist(),
            "tube_lifted_pos": tube_lifted_pos.tolist(),
            "cap_lift_delta": np.asarray(args.post_offset, dtype=np.float64).tolist(),
            "cap": cap_plan["metadata"],
            "tube_after_cap_lift": lifted_tube_plan["metadata"],
            "tube_at_start_reference": tube_plan["metadata"],
        },
    }


def plan_tube_grasp_points_at_pos(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    active_joint: str,
    random_info: dict[str, Any] | None,
    tube_pos: np.ndarray,
) -> dict[str, Any]:
    arm = resolve_arm(args)
    approach_axis, closing_axis, tool_roll = resolve_axes(args, arm)
    lift_offset = args.lift_offset
    if lift_offset is None:
        lift_offset = np.asarray([0.0, 0.0, args.lift_distance], dtype=np.float64)
    lift_offset = np.asarray(lift_offset, dtype=np.float64)

    grasp_pos = (
        tube_pos
        + np.asarray([0.0, 0.0, args.grasp_height], dtype=np.float64)
        + approach_axis * float(args.pinch_forward_offset)
        - approach_axis * resolve_grasp_outward_offset(args)
    )
    pregrasp_pos = grasp_pos - approach_axis * float(args.pregrasp_distance)
    post_pos = grasp_pos + lift_offset
    plan = marker_plan(mujoco, approach_axis, closing_axis, tool_roll, pregrasp_pos, grasp_pos, post_pos)
    plan["metadata"] = {
        "task": args.task,
        "arm": arm,
        "active_joint": active_joint,
        "slot_index": random_info.get("slot_index") if random_info else None,
        "slot_name": random_info.get("slot_name") if random_info else None,
        "tube_origin_pos": tube_pos.tolist(),
        "grasp_height": args.grasp_height,
        "pinch_forward_offset": args.pinch_forward_offset,
        "grasp_outward_offset": resolve_grasp_outward_offset(args),
        "lift_offset": lift_offset.tolist(),
        "tool_roll": tool_roll,
    }
    return plan


def plan_pipette_grasp_points(
    model: Any,
    data: Any,
    mujoco: Any,
    args: argparse.Namespace,
    active_joint: str,
    random_info: dict[str, Any] | None,
) -> dict[str, Any]:
    arm = resolve_arm(args)
    target = local_grasp_to_world_target(
        model,
        data,
        mujoco,
        LocalGraspPose(
            body=args.pipette_body,
            pos=tuple(np.asarray(args.handle_grasp_offset, dtype=np.float64).tolist()),
            euler=tuple(np.asarray(args.handle_grasp_euler, dtype=np.float64).tolist()),
            gripper_pos=tuple(np.asarray(args.grasp_to_gripper_offset, dtype=np.float64).tolist()),
            gripper_euler=tuple(np.asarray(args.grasp_to_gripper_euler, dtype=np.float64).tolist()),
        ),
    )
    button_dir = unit(target.body_mat[:, 2], "pipette_button_direction")
    tip_dir = -button_dir
    approach_axis = target.gripper_mat[:, 2]
    closing_axis = target.gripper_mat[:, 1]
    pregrasp_pos = target.gripper_pos - approach_axis * float(args.pregrasp_distance)
    post_pos = target.gripper_pos + np.asarray(args.pipette_lift_offset, dtype=np.float64)
    plan = marker_plan(mujoco, approach_axis, closing_axis, 0.0, pregrasp_pos, target.gripper_pos, post_pos)

    frames = [
        {
            "name": "pipette_body_frame",
            "pos": target.body_pos,
            "mat": target.body_mat,
            "scale": 0.075,
            "radius": 0.0045,
            "pose": frame_pose(mujoco, "pipette_body_frame", target.body_pos, target.body_mat),
        },
        {
            "name": "local_grasp_frame",
            "pos": target.grasp_pos,
            "mat": target.grasp_mat,
            "scale": 0.07,
            "radius": 0.004,
            "pose": frame_pose(mujoco, "local_grasp_frame", target.grasp_pos, target.grasp_mat),
        },
        {
            "name": "planned_grasp_gripper_frame",
            "pos": target.gripper_pos,
            "mat": target.gripper_mat,
            "scale": 0.085,
            "radius": 0.005,
            "pose": frame_pose(mujoco, "planned_grasp_gripper_frame", target.gripper_pos, target.gripper_mat),
        },
    ]

    gripper_site_name = args.gripper_site or str(ARM_DEFAULTS[arm]["gripper_site"])
    gripper_frame = site_frame(model, data, mujoco, gripper_site_name)
    if gripper_frame is not None:
        gripper_pos, _, gripper_mat = gripper_frame
        frames.append(
            {
                "name": "current_gripper_site_frame",
                "pos": gripper_pos,
                "mat": gripper_mat,
                "scale": 0.085,
                "radius": 0.004,
                "pose": frame_pose(mujoco, "current_gripper_site_frame", gripper_pos, gripper_mat),
            }
        )

    tip_frame = site_frame(model, data, mujoco, args.pipette_tip_site)
    tip_site_pose = None
    target_tip_info = None
    if tip_frame is not None:
        tip_pos, _, tip_mat = tip_frame
        tip_site_pose = frame_pose(mujoco, "pipette_tip_site_frame", tip_pos, tip_mat)
        frames.append(
            {
                "name": "pipette_tip_site_frame",
                "pos": tip_pos,
                "mat": tip_mat,
                "scale": 0.045,
                "radius": 0.003,
                "pose": tip_site_pose,
            }
        )
        plan["markers"].append(
            {"name": "pipette_tip_site", "pos": tip_pos, "radius": 0.01, "rgba": np.asarray([0.1, 0.9, 1.0, 0.85], dtype=np.float32)}
        )
        target_tip = nearest_active_tip_site_frames(model, data, mujoco, args, tip_pos, random_info)
        if target_tip is not None:
            mount_pose = frame_pose(
                mujoco,
                "target_tip_mount_site_frame",
                target_tip["mount_pos"],
                target_tip["mount_mat"],
            )
            frames.append(
                {
                    "name": "target_tip_mount_site_frame",
                    "pos": target_tip["mount_pos"],
                    "mat": target_tip["mount_mat"],
                    "scale": 0.05,
                    "radius": 0.0035,
                    "pose": mount_pose,
                }
            )
            plan["markers"].append(
                {
                    "name": "target_tip_mount_site",
                    "pos": target_tip["mount_pos"],
                    "radius": 0.008,
                    "rgba": np.asarray([1.0, 0.1, 0.1, 0.85], dtype=np.float32),
                }
            )
            end_pose = None
            if target_tip["end_frame"] is not None:
                end_pos, _, end_mat = target_tip["end_frame"]
                end_pose = frame_pose(mujoco, "target_tip_end_site_frame", end_pos, end_mat)
                frames.append(
                    {
                        "name": "target_tip_end_site_frame",
                        "pos": end_pos,
                        "mat": end_mat,
                        "scale": 0.04,
                        "radius": 0.0025,
                        "pose": end_pose,
                    }
                )
                plan["markers"].append(
                    {
                        "name": "target_tip_end_site",
                        "pos": end_pos,
                        "radius": 0.006,
                        "rgba": np.asarray([0.1, 0.9, 0.1, 0.85], dtype=np.float32),
                    }
                )
            target_tip_info = {
                "joint": target_tip["joint"],
                "mount_site": target_tip["mount_site"],
                "mount_site_pose": mount_pose,
                "end_site": target_tip["end_site"],
                "end_site_pose": end_pose,
                "xy_distance_to_pipette_tip_site": target_tip["xy_distance"],
            }

    plan["frames"] = frames
    plan["metadata"] = {
        "task": args.task,
        "arm": arm,
        "gripper_site": gripper_site_name,
        "pipette_body": args.pipette_body,
        "pipette_body_pos": target.body_pos.tolist(),
        "pipette_body_quat_wxyz": target.body_quat.tolist(),
        "pipette_tip_site": args.pipette_tip_site,
        "pipette_tip_site_pose": tip_site_pose,
        "target_tip": target_tip_info,
        "handle_grasp_offset_body": np.asarray(args.handle_grasp_offset, dtype=np.float64).tolist(),
        "handle_grasp_euler_xyz": np.asarray(args.handle_grasp_euler, dtype=np.float64).tolist(),
        "grasp_to_gripper_offset": np.asarray(args.grasp_to_gripper_offset, dtype=np.float64).tolist(),
        "grasp_to_gripper_euler_xyz": np.asarray(args.grasp_to_gripper_euler, dtype=np.float64).tolist(),
        "approach_axis": approach_axis.tolist(),
        "closing_axis": closing_axis.tolist(),
        "button_direction_world": button_dir.tolist(),
        "tip_direction_world": tip_dir.tolist(),
        "frames": [frame["pose"] for frame in frames],
    }
    return plan


PLANNERS: dict[str, Planner] = {
    "bimanual_unscrew_cap": plan_bimanual_unscrew_cap_points,
    "tube_grasp": plan_tube_grasp_points,
    "cap_grasp": plan_cap_grasp_points,
    "tube_then_cap_grasp": plan_tube_then_cap_grasp_points,
    "pipette_grasp": plan_pipette_grasp_points,
}


def current_site_info(model: Any, data: Any, mujoco: Any, site_name: str) -> dict[str, Any] | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return None
    site_mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    site_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(site_quat, site_mat.reshape(-1))
    return {
        "name": site_name,
        "pos": np.asarray(data.site_xpos[site_id], dtype=np.float64).tolist(),
        "quat_wxyz": site_quat.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize task keypoints for tube/cap manipulation.")
    parser.add_argument("--task", choices=sorted(PLANNERS), default="tube_grasp", help="Keypoint planner to visualize.")
    parser.add_argument("--model", default="model/scenes/scene_mujoco_fast_tubes.xml", help="MuJoCo XML model path.")
    parser.add_argument("--keyframe", default="home", help="Initial keyframe name. Use empty string to skip.")
    parser.add_argument("--reset-config", default="configs/reset_single_tube_random.json", help="Reset config JSON.")
    parser.add_argument("--reset-seed", type=int, default=None, help="Optional reset random seed.")
    parser.add_argument("--active-joint", default="centrifuge_50ml_screw_joint_1", help="Tube free joint to use if reset info is absent.")
    parser.add_argument("--arm", choices=sorted(ARM_DEFAULTS), default=None, help="Robot arm to use. Defaults depend on --task.")
    parser.add_argument("--gripper-site", default=None, help="Current gripper-center site to print for comparison.")
    parser.add_argument("--approach-axis", default=None, help="World direction from pregrasp to grasp; also gripper local +Z.")
    parser.add_argument("--closing-axis", default=None, help="World direction for Robotiq local +Y finger closing axis.")
    parser.add_argument("--tool-roll", type=float, default=None, help="Roll the gripper around local approach/Z axis in radians.")
    parser.add_argument("--pregrasp-distance", type=float, default=0.10, help="Distance before grasp along -approach axis.")
    parser.add_argument("--grasp-outward-offset", type=float, default=None, help="Move grasp point outward, opposite the approach axis. Defaults depend on --task.")

    parser.add_argument("--grasp-height", type=float, default=0.08, help="[tube_grasp] Height above tube free-joint origin.")
    parser.add_argument("--pinch-forward-offset", type=float, default=0.0, help="[tube_grasp] Move pinch target along approach axis.")
    parser.add_argument("--lift-distance", type=float, default=0.10, help="[tube_grasp] Vertical lift distance if --lift-offset is none.")
    parser.add_argument("--lift-offset", type=lambda value: parse_optional_vec3(value, "lift_offset"), default=[0.25, 0.0, 0.12], help="[tube_grasp] World XYZ offset from grasp to post point.")

    parser.add_argument("--cap-body", default=None, help="[cap_grasp] Cap body name. Defaults to the cap attached to active tube.")
    parser.add_argument("--cap-offset", type=lambda value: parse_optional_vec3(value, "cap_offset"), default=[0.0, 0.0, 0.0], help="[cap_grasp] World XYZ offset from cap body center to grasp point.")
    parser.add_argument("--cap-pregrasp-distance", type=float, default=0.10, help="[bimanual_unscrew_cap] Distance before cap grasp along the approach axis.")
    parser.add_argument("--post-distance", type=float, default=0.08, help="[cap_grasp] Vertical post point distance if --post-offset is none.")
    parser.add_argument("--post-offset", type=lambda value: parse_optional_vec3(value, "post_offset"), default=[0.0, 0.0, 0.08], help="[cap_grasp] World XYZ offset from grasp to post point.")

    parser.add_argument("--tube-arm", choices=sorted(ARM_DEFAULTS), default=None, help="[tube_then_cap_grasp] Arm for tube grasp stage.")
    parser.add_argument("--cap-arm", choices=sorted(ARM_DEFAULTS), default=None, help="[tube_then_cap_grasp] Arm for cap grasp stage.")
    parser.add_argument("--tube-approach-axis", default=None, help="[tube_then_cap_grasp] Approach axis for tube grasp stage.")
    parser.add_argument("--tube-closing-axis", default=None, help="[tube_then_cap_grasp] Closing axis for tube grasp stage.")
    parser.add_argument("--cap-approach-axis", default=None, help="[tube_then_cap_grasp] Approach axis for cap grasp stage.")
    parser.add_argument("--cap-closing-axis", default=None, help="[tube_then_cap_grasp] Closing axis for cap grasp stage.")
    parser.add_argument("--tube-tool-roll", type=float, default=float(np.pi), help="[tube_then_cap_grasp] Tool roll for tube grasp stage.")
    parser.add_argument("--cap-tool-roll", type=float, default=None, help="[tube_then_cap_grasp] Tool roll for cap grasp stage.")
    parser.add_argument("--tube-grasp-outward-offset", type=float, default=None, help="[tube_then_cap_grasp] Outward offset for tube grasp stage.")
    parser.add_argument("--cap-grasp-outward-offset", type=float, default=None, help="[tube_then_cap_grasp] Outward offset for cap grasp stage.")

    parser.add_argument("--pipette-body", default="pippipette", help="[pipette_grasp] Pipette body used as the local grasp reference.")
    parser.add_argument("--pipette-tip-site", default="piptip_site", help="[pipette_grasp] Pipette tip site to visualize.")
    parser.add_argument("--tip-joint-prefix", default=PipetteGraspTaskConfig.tip_joint_prefix, help="[pipette_grasp] Prefix for visible tip free joints.")
    parser.add_argument("--tip-site-prefix", default=PipetteGraspTaskConfig.tip_site_prefix, help="[pipette_grasp] Prefix added to attached tip sites.")
    parser.add_argument("--tip-mount-site-suffix", default=PipetteGraspTaskConfig.tip_mount_site_suffix, help="[pipette_grasp] Suffix for tip mount interface site.")
    parser.add_argument("--tip-end-site-suffix", default=PipetteGraspTaskConfig.tip_end_site_suffix, help="[pipette_grasp] Suffix for tip end site.")
    parser.add_argument(
        "--handle-grasp-offset",
        type=lambda value: parse_vec3(value, "handle_grasp_offset"),
        default=list(PipetteGraspTaskConfig.handle_grasp_offset),
        help="[pipette_grasp] Body-local handle grasp offset.",
    )
    parser.add_argument(
        "--handle-grasp-euler",
        type=lambda value: parse_vec3(value, "handle_grasp_euler"),
        default=list(PipetteGraspTaskConfig.handle_grasp_euler),
        help="[pipette_grasp] Body-local grasp frame XYZ Euler angles in radians.",
    )
    parser.add_argument(
        "--grasp-to-gripper-offset",
        type=lambda value: parse_vec3(value, "grasp_to_gripper_offset"),
        default=list(PipetteGraspTaskConfig.grasp_to_gripper_offset),
        help="[pipette_grasp] Grasp-frame-local gripper target offset.",
    )
    parser.add_argument(
        "--grasp-to-gripper-euler",
        type=lambda value: parse_vec3(value, "grasp_to_gripper_euler"),
        default=list(PipetteGraspTaskConfig.grasp_to_gripper_euler),
        help="[pipette_grasp] Grasp-frame-local gripper XYZ Euler angles in radians.",
    )
    parser.add_argument(
        "--pipette-lift-offset",
        type=lambda value: parse_vec3(value, "pipette_lift_offset"),
        default=list(PipetteGraspTaskConfig.lift_offset),
        help="[pipette_grasp] World XYZ offset from grasp to post point.",
    )

    parser.add_argument("--viewer", action="store_true", help="Preview target markers in the MuJoCo viewer.")
    parser.add_argument("--steps-per-sync", type=int, default=5, help="Simulation steps between viewer syncs.")
    parser.add_argument("--settle-steps", type=int, default=20, help="Control steps before computing target markers.")
    parser.add_argument("--control-dt", type=float, default=0.05, help="Control timestep used to convert settle steps into MuJoCo substeps.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override MuJoCo substeps per control step.")
    parser.add_argument("--open-gripper", type=float, default=0.0, help="Open gripper command used during settle.")
    parser.add_argument("--no-hold-active-tube", dest="hold_active_tube", action="store_false", help="Disable scripted holding of active tube.")
    parser.set_defaults(hold_active_tube=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "pipette_grasp":
        if args.model == "model/scenes/scene_mujoco_fast_tubes.xml":
            args.model = "model/scenes/scene_mujoco_fast_tubes_pipette.xml"
        if args.reset_config == "configs/reset_single_tube_random.json":
            args.reset_config = "configs/reset_pipette_tips_random_subset.json"
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer else "egl")

    import mujoco

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    reset_config_path = Path(args.reset_config)
    if not reset_config_path.is_absolute():
        reset_config_path = ROOT / reset_config_path

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)

    reset_info = apply_reset_config(
        model,
        data,
        mujoco,
        load_reset_config(reset_config_path),
        np.random.default_rng(args.reset_seed),
    )
    mujoco.mj_forward(model, data)

    random_info = random_reset_info(reset_info)
    active_joint = str(random_info["active_joint"]) if random_info else args.active_joint
    arm = resolve_arm(args)
    gripper_site = args.gripper_site or str(ARM_DEFAULTS[arm]["gripper_site"])
    gripper_actuator = str(ARM_DEFAULTS[arm]["gripper_actuator"])
    held_tube_state = capture_free_joint_state(model, data, mujoco, active_joint) if args.hold_active_tube else None

    settle_action = np.asarray(data.ctrl, dtype=np.float64).copy()
    settle_action[actuator_id(model, mujoco, gripper_actuator)] = float(args.open_gripper)
    data.ctrl[:] = settle_action
    frame_skip = max(1, int(args.frame_skip)) if args.frame_skip is not None else max(1, round(float(args.control_dt) / model.opt.timestep))
    for _ in range(max(0, args.settle_steps)):
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        if held_tube_state is not None:
            restore_free_joint_state(model, data, mujoco, active_joint, held_tube_state)
    mujoco.mj_forward(model, data)

    plan = PLANNERS[args.task](model, data, mujoco, args, active_joint, random_info)
    result = {
        **plan["metadata"],
        "hold_active_tube": bool(args.hold_active_tube),
        "settle_steps": args.settle_steps,
        "frame_skip": frame_skip,
        "control_dt": float(frame_skip * model.opt.timestep),
        "frame_convention": {
            "quat_order": "wxyz",
            "gripper_local_z": "approach direction from wrist/palm toward pinch center",
            "gripper_local_y": "Robotiq finger closing axis",
        },
        "poses": plan["poses"],
        "frames": [frame["pose"] for frame in plan.get("frames", [])],
        "current_site": current_site_info(model, data, mujoco, gripper_site),
        "reset_info": reset_info,
    }
    print(json.dumps(result, indent=2))

    if not args.viewer:
        return

    import mujoco.viewer

    print("viewer markers: yellow=pregrasp, green=grasp, blue=post, cyan=pipette tip site")
    print("viewer frames: red=X, green=Y, blue=Z; planned_grasp_gripper_frame shows desired gripper end-effector frame")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            for _ in range(max(1, args.steps_per_sync)):
                mujoco.mj_step(model, data)
            if held_tube_state is not None:
                restore_free_joint_state(model, data, mujoco, active_joint, held_tube_state)

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                for marker in plan["markers"]:
                    add_sphere(mujoco, viewer, marker["pos"], marker["radius"], marker["rgba"])
                for connector in plan["connectors"]:
                    geom_type = mujoco.mjtGeom.mjGEOM_ARROW if connector["type"] == "arrow" else mujoco.mjtGeom.mjGEOM_CAPSULE
                    add_connector(
                        mujoco,
                        viewer,
                        geom_type,
                        connector["start"],
                        connector["end"],
                        connector["radius"],
                        connector["rgba"],
                    )
                for frame in plan.get("frames", []):
                    add_frame_axes(
                        mujoco,
                        viewer,
                        frame["pos"],
                        frame["mat"],
                        frame["scale"],
                        frame["radius"],
                    )
            viewer.sync()

            elapsed = time.time() - step_start
            target = model.opt.timestep * max(1, args.steps_per_sync)
            if elapsed < target:
                time.sleep(target - elapsed)


if __name__ == "__main__":
    main()
