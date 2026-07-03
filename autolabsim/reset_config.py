from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ResetConfig = dict[str, Any]


def load_reset_config(config: str | Path | ResetConfig | None) -> ResetConfig:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config

    path = Path(config)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_reset_config(model: Any, data: Any, mujoco: Any, config: ResetConfig, rng: np.random.Generator | None = None) -> ResetConfig:
    if not config:
        return {}

    resolved: ResetConfig = {}
    rng = rng or np.random.default_rng()

    _apply_actuator_targets(model, data, mujoco, config.get("actuators", {}))
    _apply_free_joint_poses(model, data, mujoco, config.get("free_joints", {}))
    resolved.update(_apply_random_single_free_joint(model, data, mujoco, config.get("random_single_free_joint"), rng))
    return resolved


def _apply_actuator_targets(model: Any, data: Any, mujoco: Any, targets: dict[str, float]) -> None:
    for actuator_name, value in targets.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Unknown actuator in reset config: {actuator_name}")

        value = float(value)
        data.ctrl[actuator_id] = value

        # Position-like robot actuators should start at their target, otherwise
        # the controller immediately pulls the arm away from the desired reset.
        if actuator_name.startswith("2f85"):
            continue

        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id < 0:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        data.qpos[qadr] = value


def _apply_free_joint_poses(model: Any, data: Any, mujoco: Any, poses: dict[str, dict[str, Any]]) -> None:
    for joint_or_body_name, pose in poses.items():
        joint_id = _free_joint_id(model, mujoco, joint_or_body_name)
        if joint_id < 0:
            raise ValueError(f"Unknown free joint or body in reset config: {joint_or_body_name}")

        qadr = int(model.jnt_qposadr[joint_id])
        dofadr = int(model.jnt_dofadr[joint_id])

        if "pos" in pose:
            data.qpos[qadr : qadr + 3] = np.asarray(pose["pos"], dtype=np.float64)
        if "quat" in pose:
            quat = np.asarray(pose["quat"], dtype=np.float64)
            norm = np.linalg.norm(quat)
            if norm == 0:
                raise ValueError(f"Quaternion for {joint_or_body_name} must not be all zeros")
            data.qpos[qadr + 3 : qadr + 7] = quat / norm

        # Reset velocities unless explicitly provided. This avoids stale motion
        # at episode start when positions are changed.
        data.qvel[dofadr : dofadr + 6] = 0.0
        if "vel" in pose:
            data.qvel[dofadr : dofadr + 3] = np.asarray(pose["vel"], dtype=np.float64)
        if "angvel" in pose:
            data.qvel[dofadr + 3 : dofadr + 6] = np.asarray(pose["angvel"], dtype=np.float64)


def _apply_random_single_free_joint(
    model: Any,
    data: Any,
    mujoco: Any,
    config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> ResetConfig:
    if not config:
        return {}

    joints = list(config["joints"])
    slots = list(config["slots"])
    active_joint = str(config.get("active_joint", joints[0]))
    inactive_pose = config.get("inactive_pose", {"pos": [0.0, 0.0, -10.0], "quat": [1.0, 0.0, 0.0, 0.0]})
    companion_joints = config.get("companion_joints", {})
    slot_index = int(rng.integers(0, len(slots)))
    slot = slots[slot_index]
    active_companions: list[str] = []

    for joint_name in joints:
        pose = slot if joint_name == active_joint else inactive_pose
        _apply_free_joint_poses(model, data, mujoco, {joint_name: pose})
        for companion_pose in companion_joints.get(joint_name, []):
            companion_joint = str(companion_pose["joint"])
            resolved_pose = _companion_pose(pose, companion_pose)
            _apply_free_joint_poses(model, data, mujoco, {companion_joint: resolved_pose})
            if joint_name == active_joint:
                active_companions.append(companion_joint)

    return {
        "random_single_free_joint": {
            "active_joint": active_joint,
            "active_companion_joints": active_companions,
            "slot_index": slot_index,
            "slot_name": slot.get("name", str(slot_index)),
            "pose": {
                "pos": list(slot["pos"]),
                "quat": list(slot["quat"]),
            },
        }
    }


def _companion_pose(parent_pose: dict[str, Any], companion_pose: dict[str, Any]) -> dict[str, Any]:
    parent_pos = np.asarray(parent_pose.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
    pos_offset = np.asarray(companion_pose.get("pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    quat = companion_pose.get("quat", parent_pose.get("quat", [1.0, 0.0, 0.0, 0.0]))
    resolved: dict[str, Any] = {
        "pos": (parent_pos + pos_offset).tolist(),
        "quat": quat,
    }
    if "vel" in companion_pose:
        resolved["vel"] = companion_pose["vel"]
    if "angvel" in companion_pose:
        resolved["angvel"] = companion_pose["angvel"]
    return resolved


def _free_joint_id(model: Any, mujoco: Any, joint_or_body_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_or_body_name)
    if joint_id >= 0 and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
        return int(joint_id)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint_or_body_name)
    if body_id < 0:
        return -1

    for candidate in range(model.njnt):
        if int(model.jnt_bodyid[candidate]) == int(body_id) and int(model.jnt_type[candidate]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(candidate)

    return -1
