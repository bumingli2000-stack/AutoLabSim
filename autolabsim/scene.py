'''
MuJoCo 场景访问辅助层
mujoco_env.py 提供运行环境
scene.py 提供读写场景中的对象接口
'''
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .math3d import normalize_quat, quat_conjugate, quat_multiply, quat_to_mat


def actuator_id(model: Any, mujoco: Any, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise ValueError(f"Unknown actuator: {name}")
    return int(idx)


def equality_id(model: Any, mujoco: Any, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if idx < 0:
        raise ValueError(f"Unknown equality constraint: {name}")
    return int(idx)


def joint_qpos_ids(model: Any, mujoco: Any, joint_names: Sequence[str]) -> list[int]:
    qpos_ids: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Unknown joint: {joint_name}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
    return qpos_ids


def free_joint_addresses(model: Any, mujoco: Any, joint_name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Unknown joint: {joint_name}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"Joint is not a free joint: {joint_name}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def free_joint_pose(model: Any, data: Any, mujoco: Any, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    qadr, _ = free_joint_addresses(model, mujoco, joint_name)
    pos = np.asarray(data.qpos[qadr : qadr + 3], dtype=np.float64).copy()
    quat = normalize_quat(np.asarray(data.qpos[qadr + 3 : qadr + 7], dtype=np.float64))
    return pos, quat


def free_joint_pos(model: Any, data: Any, mujoco: Any, joint_name: str) -> np.ndarray:
    pos, _ = free_joint_pose(model, data, mujoco, joint_name)
    return pos


def set_free_joint_pose(
    model: Any,
    data: Any,
    mujoco: Any,
    joint_name: str,
    pos: np.ndarray,
    quat: np.ndarray | None = None,
) -> None:
    qadr, dadr = free_joint_addresses(model, mujoco, joint_name)
    data.qpos[qadr : qadr + 3] = np.asarray(pos, dtype=np.float64)
    if quat is not None:
        data.qpos[qadr + 3 : qadr + 7] = normalize_quat(quat)
    data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def capture_free_joint_state(model: Any, data: Any, mujoco: Any, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    qadr, dadr = free_joint_addresses(model, mujoco, joint_name)
    return data.qpos[qadr : qadr + 7].copy(), data.qvel[dadr : dadr + 6].copy()


def restore_free_joint_state(
    model: Any,
    data: Any,
    mujoco: Any,
    joint_name: str,
    state: tuple[np.ndarray, np.ndarray],
) -> None:
    qadr, dadr = free_joint_addresses(model, mujoco, joint_name)
    qpos, qvel = state
    data.qpos[qadr : qadr + 7] = qpos
    data.qvel[dadr : dadr + 6] = qvel
    mujoco.mj_forward(model, data)


def site_pose(model: Any, data: Any, mujoco: Any, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")
    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(data.site_xmat[site_id], dtype=np.float64))
    return pos, normalize_quat(quat)


def body_pos(model: Any, data: Any, mujoco: Any, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Unknown body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def capture_site_attachment(
    model: Any,
    data: Any,
    mujoco: Any,
    joint_name: str,
    site_name: str,
) -> dict[str, np.ndarray]:
    joint_pos, joint_quat = free_joint_pose(model, data, mujoco, joint_name)
    site_world_pos, site_world_quat = site_pose(model, data, mujoco, site_name)
    site_world_rot = quat_to_mat(site_world_quat)
    local_pos = site_world_rot.T @ (joint_pos - site_world_pos)
    local_quat = normalize_quat(quat_multiply(quat_conjugate(site_world_quat), joint_quat))
    return {"local_pos": local_pos, "local_quat": local_quat}


def apply_site_attachment(
    model: Any,
    data: Any,
    mujoco: Any,
    joint_name: str,
    site_name: str,
    attachment: dict[str, np.ndarray],
) -> None:
    site_world_pos, site_world_quat = site_pose(model, data, mujoco, site_name)
    site_world_rot = quat_to_mat(site_world_quat)
    world_pos = site_world_pos + site_world_rot @ np.asarray(attachment["local_pos"], dtype=np.float64)
    world_quat = normalize_quat(quat_multiply(site_world_quat, np.asarray(attachment["local_quat"], dtype=np.float64)))
    set_free_joint_pose(model, data, mujoco, joint_name, world_pos, world_quat)
