'''
机械臂正逆运动学
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class IKResult:
    success: bool
    iterations: int
    pos_error: float
    rot_error: float
    qpos: np.ndarray
    message: str


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Quaternion must not be all zeros")
    return quat / norm


def quat_error_rotvec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    target = normalize_quat(target)
    current = normalize_quat(current)
    error = quat_multiply(target, quat_conjugate(current))
    if error[0] < 0.0:
        error *= -1.0

    vec = error[1:]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vec_norm, float(error[0]))
    return vec / vec_norm * angle


def site_pose(model: Any, data: Any, mujoco: Any, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")

    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return pos, normalize_quat(quat)


def joint_ids(model: Any, mujoco: Any, joint_names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Unknown joint: {joint_name}")
        ids.append(int(joint_id))
    return ids


def actuator_ids(model: Any, mujoco: Any, actuator_names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    for actuator_name in actuator_names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"Unknown actuator: {actuator_name}")
        ids.append(int(actuator_id))
    return ids


def sync_actuators_to_joint_qpos(model: Any, data: Any, mujoco: Any, actuator_names: Sequence[str]) -> None:
    for actuator_id in actuator_ids(model, mujoco, actuator_names):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id < 0:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = data.qpos[qadr]


def solve_site_ik(
    model: Any,
    data: Any,
    mujoco: Any,
    site_name: str,
    joint_names: Sequence[str],
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    max_iters: int = 500,
    pos_tol: float = 0.003,
    rot_tol: float = 0.05,
    damping: float = 0.08,
    max_step: float = 0.04,
    pos_weight: float = 1.0,
    rot_weight: float = 0.35,
    position_only: bool = False,
) -> IKResult:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site: {site_name}")

    joints = joint_ids(model, mujoco, joint_names)
    dof_ids = np.asarray([int(model.jnt_dofadr[joint_id]) for joint_id in joints], dtype=np.int32)
    qpos_ids = np.asarray([int(model.jnt_qposadr[joint_id]) for joint_id in joints], dtype=np.int32)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat = normalize_quat(target_quat)

    last_pos_error = float("inf")
    last_rot_error = float("inf")
    message = "maximum iterations reached"

    for iteration in range(1, max_iters + 1):
        mujoco.mj_forward(model, data)
        current_pos, current_quat = site_pose(model, data, mujoco, site_name)
        pos_err = target_pos - current_pos
        rot_err = quat_error_rotvec(target_quat, current_quat)
        last_pos_error = float(np.linalg.norm(pos_err))
        last_rot_error = float(np.linalg.norm(rot_err))

        if last_pos_error <= pos_tol and (position_only or last_rot_error <= rot_tol):
            message = "converged"
            return IKResult(True, iteration, last_pos_error, last_rot_error, data.qpos.copy(), message)

        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

        if position_only:
            jac = pos_weight * jacp[:, dof_ids]
            err = pos_weight * pos_err
        else:
            jac = np.vstack([pos_weight * jacp[:, dof_ids], rot_weight * jacr[:, dof_ids]])
            err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])

        lhs = jac @ jac.T + (damping**2) * np.eye(jac.shape[0], dtype=np.float64)
        dq = jac.T @ np.linalg.solve(lhs, err)
        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step:
            dq *= max_step / step_norm

        data.qpos[qpos_ids] += dq
        for joint_id, qpos_id in zip(joints, qpos_ids, strict=True):
            if int(model.jnt_limited[joint_id]):
                low, high = model.jnt_range[joint_id]
                data.qpos[qpos_id] = np.clip(data.qpos[qpos_id], low, high)

    mujoco.mj_forward(model, data)
    return IKResult(False, max_iters, last_pos_error, last_rot_error, data.qpos.copy(), message)
