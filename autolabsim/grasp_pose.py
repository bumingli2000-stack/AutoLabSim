'''
物体局部抓取位姿到夹爪目标位姿的通用转换。
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .math3d import euler_xyz_to_mat, mat_to_quat


@dataclass(frozen=True)
class LocalGraspPose:
    body: str
    pos: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GraspTargetPose:
    body_pos: np.ndarray
    body_quat: np.ndarray
    body_mat: np.ndarray
    grasp_pos: np.ndarray
    grasp_quat: np.ndarray
    grasp_mat: np.ndarray
    gripper_pos: np.ndarray
    gripper_quat: np.ndarray
    gripper_mat: np.ndarray


def body_pose(model: Any, data: Any, mujoco: Any, body_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Unknown body: {body_name}")
    pos = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    mat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
    return pos, mat_to_quat(mujoco, mat), mat


def local_grasp_to_world_target(model: Any, data: Any, mujoco: Any, grasp: LocalGraspPose) -> GraspTargetPose:
    body_pos, body_quat, body_mat = body_pose(model, data, mujoco, grasp.body)
    grasp_local_pos = np.asarray(grasp.pos, dtype=np.float64)
    grasp_local_mat = euler_xyz_to_mat(np.asarray(grasp.euler, dtype=np.float64))
    gripper_local_pos = np.asarray(grasp.gripper_pos, dtype=np.float64)
    gripper_local_mat = euler_xyz_to_mat(np.asarray(grasp.gripper_euler, dtype=np.float64))

    grasp_pos = body_pos + body_mat @ grasp_local_pos
    grasp_mat = body_mat @ grasp_local_mat
    gripper_pos = grasp_pos + grasp_mat @ gripper_local_pos
    gripper_mat = grasp_mat @ gripper_local_mat

    return GraspTargetPose(
        body_pos=body_pos,
        body_quat=body_quat,
        body_mat=body_mat,
        grasp_pos=grasp_pos,
        grasp_quat=mat_to_quat(mujoco, grasp_mat),
        grasp_mat=grasp_mat,
        gripper_pos=gripper_pos,
        gripper_quat=mat_to_quat(mujoco, gripper_mat),
        gripper_mat=gripper_mat,
    )
