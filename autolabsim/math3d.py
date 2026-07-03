'''
几何和姿态数学工具层
'''
from __future__ import annotations

import numpy as np


def unit(vec: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise ValueError(f"{name} must not be a zero vector")
    return arr / norm


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Quaternion must not be all zeros")
    arr = arr / norm
    if arr[0] < 0.0:
        arr *= -1.0
    return arr


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(lhs, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(rhs, dtype=np.float64)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(quat)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat(mujoco, mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(-1))
    return normalize_quat(quat)


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = unit(axis, "axis")
    half = angle * 0.5
    return normalize_quat(
        np.asarray(
            [np.cos(half), *(np.sin(half) * axis)],
            dtype=np.float64,
        )
    )


def signed_angle_about_axis(
    reference_quat: np.ndarray,
    current_quat: np.ndarray,
    axis_world: np.ndarray,
) -> float:
    axis_world = unit(axis_world, "axis_world")
    delta = quat_multiply(current_quat, quat_conjugate(reference_quat))
    delta = normalize_quat(delta)
    vec = delta[1:]
    sin_half = float(np.dot(vec, axis_world))
    cos_half = float(delta[0])
    return 2.0 * np.arctan2(sin_half, cos_half)


def gripper_quat_from_axes(
    mujoco,
    approach_axis: np.ndarray,
    closing_axis: np.ndarray,
    tool_roll: float,
) -> np.ndarray:
    z_axis = unit(approach_axis, "approach_axis")
    raw_y = unit(closing_axis, "closing_axis")
    y_axis = raw_y - np.dot(raw_y, z_axis) * z_axis
    y_axis = unit(y_axis, "closing_axis projected perpendicular to approach_axis")
    x_axis = unit(np.cross(y_axis, z_axis), "derived x_axis")
    y_axis = unit(np.cross(z_axis, x_axis), "derived y_axis")

    if abs(tool_roll) > 1e-12:
        cos_roll = float(np.cos(tool_roll))
        sin_roll = float(np.sin(tool_roll))
        rolled_x = cos_roll * x_axis + sin_roll * y_axis
        rolled_y = -sin_roll * x_axis + cos_roll * y_axis
        x_axis = unit(rolled_x, "rolled x_axis")
        y_axis = unit(rolled_y, "rolled y_axis")

    xmat = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
    return mat_to_quat(mujoco, xmat)
