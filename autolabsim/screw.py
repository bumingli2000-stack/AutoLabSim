'''
旋拧行为系统层
管理 cap/tube 的旋拧状态
在抓住瓶盖后进入 engage
跟踪当前 twist angle
按螺纹节距把“旋转”变成“旋转 + 轴向抬升”
达到 release angle 后判定“已经拧开”
拧开后让 cap 继续跟随 gripper
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .math3d import axis_angle_quat, normalize_quat, quat_conjugate, quat_multiply, quat_to_mat, signed_angle_about_axis
from .scene import (
    apply_site_attachment,
    capture_site_attachment,
    equality_id,
    free_joint_pose,
    set_free_joint_pose,
    site_pose,
)
from .sim import System


@dataclass
class ScrewProgress:
    engaged: bool = False
    released: bool = False
    twist_angle: float = 0.0
    lift_distance: float = 0.0


class ScrewCapSystem(System):
    def __init__(
        self,
        *,
        tube_joint: str,
        cap_joint: str,
        cap_site: str,
        weld_name: str,
        release_angle: float,
        thread_pitch: float,
        max_lift: float,
    ):
        self.tube_joint = tube_joint
        self.cap_joint = cap_joint
        self.cap_site = cap_site
        self.weld_name = weld_name
        self.release_angle = float(release_angle)
        self.thread_pitch = float(thread_pitch)
        self.max_lift = float(max_lift)
        self.progress = ScrewProgress()
        self._weld_id: int | None = None
        self._engaged_attachment: dict[str, np.ndarray] | None = None
        self._grasp_reference_quat: np.ndarray | None = None
        self._tube_reference_pos: np.ndarray | None = None
        self._tube_reference_quat: np.ndarray | None = None
        self._cap_relative_pos: np.ndarray | None = None
        self._cap_relative_quat: np.ndarray | None = None
        self._follow_after_release = False
        self._commanded_twist: float | None = None

    def on_reset(self, env) -> None:
        self._weld_id = equality_id(env.model, env.mujoco, self.weld_name)
        env.data.eq_active[self._weld_id] = 1
        env.mujoco.mj_forward(env.model, env.data)
        self.progress = ScrewProgress()
        self._engaged_attachment = None
        self._grasp_reference_quat = None
        self._tube_reference_pos = None
        self._tube_reference_quat = None
        self._cap_relative_pos = None
        self._cap_relative_quat = None
        self._follow_after_release = False
        self._commanded_twist = None

    def set_commanded_twist(self, angle: float | None) -> None:
        self._commanded_twist = None if angle is None else float(angle)

    def release_follow(self) -> None:
        """Stop scripted cap following after the cap has been placed."""
        self._follow_after_release = False
        self._engaged_attachment = None

    def start_follow_after_release(self, env) -> None:
        """Attach the released cap to the cap gripper for transport."""
        if not self.progress.released:
            return
        self._engaged_attachment = capture_site_attachment(env.model, env.data, env.mujoco, self.cap_joint, self.cap_site)
        self._follow_after_release = True

    def engage(self, env) -> None:
        if self.progress.engaged:
            return
        self.progress.engaged = True
        self._engaged_attachment = capture_site_attachment(env.model, env.data, env.mujoco, self.cap_joint, self.cap_site)
        _, self._grasp_reference_quat = site_pose(env.model, env.data, env.mujoco, self.cap_site)
        self._tube_reference_pos, self._tube_reference_quat = free_joint_pose(env.model, env.data, env.mujoco, self.tube_joint)
        cap_pos, cap_quat = free_joint_pose(env.model, env.data, env.mujoco, self.cap_joint)
        tube_rot = quat_to_mat(self._tube_reference_quat)
        self._cap_relative_pos = tube_rot.T @ (cap_pos - self._tube_reference_pos)
        self._cap_relative_quat = normalize_quat(quat_multiply(quat_conjugate(self._tube_reference_quat), cap_quat))
        if self._weld_id is not None:
            env.data.eq_active[self._weld_id] = 0
            env.mujoco.mj_forward(env.model, env.data)

    def after_step(self, env, action, obs) -> None:
        if self.progress.released and self._follow_after_release and self._engaged_attachment is not None:
            apply_site_attachment(env.model, env.data, env.mujoco, self.cap_joint, self.cap_site, self._engaged_attachment)
            return
        if not self.progress.engaged or self.progress.released:
            return
        if self._grasp_reference_quat is None or self._tube_reference_pos is None or self._tube_reference_quat is None:
            return

        _, current_site_quat = site_pose(env.model, env.data, env.mujoco, self.cap_site)
        tube_pos, tube_quat = free_joint_pose(env.model, env.data, env.mujoco, self.tube_joint)
        tube_axis = quat_to_mat(tube_quat)[:, 2]
        screw_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        raw_angle = signed_angle_about_axis(self._grasp_reference_quat, current_site_quat, screw_axis)
        measured_twist = float(np.clip(max(0.0, -raw_angle), 0.0, self.release_angle))
        twist_angle = measured_twist if self._commanded_twist is None else float(np.clip(self._commanded_twist, 0.0, self.release_angle))
        lift_distance = min(self.max_lift, self.thread_pitch * (twist_angle / (2.0 * np.pi)))

        base_pos = tube_pos + quat_to_mat(tube_quat) @ self._cap_relative_pos
        base_quat = normalize_quat(quat_multiply(tube_quat, self._cap_relative_quat))
        twist_quat = axis_angle_quat(screw_axis, -twist_angle)
        world_quat = normalize_quat(quat_multiply(twist_quat, base_quat))
        world_pos = base_pos.copy()
        world_pos[2] = base_pos[2] + lift_distance
        set_free_joint_pose(env.model, env.data, env.mujoco, self.cap_joint, world_pos, world_quat)

        self.progress.twist_angle = twist_angle
        self.progress.lift_distance = lift_distance
        if twist_angle >= self.release_angle - 1e-6:
            self.progress.released = True
