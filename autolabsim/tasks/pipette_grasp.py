'''
单臂夹起移液枪任务
'''
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..grasp_pose import LocalGraspPose, local_grasp_to_world_target
from ..ik import solve_site_ik
from ..math3d import euler_xyz_to_mat, mat_to_quat, normalize_quat, quat_conjugate, quat_multiply, quat_to_mat
from ..mujoco_env import EnvConfig
from ..recorder import EpisodeRecorder
from ..scene import (
    actuator_id,
    capture_free_joint_state,
    capture_site_attachment,
    equality_id,
    free_joint_pose,
    joint_qpos_ids,
    restore_free_joint_state,
    site_pose,
)
from ..task import AutoLabTask, TaskConfig
from .common import ARM_DEFAULTS, json_safe, random_reset_info


@dataclass(frozen=True)
class FrameRef:
    kind: str = "world"
    name: str | None = None


@dataclass(frozen=True)
class TaskTarget:
    name: str
    parent: FrameRef
    pos: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] | None = None
    arm: str = "first"
    controlled_site: str = ""
    servo_mode: str = "none"


@dataclass(frozen=True)
class ResolvedTaskTarget:
    spec: TaskTarget
    pos: np.ndarray
    quat: np.ndarray
    mat: np.ndarray


@dataclass(frozen=True)
class PipetteGraspTaskConfig:
    env: EnvConfig
    out_dir: Path
    episode_index: int
    seed: int
    cameras: tuple[str, ...] = ("overview_camera",)
    with_images: bool = False
    arm: str = "first"
    open_gripper: float = 0.0
    close_gripper: float = 180.0
    settle_steps: int = 20
    free_settle_steps: int = 20
    steps_per_segment: int = 50
    close_steps: int = 12
    hold_steps: int = 20
    grasp_hold_steps: int = 8
    pregrasp_distance: float = 0.08
    handle_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.15)
    handle_grasp_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_to_gripper_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_to_gripper_euler: tuple[float, float, float] = (float(np.pi), 0.0, 0.0)
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift_retry_fractions: tuple[float, ...] = (1.0, 0.75, 0.5)
    pipette_tip_site: str = "piptip_site"
    tip_joint_prefix: str = "pipette_tip_joint_"
    tip_site_prefix: str = "tip"
    tip_mount_site_suffix: str = "mount_site"
    tip_end_site_suffix: str = "tip_end_site"
    tip_pose_servo_enabled: bool = True
    tip_hover_height: float = 0.10
    tip_mount_offset: tuple[float, float, float] = (0.0, 0.0, -0.06)
    tip_mount_target_euler: tuple[float, float, float] = (0.0, 0.0, float(np.pi))
    tool_roll: float = 0.0
    pipette_joint: str = "pipette_joint"
    pipette_body: str = "pippipette"
    parking_weld: str = "pipette_rack_weld"
    vertical_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    tube_joint: str = "centrifuge_50ml_screw_joint_1"
    tube_top_offset: float = 0.103
    tube_hover_height: float = 0.10
    tube_near_height: float = 0.03
    tube_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ik_max_iters: int = 800
    ik_pos_tol: float = 0.0005
    ik_rot_tol: float = 0.02
    ik_damping: float = 0.01
    waypoint_settle_steps: int = 20
    waypoint_settle_pos_tol: float = 0.0005
    visual_servo_enabled: bool = True
    visual_servo_max_iters: int = 12
    visual_servo_steps: int = 10
    visual_servo_pos_tol: float = 0.0001
    visual_servo_rot_tol: float = 0.02
    visual_servo_gain: float = 0.8
    visual_servo_integral_gain: float = 0.25
    visual_servo_max_correction: float = 0.02


class PipetteGraspTask(AutoLabTask):
    name = "pipette_grasp"

    def __init__(self, config: PipetteGraspTaskConfig):
        self.runtime = config
        self.arm = ARM_DEFAULTS[config.arm]
        self.execution_site_errors: list[dict[str, Any]] = []
        self.visual_servo_events: list[dict[str, Any]] = []
        self.tip_target_info: dict[str, Any] | None = None
        self.tube_target_info: dict[str, Any] | None = None
        super().__init__(
            TaskConfig(
                env=config.env,
                with_images=config.with_images,
                cameras=config.cameras,
            )
        )

    def run(self) -> dict[str, Any]:
        obs = self.reset()
        action = np.asarray(obs["ctrl"], dtype=np.float64).copy()
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.arm["gripper_actuator"]))
        action[gripper_id] = self.runtime.open_gripper

        for _ in range(max(0, int(self.runtime.free_settle_steps))):
            obs, *_ = self.manager.step(action)

        initial_pipette_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_joint,
        )
        recorder = EpisodeRecorder(cameras=self.runtime.cameras, with_images=self.runtime.with_images)
        self._hold_action(
            recorder,
            action,
            self.runtime.settle_steps,
            "settle",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
        )

        grasp_plan = self._plan_grasp_waypoints(self.runtime.open_gripper)
        self._execute_plan(
            recorder,
            grasp_plan,
            "move_to_pipette",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
            debug_site=str(self.arm["gripper_site"]),
        )

        grasp_action = np.asarray(grasp_plan[-1]["action"], dtype=np.float64).copy()
        self._hold_action(
            recorder,
            grasp_action,
            self.runtime.grasp_hold_steps,
            "hold_at_grasp",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
        )

        close_action = grasp_action.copy()
        close_action[gripper_id] = self.runtime.close_gripper
        self._move_action(
            recorder,
            close_action,
            self.runtime.close_steps,
            "close_gripper",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
        )
        self._set_equality_active_if_exists(self.runtime.parking_weld, False)

        attachment = capture_site_attachment(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_joint,
            str(self.arm["gripper_site"]),
        )
        lift_plan = self._plan_tip_hover_waypoints(attachment, close_action)
        self._execute_plan(
            recorder,
            lift_plan,
            "move_pipette_to_tip_hover",
            follow_attachments=[(self.runtime.pipette_joint, str(self.arm["gripper_site"]), attachment)],
            debug_site=self.runtime.pipette_tip_site,
        )
        self._hold_action(
            recorder,
            np.asarray(lift_plan[-1]["action"], dtype=np.float64),
            self.runtime.hold_steps,
            "hold_lifted",
            follow_attachments=[(self.runtime.pipette_joint, str(self.arm["gripper_site"]), attachment)],
        )
        if self.tip_target_info is not None:
            self._record_site_target_error(
                "hold_over_tip",
                self.runtime.pipette_tip_site,
                self.tip_target_info["target_tip_hover_pos"],
            )

        target_tip_joint = self._target_tip_joint()
        target_tip_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            target_tip_joint,
        )
        mount_down_plan = self._plan_tip_mount_down_waypoints(attachment)
        self._execute_plan(
            recorder,
            mount_down_plan,
            "mount_pipette_tip",
            follow_attachments=[(self.runtime.pipette_joint, str(self.arm["gripper_site"]), attachment)],
            fixed_joint_states=[(target_tip_joint, target_tip_state)],
            debug_site=self.runtime.pipette_tip_site,
        )
        tip_attachment = capture_site_attachment(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            target_tip_joint,
            self.runtime.pipette_tip_site,
        )

        attached_follow = [
            (self.runtime.pipette_joint, str(self.arm["gripper_site"]), attachment),
            (target_tip_joint, self.runtime.pipette_tip_site, tip_attachment),
        ]
        tip_retract_plan = self._plan_tip_retract_waypoints(attachment)
        self._execute_plan(
            recorder,
            tip_retract_plan,
            "retract_mounted_tip",
            follow_attachments=attached_follow,
            debug_site=self.runtime.pipette_tip_site,
        )

        tube_plan = self._plan_tube_approach_waypoints(attachment)
        self._execute_plan(
            recorder,
            tube_plan,
            "move_tip_to_tube",
            follow_attachments=attached_follow,
            debug_site=self.runtime.pipette_tip_site,
        )
        self._hold_action(
            recorder,
            np.asarray(tube_plan[-1]["action"], dtype=np.float64),
            self.runtime.hold_steps,
            "hold_tip_near_tube",
            follow_attachments=attached_follow,
        )
        if self.tube_target_info is not None:
            self._record_site_target_error(
                "hold_tip_near_tube",
                self.runtime.pipette_tip_site,
                self.tube_target_info["tube_near_tip_site_pos"],
            )

        arrays = recorder.to_arrays()
        metadata = self._make_metadata(
            grasp_plan,
            lift_plan,
            mount_down_plan,
            tip_retract_plan,
            tube_plan,
            tip_attachment=tip_attachment,
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    def _plan_grasp_waypoints(self, gripper_value: float) -> list[dict[str, Any]]:
        target = local_grasp_to_world_target(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            LocalGraspPose(
                body=self.runtime.pipette_body,
                pos=self.runtime.handle_grasp_offset,
                euler=self.runtime.handle_grasp_euler,
                gripper_pos=self.runtime.grasp_to_gripper_offset,
                gripper_euler=self.runtime.grasp_to_gripper_euler,
            ),
        )
        waypoints = []
        if self.runtime.pregrasp_distance > 0.0:
            approach = target.gripper_mat[:, 2]
            pregrasp_pos = target.gripper_pos - approach * self.runtime.pregrasp_distance
            waypoints.append(
                self._target_to_waypoint(
                    TaskTarget(
                        name="pipette_pregrasp",
                        parent=FrameRef("world"),
                        pos=tuple(pregrasp_pos.tolist()),
                        quat_wxyz=tuple(target.gripper_quat.tolist()),
                        arm=self.runtime.arm,
                        controlled_site=str(self.arm["gripper_site"]),
                        servo_mode="none",
                    )
                )
            )
        waypoints.append(
            self._target_to_waypoint(
                TaskTarget(
                    name="pipette_grasp",
                    parent=FrameRef("world"),
                    pos=tuple(target.gripper_pos.tolist()),
                    quat_wxyz=tuple(target.gripper_quat.tolist()),
                    arm=self.runtime.arm,
                    controlled_site=str(self.arm["gripper_site"]),
                    servo_mode="position",
                )
            )
        )
        return self._plan_arm(waypoints, gripper_value)

    def _plan_tip_hover_waypoints(self, attachment: dict[str, np.ndarray], start_action: np.ndarray) -> list[dict[str, Any]]:
        _, current_joint_quat = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_joint)
        current_joint_quat = normalize_quat(current_joint_quat)
        aligned_joint_quat = normalize_quat(np.asarray(self.runtime.vertical_quat, dtype=np.float64))
        _, current_site_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_tip_site)
        aligned_site_quat = self._site_quat_for_joint_quat(self.runtime.pipette_tip_site, aligned_joint_quat)
        target_tip = self._nearest_active_tip()
        mount_target_pos, mount_target_quat = self._tip_mount_target_pose(target_tip)
        target_tip_hover_pos = mount_target_pos + np.asarray([0.0, 0.0, self.runtime.tip_hover_height], dtype=np.float64)
        self.tip_target_info = {
            "tip_joint": target_tip["joint"],
            "tip_slot_name": target_tip.get("slot_name"),
            "tip_pos": target_tip["pos"].tolist(),
            "tip_mount_site": target_tip.get("mount_site"),
            "tip_mount_pos": target_tip["mount_pos"].tolist(),
            "tip_mount_quat": target_tip["mount_quat"].tolist(),
            "tip_end_site": target_tip.get("end_site"),
            "tip_end_pos": target_tip["end_pos"].tolist(),
            "tip_end_quat": target_tip["end_quat"].tolist(),
            "tip_mount_target_pos": mount_target_pos.tolist(),
            "tip_mount_target_quat": mount_target_quat.tolist(),
            "target_tip_hover_pos": target_tip_hover_pos.tolist(),
            "tip_xy_distance": target_tip["xy_distance"],
        }

        waypoints = [
            self._target_to_waypoint(
                TaskTarget(
                    name="pipette_tip_hover_translate",
                    parent=FrameRef("world"),
                    pos=tuple(target_tip_hover_pos.tolist()),
                    quat_wxyz=tuple(current_site_quat.tolist()),
                    arm=self.runtime.arm,
                    controlled_site=self.runtime.pipette_tip_site,
                    servo_mode="none",
                ),
                attachment=attachment,
            ),
            self._target_to_waypoint(
                TaskTarget(
                    name="pipette_tip_hover_align",
                    parent=FrameRef("world"),
                    pos=tuple(target_tip_hover_pos.tolist()),
                    quat_wxyz=tuple((mount_target_quat if self.runtime.tip_pose_servo_enabled else aligned_site_quat).tolist()),
                    arm=self.runtime.arm,
                    controlled_site=self.runtime.pipette_tip_site,
                    servo_mode="pose" if self.runtime.tip_pose_servo_enabled else "position",
                ),
                attachment=attachment,
            ),
        ]
        plan = self._plan_arm(waypoints, self.runtime.close_gripper)
        plan[0]["tip_target"] = self.tip_target_info
        plan[0]["keeps_current_pipette_quat"] = True
        plan[1]["tip_target"] = self.tip_target_info
        plan[1]["aligns_pipette_quat"] = aligned_joint_quat.tolist()
        for item in plan:
            item["action"] = np.asarray(item["action"], dtype=np.float64)
        return plan

    def _plan_tip_mount_down_waypoints(self, attachment: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        target_tip = self._target_tip_info()
        target = self._tip_mount_task_target(
            "pipette_tip_mount_down",
            target_tip,
            extra_offset=(0.0, 0.0, 0.0),
            servo_mode="pose" if self.runtime.tip_pose_servo_enabled else "position",
        )
        plan = self._plan_arm([self._target_to_waypoint(target, attachment=attachment)], self.runtime.close_gripper)
        plan[0]["tip_target"] = self.tip_target_info
        plan[0]["mount_tip_site_pos"] = plan[0]["task_target"]["world_pos"]
        plan[0]["action"] = np.asarray(plan[0]["action"], dtype=np.float64)
        return plan

    def _plan_tip_retract_waypoints(self, attachment: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        target_tip = self._target_tip_info()
        target = self._tip_mount_task_target(
            "pipette_tip_mounted_retract",
            target_tip,
            extra_offset=(0.0, 0.0, self.runtime.tip_hover_height),
            servo_mode="pose" if self.runtime.tip_pose_servo_enabled else "position",
        )
        plan = self._plan_arm([self._target_to_waypoint(target, attachment=attachment)], self.runtime.close_gripper)
        plan[0]["tip_target"] = self.tip_target_info
        plan[0]["retract_tip_site_pos"] = plan[0]["task_target"]["world_pos"]
        plan[0]["action"] = np.asarray(plan[0]["action"], dtype=np.float64)
        return plan

    def _plan_tube_approach_waypoints(self, attachment: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        active_tube_joint = self._active_tube_joint()
        tube_pos, _ = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, active_tube_joint)
        tube_parent = FrameRef("free_joint", active_tube_joint)
        _, current_site_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_tip_site)
        tube_relative_base = np.asarray(self.runtime.tube_target_offset, dtype=np.float64)
        hover_relative_pos = tuple(
            (tube_relative_base + np.asarray([0.0, 0.0, self.runtime.tube_top_offset + self.runtime.tube_hover_height])).tolist()
        )
        near_relative_pos = tuple(
            (tube_relative_base + np.asarray([0.0, 0.0, self.runtime.tube_top_offset + self.runtime.tube_near_height])).tolist()
        )
        relative_site_quat = tuple(self._relative_quat_for_world_quat(tube_parent, current_site_quat).tolist())
        hover_target = TaskTarget(
            name="pipette_tip_tube_hover",
            parent=tube_parent,
            pos=hover_relative_pos,
            quat_wxyz=relative_site_quat,
            arm=self.runtime.arm,
            controlled_site=self.runtime.pipette_tip_site,
            servo_mode="none",
        )
        near_target = TaskTarget(
            name="pipette_tip_tube_near",
            parent=tube_parent,
            pos=near_relative_pos,
            quat_wxyz=relative_site_quat,
            arm=self.runtime.arm,
            controlled_site=self.runtime.pipette_tip_site,
            servo_mode="position",
        )
        hover_resolved = self._resolve_task_target(hover_target)
        near_resolved = self._resolve_task_target(near_target)
        tube_top = tube_pos + tube_relative_base + np.asarray([0.0, 0.0, self.runtime.tube_top_offset], dtype=np.float64)
        self.tube_target_info = {
            "tube_joint": active_tube_joint,
            "tube_pos": tube_pos.tolist(),
            "tube_top_pos": tube_top.tolist(),
            "tube_hover_tip_site_pos": hover_resolved.pos.tolist(),
            "tube_near_tip_site_pos": near_resolved.pos.tolist(),
        }
        plan = self._plan_arm(
            [
                self._target_to_waypoint(hover_target, attachment=attachment),
                self._target_to_waypoint(near_target, attachment=attachment),
            ],
            self.runtime.close_gripper,
        )
        for item in plan:
            item["tube_target"] = self.tube_target_info
            item["action"] = np.asarray(item["action"], dtype=np.float64)
        return plan

    def _tip_mount_target_pose(self, target_tip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        fallback_pos = target_tip.get("tip_pos", target_tip.get("pos"))
        if fallback_pos is None:
            raise KeyError("target_tip must contain one of mount_pos, tip_pos, or pos")
        mount_pos = np.asarray(target_tip.get("mount_pos", fallback_pos), dtype=np.float64)
        mount_quat = normalize_quat(
            np.asarray(target_tip.get("mount_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        )
        mount_mat = quat_to_mat(mount_quat)
        relative_pos = np.asarray(self.runtime.tip_mount_offset, dtype=np.float64)
        relative_mat = euler_xyz_to_mat(np.asarray(self.runtime.tip_mount_target_euler, dtype=np.float64))
        target_pos = mount_pos + mount_mat @ relative_pos
        target_quat = mat_to_quat(self.env.mujoco, mount_mat @ relative_mat)
        return target_pos, target_quat

    def _tip_mount_task_target(
        self,
        name: str,
        target_tip: dict[str, Any],
        *,
        extra_offset: tuple[float, float, float],
        servo_mode: str,
    ) -> TaskTarget:
        site_name = target_tip.get("tip_mount_site") or target_tip.get("mount_site")
        base_offset = np.asarray(self.runtime.tip_mount_offset, dtype=np.float64)
        pos = tuple((base_offset + np.asarray(extra_offset, dtype=np.float64)).tolist())
        parent = FrameRef("site", str(site_name)) if site_name else FrameRef("world")
        quat_wxyz = None
        euler = self.runtime.tip_mount_target_euler
        if not site_name:
            world_pos, world_quat = self._tip_mount_target_pose(target_tip)
            return TaskTarget(
                name=name,
                parent=FrameRef("world"),
                pos=tuple((world_pos + np.asarray(extra_offset, dtype=np.float64)).tolist()),
                quat_wxyz=tuple(world_quat.tolist()),
                arm=self.runtime.arm,
                controlled_site=self.runtime.pipette_tip_site,
                servo_mode=servo_mode,
            )
        if servo_mode != "pose":
            _, current_site_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_tip_site)
            quat_wxyz = tuple(self._relative_quat_for_world_quat(parent, current_site_quat).tolist())
        return TaskTarget(
            name=name,
            parent=parent,
            pos=pos,
            euler=euler,
            quat_wxyz=quat_wxyz,
            arm=self.runtime.arm,
            controlled_site=self.runtime.pipette_tip_site,
            servo_mode=servo_mode,
        )

    def _relative_quat_for_world_quat(self, parent: FrameRef, world_quat: np.ndarray) -> np.ndarray:
        _, parent_quat, _ = self._resolve_frame_ref(parent)
        return normalize_quat(quat_multiply(quat_conjugate(parent_quat), world_quat))

    def _resolve_frame_ref(self, frame: FrameRef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if frame.kind == "world":
            mat = np.eye(3, dtype=np.float64)
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return np.zeros(3, dtype=np.float64), quat, mat
        if not frame.name:
            raise ValueError(f"FrameRef kind {frame.kind!r} requires a name")
        if frame.kind == "site":
            pos, quat = site_pose(self.env.model, self.env.data, self.env.mujoco, frame.name)
            return pos, quat, quat_to_mat(quat)
        if frame.kind == "free_joint":
            pos, quat = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, frame.name)
            return pos, quat, quat_to_mat(quat)
        if frame.kind == "body":
            body_id = self.env.mujoco.mj_name2id(
                self.env.model,
                self.env.mujoco.mjtObj.mjOBJ_BODY,
                frame.name,
            )
            if body_id < 0:
                raise ValueError(f"Unknown body frame: {frame.name}")
            pos = np.asarray(self.env.data.xpos[body_id], dtype=np.float64).copy()
            mat = np.asarray(self.env.data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
            return pos, mat_to_quat(self.env.mujoco, mat), mat
        raise ValueError(f"Unknown frame kind: {frame.kind}")

    def _resolve_task_target(self, target: TaskTarget) -> ResolvedTaskTarget:
        parent_pos, _, parent_mat = self._resolve_frame_ref(target.parent)
        local_pos = np.asarray(target.pos, dtype=np.float64)
        local_mat = (
            quat_to_mat(np.asarray(target.quat_wxyz, dtype=np.float64))
            if target.quat_wxyz is not None
            else euler_xyz_to_mat(np.asarray(target.euler, dtype=np.float64))
        )
        mat = parent_mat @ local_mat
        pos = parent_pos + parent_mat @ local_pos
        return ResolvedTaskTarget(target, pos, mat_to_quat(self.env.mujoco, mat), mat)

    def _target_to_waypoint(
        self,
        target: TaskTarget,
        *,
        attachment: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_task_target(target)
        gripper_site = str(ARM_DEFAULTS[target.arm]["gripper_site"])
        if target.controlled_site == gripper_site:
            waypoint = {"name": target.name, "pos": resolved.pos, "quat": resolved.quat}
        elif target.controlled_site == self.runtime.pipette_tip_site:
            if attachment is None:
                raise ValueError(f"{target.name} controls {target.controlled_site}, so an attachment is required")
            target_joint_quat = self._joint_quat_for_site_target(target.controlled_site, resolved.quat)
            target_joint_pos = self._joint_pos_for_site_target(target.controlled_site, resolved.pos, target_joint_quat)
            waypoint = self._gripper_waypoint_for_joint_pose(
                target.name,
                target_joint_pos,
                target_joint_quat,
                attachment,
            )
        else:
            raise ValueError(f"Unsupported controlled site for {target.name}: {target.controlled_site}")

        waypoint["task_target"] = self._task_target_metadata(target, resolved)
        waypoint["servo_mode"] = target.servo_mode
        if target.servo_mode != "none":
            waypoint["debug_target_pos"] = resolved.pos.tolist()
            if target.servo_mode == "pose":
                waypoint["debug_target_quat"] = resolved.quat.tolist()
        return waypoint

    @staticmethod
    def _task_target_metadata(target: TaskTarget, resolved: ResolvedTaskTarget) -> dict[str, Any]:
        return {
            "name": target.name,
            "parent": {"kind": target.parent.kind, "name": target.parent.name},
            "relative_pos": list(target.pos),
            "relative_euler_xyz": list(target.euler),
            "relative_quat_wxyz": list(target.quat_wxyz) if target.quat_wxyz is not None else None,
            "arm": target.arm,
            "controlled_site": target.controlled_site,
            "servo_mode": target.servo_mode,
            "world_pos": resolved.pos.tolist(),
            "world_quat_wxyz": resolved.quat.tolist(),
        }

    def _site_quat_for_joint_quat(self, site_name: str, target_joint_quat: np.ndarray) -> np.ndarray:
        _, joint_quat = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_joint)
        _, site_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        local_site_quat = normalize_quat(quat_multiply(quat_conjugate(joint_quat), site_quat))
        return normalize_quat(quat_multiply(target_joint_quat, local_site_quat))

    def _target_tip_info(self) -> dict[str, Any]:
        if self.tip_target_info is None:
            raise RuntimeError("Tip target is not available; plan tip hover before mounting the tip")
        return self.tip_target_info

    def _target_tip_joint(self) -> str:
        tip_target = self._target_tip_info()
        return str(tip_target["tip_joint"])

    def _active_tube_joint(self) -> str:
        info = random_reset_info(self.env.last_reset_info)
        if info is not None and info.get("active_joint"):
            return str(info["active_joint"])
        return self.runtime.tube_joint

    def _tip_site_name(self, joint_name: str, suffix: str) -> str:
        if not joint_name.startswith(self.runtime.tip_joint_prefix):
            raise ValueError(f"Tip joint does not match prefix {self.runtime.tip_joint_prefix!r}: {joint_name}")
        return f"{self.runtime.tip_site_prefix}{joint_name[len(self.runtime.tip_joint_prefix):]}{suffix}"

    def _optional_site_pos(self, site_name: str) -> np.ndarray | None:
        try:
            pos, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        except ValueError:
            return None
        return pos

    def _optional_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            return site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        except ValueError:
            return None

    def _nearest_active_tip(self) -> dict[str, Any]:
        pipette_tip_pos, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_tip_site)
        candidates: list[dict[str, Any]] = []
        subset_info = self.env.last_reset_info.get("random_free_joint_subset", {})
        active = subset_info.get("active", []) if isinstance(subset_info, dict) else []
        for item in active:
            if not isinstance(item, dict) or "joint" not in item:
                continue
            candidates.append(
                {
                    "joint": str(item["joint"]),
                    "slot_name": item.get("slot_name"),
                }
            )

        if not candidates:
            for joint_name in self.env.joint_names:
                if joint_name.startswith(self.runtime.tip_joint_prefix):
                    candidates.append({"joint": joint_name, "slot_name": None})

        active_tips: list[dict[str, Any]] = []
        for item in candidates:
            tip_pos, _ = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, item["joint"])
            if tip_pos[2] < -1.0:
                continue
            mount_site = self._tip_site_name(item["joint"], self.runtime.tip_mount_site_suffix)
            end_site = self._tip_site_name(item["joint"], self.runtime.tip_end_site_suffix)
            mount_pose = self._optional_site_pose(mount_site)
            end_pose = self._optional_site_pose(end_site)
            if mount_pose is None:
                mount_site = None
                mount_pos = tip_pos
                mount_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                mount_pos, mount_quat = mount_pose
            if end_pose is None:
                end_site = None
                end_pos = tip_pos
                end_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                end_pos, end_quat = end_pose
            xy_distance = float(np.linalg.norm(mount_pos[:2] - pipette_tip_pos[:2]))
            active_tips.append(
                {
                    **item,
                    "pos": tip_pos,
                    "mount_site": mount_site,
                    "mount_pos": mount_pos,
                    "mount_quat": mount_quat,
                    "end_site": end_site,
                    "end_pos": end_pos,
                    "end_quat": end_quat,
                    "xy_distance": xy_distance,
                }
            )

        if not active_tips:
            raise RuntimeError("No active pipette tips are visible in the scene")
        return min(active_tips, key=lambda item: item["xy_distance"])

    def _joint_pos_for_site_target(
        self,
        site_name: str,
        target_site_pos: np.ndarray,
        target_joint_quat: np.ndarray,
    ) -> np.ndarray:
        joint_pos, joint_quat = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_joint)
        site_pos, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        local_site_pos = quat_to_mat(joint_quat).T @ (site_pos - joint_pos)
        return np.asarray(target_site_pos, dtype=np.float64) - quat_to_mat(target_joint_quat) @ local_site_pos

    def _joint_quat_for_site_target(self, site_name: str, target_site_quat: np.ndarray) -> np.ndarray:
        _, joint_quat = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_joint)
        _, site_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        local_site_quat = normalize_quat(quat_multiply(quat_conjugate(joint_quat), site_quat))
        return normalize_quat(quat_multiply(target_site_quat, quat_conjugate(local_site_quat)))

    @staticmethod
    def _gripper_waypoint_for_joint_pose(
        name: str,
        target_joint_pos: np.ndarray,
        target_joint_quat: np.ndarray,
        attachment: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        local_quat = normalize_quat(np.asarray(attachment["local_quat"], dtype=np.float64))
        site_quat = normalize_quat(quat_multiply(target_joint_quat, quat_conjugate(local_quat)))
        site_pos = np.asarray(target_joint_pos, dtype=np.float64) - quat_to_mat(site_quat) @ np.asarray(
            attachment["local_pos"],
            dtype=np.float64,
        )
        return {"name": name, "pos": site_pos, "quat": site_quat}

    def _plan_arm(self, waypoints: list[dict[str, Any]], gripper_value: float) -> list[dict[str, Any]]:
        joint_names = tuple(self.arm["joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.arm["gripper_actuator"]))

        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        plan: list[dict[str, Any]] = []
        for waypoint in waypoints:
            result = solve_site_ik(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                str(self.arm["gripper_site"]),
                joint_names,
                waypoint["pos"],
                waypoint["quat"],
                max_iters=self.runtime.ik_max_iters,
                pos_tol=self.runtime.ik_pos_tol,
                rot_tol=self.runtime.ik_rot_tol,
                damping=self.runtime.ik_damping,
            )
            action = start_ctrl.copy()
            for action_id, qpos_id in zip(arm_actuator_ids, qpos_ids, strict=True):
                action[action_id] = result.qpos[qpos_id]
            action[gripper_id] = gripper_value
            plan_item = {
                "name": waypoint["name"],
                "target_pos": np.asarray(waypoint["pos"], dtype=np.float64).tolist(),
                "target_quat_wxyz": np.asarray(waypoint["quat"], dtype=np.float64).tolist(),
                "action": action,
                "ik_success": result.success,
                "ik_pos_error": result.pos_error,
                "ik_rot_error": result.rot_error,
                "arm_joint_names": list(joint_names),
                "arm_qpos": result.qpos[qpos_ids].tolist(),
            }
            for key in ("task_target", "servo_mode", "debug_target_pos", "debug_target_quat"):
                if key in waypoint:
                    plan_item[key] = waypoint[key]
            plan.append(plan_item)

        self.env.data.qpos[:] = start_qpos
        self.env.data.qvel[:] = start_qvel
        self.env.data.ctrl[:] = start_ctrl
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        return plan

    def _execute_plan(
        self,
        recorder: EpisodeRecorder,
        plan: list[dict[str, Any]],
        phase_prefix: str,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        debug_site: str | None = None,
    ) -> None:
        for item in plan:
            phase = f"{phase_prefix}:{item['name']}"
            self._move_action(
                recorder,
                np.asarray(item["action"], dtype=np.float64),
                self.runtime.steps_per_segment,
                phase,
                fixed_joint_states=fixed_joint_states,
                follow_attachments=follow_attachments,
            )
            if debug_site is not None:
                servo_mode = str(item.get("servo_mode", "position"))
                debug_target_pos = item.get("debug_target_pos", item["target_pos"])
                debug_target_quat = item.get("debug_target_quat")
                if servo_mode == "none":
                    continue
                if self.runtime.visual_servo_enabled:
                    item["action"] = self._visual_servo_site_to_target(
                        recorder,
                        np.asarray(item["action"], dtype=np.float64),
                        phase,
                        debug_site,
                        debug_target_pos,
                        debug_target_quat if servo_mode == "pose" else None,
                        fixed_joint_states=fixed_joint_states,
                        follow_attachments=follow_attachments,
                    )
                else:
                    self._settle_until_site_reached(
                        recorder,
                        np.asarray(item["action"], dtype=np.float64),
                        phase,
                        debug_site,
                        debug_target_pos,
                        fixed_joint_states=fixed_joint_states,
                        follow_attachments=follow_attachments,
                    )
                self._record_site_target_error(phase, debug_site, debug_target_pos)

    def _move_action(
        self,
        recorder: EpisodeRecorder,
        target_action: np.ndarray,
        steps: int,
        phase: str,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> None:
        start = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        denom = max(1, int(steps))
        for step in range(1, denom + 1):
            alpha = step / denom
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            action = (1.0 - alpha) * start + alpha * target_action
            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints(fixed_joint_states, follow_attachments) or obs
            recorder.record(obs, action, phase)

    def _hold_action(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        steps: int,
        phase: str,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> None:
        for _ in range(max(0, int(steps))):
            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints(fixed_joint_states, follow_attachments) or obs
            recorder.record(obs, action, phase)

    def _settle_until_site_reached(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> None:
        target = np.asarray(target_pos, dtype=np.float64)
        settle_phase = f"{phase}:settle"
        for _ in range(max(0, int(self.runtime.waypoint_settle_steps))):
            actual, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
            if float(np.linalg.norm(target - actual)) <= float(self.runtime.waypoint_settle_pos_tol):
                return
            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints(fixed_joint_states, follow_attachments) or obs
            recorder.record(obs, action, settle_phase)

    def _visual_servo_site_to_target(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        target_quat: Any | None = None,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> np.ndarray:
        target = np.asarray(target_pos, dtype=np.float64)
        target_quat_arr = None if target_quat is None else normalize_quat(np.asarray(target_quat, dtype=np.float64))
        current_action = np.asarray(action, dtype=np.float64).copy()
        servo_phase = f"{phase}:visual_servo"
        integral_error = np.zeros(3, dtype=np.float64)
        for iteration in range(max(0, int(self.runtime.visual_servo_max_iters))):
            actual, actual_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
            error = target - actual
            error_norm = float(np.linalg.norm(error))
            rot_error_norm = 0.0
            if target_quat_arr is not None:
                rot_error_norm = self._quat_error_norm(target_quat_arr, actual_quat)
            if error_norm <= float(self.runtime.visual_servo_pos_tol) and (
                target_quat_arr is None or rot_error_norm <= float(self.runtime.visual_servo_rot_tol)
            ):
                self.visual_servo_events.append(
                    {
                        "phase": phase,
                        "site": site_name,
                        "iterations": iteration,
                        "final_error_norm": error_norm,
                        "final_rot_error_norm": rot_error_norm,
                        "converged": True,
                        "integral_gain": self.runtime.visual_servo_integral_gain,
                    }
                )
                return current_action

            gripper_pos, gripper_quat = site_pose(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                str(self.arm["gripper_site"]),
            )
            target_gripper_quat = gripper_quat
            if target_quat_arr is not None:
                local_site_quat = normalize_quat(quat_multiply(quat_conjugate(gripper_quat), actual_quat))
                target_gripper_quat = normalize_quat(quat_multiply(target_quat_arr, quat_conjugate(local_site_quat)))
            integral_error += error
            correction = (
                error * float(self.runtime.visual_servo_gain)
                + integral_error * float(self.runtime.visual_servo_integral_gain)
            )
            correction_norm = float(np.linalg.norm(correction))
            max_correction = float(self.runtime.visual_servo_max_correction)
            if max_correction > 0.0 and correction_norm > max_correction:
                correction *= max_correction / correction_norm
            correction_action = self._solve_gripper_servo_action(
                gripper_pos + correction,
                target_gripper_quat,
                current_action,
            )
            self._move_action(
                recorder,
                correction_action,
                self.runtime.visual_servo_steps,
                servo_phase,
                fixed_joint_states=fixed_joint_states,
                follow_attachments=follow_attachments,
            )
            current_action = correction_action

        actual, actual_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        final_error_norm = float(np.linalg.norm(target - actual))
        final_rot_error_norm = (
            0.0 if target_quat_arr is None else self._quat_error_norm(target_quat_arr, actual_quat)
        )
        self.visual_servo_events.append(
            {
                "phase": phase,
                "site": site_name,
                "iterations": int(self.runtime.visual_servo_max_iters),
                "final_error_norm": final_error_norm,
                "final_rot_error_norm": final_rot_error_norm,
                "converged": final_error_norm <= float(self.runtime.visual_servo_pos_tol)
                and (target_quat_arr is None or final_rot_error_norm <= float(self.runtime.visual_servo_rot_tol)),
                "integral_gain": self.runtime.visual_servo_integral_gain,
            }
        )
        return current_action

    @staticmethod
    def _quat_error_norm(target_quat: np.ndarray, actual_quat: np.ndarray) -> float:
        delta = normalize_quat(quat_multiply(target_quat, quat_conjugate(actual_quat)))
        vec_norm = float(np.linalg.norm(delta[1:]))
        if vec_norm < 1e-12:
            return 0.0
        return float(2.0 * np.arctan2(vec_norm, abs(float(delta[0]))))

    def _solve_gripper_servo_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        base_action: np.ndarray,
    ) -> np.ndarray:
        joint_names = tuple(self.arm["joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]

        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        result = solve_site_ik(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            str(self.arm["gripper_site"]),
            joint_names,
            target_pos,
            target_quat,
            max_iters=self.runtime.ik_max_iters,
            pos_tol=min(float(self.runtime.ik_pos_tol), float(self.runtime.visual_servo_pos_tol)),
            rot_tol=self.runtime.ik_rot_tol,
            damping=self.runtime.ik_damping,
        )
        action = np.asarray(base_action, dtype=np.float64).copy()
        for action_id, qpos_id in zip(arm_actuator_ids, qpos_ids, strict=True):
            action[action_id] = result.qpos[qpos_id]

        self.env.data.qpos[:] = start_qpos
        self.env.data.qvel[:] = start_qvel
        self.env.data.ctrl[:] = start_ctrl
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        return action

    def _apply_constraints(
        self,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None,
    ) -> dict[str, Any] | None:
        constrained = False
        if fixed_joint_states:
            for joint_name, state in fixed_joint_states:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, joint_name, state)
            constrained = True
        if follow_attachments:
            from ..scene import apply_site_attachment

            for joint_name, site_name, attachment in follow_attachments:
                apply_site_attachment(self.env.model, self.env.data, self.env.mujoco, joint_name, site_name, attachment)
            constrained = True
        return self.env.get_observation() if constrained else None

    def _set_equality_active_if_exists(self, name: str, active: bool) -> None:
        try:
            eq_id = equality_id(self.env.model, self.env.mujoco, name)
        except ValueError:
            return
        self.env.data.eq_active[eq_id] = 1 if active else 0
        self.env.mujoco.mj_forward(self.env.model, self.env.data)

    def _record_site_target_error(self, phase: str, site_name: str, target_pos: Any) -> None:
        target = np.asarray(target_pos, dtype=np.float64)
        actual, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        error = target - actual
        entry = {
            "phase": phase,
            "site": site_name,
            "target_pos": target.tolist(),
            "actual_site_pos": actual.tolist(),
            "target_minus_actual": error.tolist(),
            "norm": float(np.linalg.norm(error)),
        }
        self.execution_site_errors.append(entry)
        err_mm = error * 1000.0
        print(
            "[site_error] "
            f"{phase} site={site_name} "
            f"target-actual(mm)=[{err_mm[0]:+.2f}, {err_mm[1]:+.2f}, {err_mm[2]:+.2f}] "
            f"norm={entry['norm'] * 1000.0:.2f}mm"
        )

    def _make_metadata(
        self,
        grasp_plan: list[dict[str, Any]],
        lift_plan: list[dict[str, Any]],
        mount_down_plan: list[dict[str, Any]],
        tip_retract_plan: list[dict[str, Any]],
        tube_plan: list[dict[str, Any]],
        *,
        tip_attachment: dict[str, np.ndarray] | None,
        num_steps: int,
    ) -> dict[str, Any]:
        final_obs = self.env.get_observation()
        pipette_pos, pipette_quat = free_joint_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_joint,
        )
        final_tip_pos, final_tip_quat = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_tip_site,
        )
        return {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "task": self.name,
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "arm": self.runtime.arm,
            "pipette_joint": self.runtime.pipette_joint,
            "local_grasp_pose": {
                "body": self.runtime.pipette_body,
                "pos": list(self.runtime.handle_grasp_offset),
                "euler_xyz": list(self.runtime.handle_grasp_euler),
                "grasp_to_gripper_pos": list(self.runtime.grasp_to_gripper_offset),
                "grasp_to_gripper_euler_xyz": list(self.runtime.grasp_to_gripper_euler),
            },
            "reset_info": self.env.last_reset_info,
            "slot_index": self.env.last_reset_info.get("random_single_free_joint", {}).get("slot_index"),
            "slot_name": self.env.last_reset_info.get("random_single_free_joint", {}).get("slot_name"),
            "grasp_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in grasp_plan],
            "lift_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in lift_plan],
            "tip_mount_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in mount_down_plan],
            "tip_retract_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in tip_retract_plan],
            "tube_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in tube_plan],
            "tip_target": json_safe(self.tip_target_info),
            "tip_attachment": json_safe(tip_attachment),
            "tube_target": json_safe(self.tube_target_info),
            "tip_mount": {
                "tip_mount_offset": list(self.runtime.tip_mount_offset),
                "tip_mount_target_euler_xyz": list(self.runtime.tip_mount_target_euler),
                "tip_site_prefix": self.runtime.tip_site_prefix,
                "tip_mount_site_suffix": self.runtime.tip_mount_site_suffix,
                "tip_end_site_suffix": self.runtime.tip_end_site_suffix,
                "tip_pose_servo_enabled": self.runtime.tip_pose_servo_enabled,
                "tube_joint": self.runtime.tube_joint,
                "tube_top_offset": self.runtime.tube_top_offset,
                "tube_hover_height": self.runtime.tube_hover_height,
                "tube_near_height": self.runtime.tube_near_height,
                "tube_target_offset": list(self.runtime.tube_target_offset),
            },
            "visual_servo": {
                "enabled": self.runtime.visual_servo_enabled,
                "max_iters": self.runtime.visual_servo_max_iters,
                "steps": self.runtime.visual_servo_steps,
                "pos_tol": self.runtime.visual_servo_pos_tol,
                "rot_tol": self.runtime.visual_servo_rot_tol,
                "gain": self.runtime.visual_servo_gain,
                "integral_gain": self.runtime.visual_servo_integral_gain,
                "max_correction": self.runtime.visual_servo_max_correction,
                "events": json_safe(self.visual_servo_events),
            },
            "execution_site_errors": json_safe(self.execution_site_errors),
            "final_time": float(final_obs["time"]),
            "final_state_summary": {
                "pipette_pos": pipette_pos.tolist(),
                "pipette_quat": pipette_quat.tolist(),
                "pipette_tip_site_pos": final_tip_pos.tolist(),
                "pipette_tip_site_quat": final_tip_quat.tolist(),
            },
        }
