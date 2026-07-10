"""TaskTarget construction for the pipette workflow.

This module describes where the robot or held tool should move. It does not
solve IK, execute trajectories, manage attachments, or write episode metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..math3d import euler_xyz_to_mat, mat_to_quat, normalize_quat, quat_to_mat
from ..planner import TaskTargetPlanner, site_quat_for_joint_quat
from ..scene import free_joint_pose, site_pose
from ..task_target import FrameRef, GripperCommand, PoseOffset, TaskTarget
from .pipette_scene import PipetteSceneQuery


@dataclass(frozen=True)
class PipetteHandleGraspConfig:
    pregrasp_distance: float = 0.08
    handle_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.15)
    handle_grasp_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_to_gripper_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_to_gripper_euler: tuple[float, float, float] = (
        float(np.pi),
        0.0,
        0.0,
    )
    middle_grasp_arm: str = "second"
    middle_pregrasp_distance: float = 0.04
    middle_grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.12)
    middle_grasp_euler: tuple[float, float, float] = (
        -float(np.pi / 2),
        float(np.pi / 2),
        float(np.pi / 2),
    )
    middle_grasp_to_gripper_offset: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    middle_grasp_to_gripper_euler: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    first_retreat_after_handoff_offset: tuple[float, float, float] = (
        0.0,
        0.0,
        0.10,
    )


@dataclass(frozen=True)
class PipetteModelConfig:
    pipette_tip_site: str = "piptip_site"
    pipette_joint: str = "pipette_joint"
    pipette_body: str = "pippipette"
    parking_weld: str = "pipette_rack_weld"


@dataclass(frozen=True)
class PipetteTipTargetConfig:
    tip_joint_prefix: str = "pipette_tip_joint_"
    tip_site_prefix: str = "tip"
    tip_mount_site_suffix: str = "mount_site"
    tip_end_site_suffix: str = "tip_end_site"
    tip_pose_servo_enabled: bool = True
    tip_hover_height: float = 0.03
    tip_mount_offset: tuple[float, float, float] = (0.0, 0.0, -0.06)
    tip_mount_target_euler: tuple[float, float, float] = (
        0.0,
        0.0,
        float(np.pi),
    )
    vertical_quat: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )


@dataclass(frozen=True)
class PipetteTubeTargetConfig:
    tube_joint: str = "centrifuge_50mltube free joint 原点的_screw_joint_1"
    tube_top_offset: float = 0.103
    tube_hover_height: float = 0.10
    tube_near_height: float = 0.03
    tube_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TipHoverTargets:
    targets: tuple[TaskTarget, ...]
    aligned_joint_quat: np.ndarray

'''负责构造：
    抓取目标；
    预抓取目标；
    枪头 hover 目标；
    装枪头目标；
    离心管 hover/near 目标。
'''
class PipetteTargetBuilder:
    """构造移液枪任务中的 TaskTarget。

        该对象带有少量任务状态：

        - tip_hover_targets() 会选择并缓存当前目标枪头；
        - tube_hover_targets()/tube_near_targets() 会缓存当前离心管目标；
        - 后续装枪头和 metadata 生成依赖这些缓存信息。

        因此，一个 episode 应使用同一个 PipetteTargetBuilder 实例。
    """
    def __init__(
        self,
        env: Any,
        planner: TaskTargetPlanner,
        scene: PipetteSceneQuery,
        arm_defaults: dict[str, dict[str, Any]],
        *,
        primary_arm: str,
        close_steps: int,
        grasp: PipetteHandleGraspConfig,
        pipette: PipetteModelConfig,
        tips: PipetteTipTargetConfig,
        tube: PipetteTubeTargetConfig,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.mujoco = env.mujoco
        self.planner = planner
        self.scene = scene
        self.arm_defaults = arm_defaults
        self.primary_arm = primary_arm
        self.close_steps = close_steps
        self.grasp = grasp
        self.pipette = pipette
        self.tips = tips
        self.tube = tube
        self.tip_target_info: dict[str, Any] | None = None
        self.tube_target_info: dict[str, Any] | None = None

    '''
    构造抓取目标, 预抓取目标, 以及抓取后撤退目标
    1. primary_grasp_targets: 构造 pipette 抓取目标和预抓取目标
    2. middle_grasp_targets: 构造 pipette 中间抓取目标和预抓取目标
    3. first_retreat_after_handoff_targets: 构造 pipette 抓取后撤退目标
    4. tip_hover_targets: 构造 pipette tip hover 目标
    5. tip_mount_down_targets: 构造 pipette tip 装载目标
    6. tip_retract_targets: 构造 pipette tip 撤退目标
    7. tube_hover_targets: 构造离心管 hover 目标
    8. tube_near_targets: 构造离心管 near 目标
    9. target_tip_info: 获取当前的 tip 目标信息
    10. target_tip_joint: 获取当前的 tip 关节名称
    11. gripper_site: 获取指定机械臂的 gripper site
    12. _tube_approach_targets: 构造离心管 hover 和 near 目标
    13. _tip_mount_target_pose: 计算 tip 装载目标的世界坐标和四元数
    14. _tip_mount_task_target: 构造 tip 装载任务目标
    15. _open_during: 构造 gripper 打开命令
    16. _closed_during: 构造 gripper 关闭命令       
    '''
    def primary_grasp_targets(self) -> list[TaskTarget]:
        """构造 first 机械臂抓取移液枪手柄的目标序列。

            位姿链：
                pipette body
                → handle grasp frame
                → first gripper site

            如果 pregrasp_distance > 0，则先生成沿夹爪局部
            approach 轴后退的预抓取点，再生成最终抓取点。
        """
        gripper_site = self.gripper_site(self.primary_arm)
        grasp_target = TaskTarget(
            name="pipette_grasp",
            parent=FrameRef("body", self.pipette.pipette_body),
            pos=self.grasp.handle_grasp_offset,
            euler=self.grasp.handle_grasp_euler,
            arm=self.primary_arm,
            controlled_site=gripper_site,
            servo_mode="pose",
            gripper=self._open_during(),
            target_offset=PoseOffset(
                pos=self.grasp.grasp_to_gripper_offset,
                euler=self.grasp.grasp_to_gripper_euler,
            ),
        )
        targets: list[TaskTarget] = []
        if self.grasp.pregrasp_distance > 0.0:
            targets.append(
                replace(
                    grasp_target.with_approach_offset(
                        -self.grasp.pregrasp_distance,
                        name="pipette_pregrasp",
                    ),
                    servo_mode="none",
                    gripper=self._open_during(),
                )
            )
        targets.append(grasp_target)
        return targets

    def middle_grasp_targets(self) -> list[TaskTarget]:
        arm_name = self.grasp.middle_grasp_arm
        target = TaskTarget(
            name="pipette_middle_grasp",
            parent=FrameRef("body", self.pipette.pipette_body),
            pos=self.grasp.middle_grasp_offset,
            euler=self.grasp.middle_grasp_euler,
            arm=arm_name,
            controlled_site=self.gripper_site(arm_name),
            servo_mode="none",
            gripper=GripperCommand(
                180.0,
                timing="after",
                steps=self.close_steps,
            ),
            target_offset=PoseOffset(
                pos=self.grasp.middle_grasp_to_gripper_offset,
                euler=self.grasp.middle_grasp_to_gripper_euler,
            ),
        )
        targets: list[TaskTarget] = []
        if self.grasp.middle_pregrasp_distance > 0.0:
            targets.append(
                replace(
                    target.with_approach_offset(
                        -self.grasp.middle_pregrasp_distance,
                        name="pipette_middle_pregrasp",
                    ),
                    servo_mode="none",
                    gripper=self._open_during(),
                )
            )
        targets.append(target)
        return targets

    def first_retreat_after_handoff_targets(self) -> list[TaskTarget]:
        gripper_site = self.gripper_site(self.primary_arm)
        gripper_pos, gripper_quat = site_pose(
            self.model,
            self.data,
            self.mujoco,
            gripper_site,
        )
        target_pos = gripper_pos + np.asarray(
            self.grasp.first_retreat_after_handoff_offset,
            dtype=np.float64,
        )
        return [
            TaskTarget(
                name="first_gripper_retreat_after_handoff",
                parent=FrameRef("world"),
                pos=tuple(target_pos.tolist()),
                quat_wxyz=tuple(gripper_quat.tolist()),
                arm=self.primary_arm,
                controlled_site=gripper_site,
                servo_mode="position",
                gripper=self._open_during(),
            )
        ]

    def tip_hover_targets(self, arm_name: str) -> TipHoverTargets:
        """构造移液枪移动到枪头上方的两阶段目标。

            阶段一：
                保持当前枪头姿态，只平移到目标枪头上方。

            阶段二：
                保持位置不变，将 pipette_tip_site 的姿态调整为
                枪头 mount site 的目标姿态。

            这样可以避免机械臂在长距离移动过程中同时进行大角度旋转。
        """
        aligned_joint_quat = normalize_quat(
            np.asarray(self.tips.vertical_quat, dtype=np.float64)
        )
        _, current_site_quat = site_pose(
            self.model,
            self.data,
            self.mujoco,
            self.pipette.pipette_tip_site,
        )
        aligned_site_quat = site_quat_for_joint_quat(
            self.model,
            self.data,
            self.mujoco,
            self.pipette.pipette_joint,
            self.pipette.pipette_tip_site,
            aligned_joint_quat,
        )
        target_tip = self.scene.nearest_active_tip()
        mount_target_pos, mount_target_quat = self._tip_mount_target_pose(
            target_tip
        )
        target_tip_hover_pos = mount_target_pos + np.asarray(
            [0.0, 0.0, self.tips.tip_hover_height],
            dtype=np.float64,
        )
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

        targets = (
            TaskTarget(
                name="pipette_tip_hover_translate",
                parent=FrameRef("world"),
                pos=tuple(target_tip_hover_pos.tolist()),
                quat_wxyz=tuple(current_site_quat.tolist()),
                arm=arm_name,
                controlled_site=self.pipette.pipette_tip_site,
                servo_mode="pose",
                gripper=self._closed_during(),
            ),
            TaskTarget(
                name="pipette_tip_hover_align",
                parent=FrameRef("world"),
                pos=tuple(target_tip_hover_pos.tolist()),
                quat_wxyz=tuple(
                    (
                        mount_target_quat
                        if self.tips.tip_pose_servo_enabled
                        else aligned_site_quat
                    ).tolist()
                ),
                arm=arm_name,
                controlled_site=self.pipette.pipette_tip_site,
                servo_mode=(
                    "pose"
                    if self.tips.tip_pose_servo_enabled
                    else "position"
                ),
                gripper=self._closed_during(),
            ),
        )
        return TipHoverTargets(targets, aligned_joint_quat)

    def tip_mount_down_targets(self, arm_name: str) -> list[TaskTarget]:
        target = self._tip_mount_task_target(
            "pipette_tip_mount_down",
            self.target_tip_info(),
            extra_offset=(0.0, 0.0, 0.0),
            servo_mode=(
                "pose" if self.tips.tip_pose_servo_enabled else "position"
            ),
            arm_name=arm_name,
        )
        return [target]

    def tip_retract_targets(self, arm_name: str) -> list[TaskTarget]:
        target = self._tip_mount_task_target(
            "pipette_tip_mounted_retract",
            self.target_tip_info(),
            extra_offset=(0.0, 0.0, self.tips.tip_hover_height),
            servo_mode=(
                "pose" if self.tips.tip_pose_servo_enabled else "position"
            ),
            arm_name=arm_name,
        )
        return [target]

    def tube_hover_targets(self, arm_name: str) -> list[TaskTarget]:
        hover_target, _ = self._tube_approach_targets(arm_name)
        return [hover_target]

    def tube_near_targets(self, arm_name: str) -> list[TaskTarget]:
        _, near_target = self._tube_approach_targets(arm_name)
        return [near_target]

    def target_tip_info(self) -> dict[str, Any]:
        if self.tip_target_info is None:
            raise RuntimeError(
                "Tip target is unavailable; build tip-hover targets first"
            )
        return self.tip_target_info

    def target_tip_joint(self) -> str:
        return str(self.target_tip_info()["tip_joint"])

    def gripper_site(self, arm_name: str) -> str:
        return str(self.arm_defaults[arm_name]["gripper_site"])

    def _tube_approach_targets(
        self,
        arm_name: str,
    ) -> tuple[TaskTarget, TaskTarget]:
        active_tube_joint = self.scene.active_tube_joint()
        tube_pos, _ = free_joint_pose(
            self.model,
            self.data,
            self.mujoco,
            active_tube_joint,
        )
        tube_parent = FrameRef("free_joint", active_tube_joint)
        _, current_site_quat = site_pose(
            self.model,
            self.data,
            self.mujoco,
            self.pipette.pipette_tip_site,
        )
        tube_relative_base = np.asarray(
            self.tube.tube_target_offset,
            dtype=np.float64,
        )
        hover_relative_pos = tuple(
            (
                tube_relative_base
                + np.asarray(
                    [
                        0.0,
                        0.0,
                        self.tube.tube_top_offset
                        + self.tube.tube_hover_height,
                    ]
                )
            ).tolist()
        )
        near_relative_pos = tuple(
            (
                tube_relative_base
                + np.asarray(
                    [
                        0.0,
                        0.0,
                        self.tube.tube_top_offset
                        + self.tube.tube_near_height,
                    ]
                )
            ).tolist()
        )
        relative_site_quat = tuple(
            self.planner.relative_quat_for_world_quat(
                tube_parent,
                current_site_quat,
            ).tolist()
        )
        hover_target = TaskTarget(
            name="pipette_tip_tube_hover",
            parent=tube_parent,
            pos=hover_relative_pos,
            quat_wxyz=relative_site_quat,
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode="pose",
            gripper=self._closed_during(),
        )
        near_target = TaskTarget(
            name="pipette_tip_tube_near",
            parent=tube_parent,
            pos=near_relative_pos,
            quat_wxyz=relative_site_quat,
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode="position",
            gripper=self._closed_during(),
        )
        hover_resolved = self.planner.resolve(hover_target)
        near_resolved = self.planner.resolve(near_target)
        tube_top = (
            tube_pos
            + tube_relative_base
            + np.asarray([0.0, 0.0, self.tube.tube_top_offset])
        )
        self.tube_target_info = {
            "tube_joint": active_tube_joint,
            "tube_pos": tube_pos.tolist(),
            "tube_top_pos": tube_top.tolist(),
            "tube_hover_tip_site_pos": hover_resolved.pos.tolist(),
            "tube_near_tip_site_pos": near_resolved.pos.tolist(),
        }
        return hover_target, near_target

    def _tip_mount_target_pose(
        self,
        target_tip: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        fallback_pos = target_tip.get("tip_pos", target_tip.get("pos"))
        if fallback_pos is None:
            raise KeyError(
                "target_tip must contain one of mount_pos, tip_pos, or pos"
            )
        mount_pos = np.asarray(
            target_tip.get("mount_pos", fallback_pos),
            dtype=np.float64,
        )
        mount_quat = normalize_quat(
            np.asarray(
                target_tip.get("mount_quat", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
        )
        mount_mat = quat_to_mat(mount_quat)
        relative_pos = np.asarray(
            self.tips.tip_mount_offset,
            dtype=np.float64,
        )
        relative_mat = euler_xyz_to_mat(
            np.asarray(
                self.tips.tip_mount_target_euler,
                dtype=np.float64,
            )
        )
        target_pos = mount_pos + mount_mat @ relative_pos
        target_quat = mat_to_quat(
            self.mujoco,
            mount_mat @ relative_mat,
        )
        return target_pos, target_quat

    def _tip_mount_task_target(
        self,
        name: str,
        target_tip: dict[str, Any],
        *,
        extra_offset: tuple[float, float, float],
        servo_mode: str,
        arm_name: str,
    ) -> TaskTarget:
        site_name = target_tip.get("tip_mount_site") or target_tip.get(
            "mount_site"
        )
        base_offset = np.asarray(
            self.tips.tip_mount_offset,
            dtype=np.float64,
        )
        pos = tuple(
            (
                base_offset
                + np.asarray(extra_offset, dtype=np.float64)
            ).tolist()
        )
        parent = (
            FrameRef("site", str(site_name))
            if site_name
            else FrameRef("world")
        )

        if not site_name:
            world_pos, world_quat = self._tip_mount_target_pose(target_tip)
            return TaskTarget(
                name=name,
                parent=FrameRef("world"),
                pos=tuple(
                    (
                        world_pos
                        + np.asarray(extra_offset, dtype=np.float64)
                    ).tolist()
                ),
                quat_wxyz=tuple(world_quat.tolist()),
                arm=arm_name,
                controlled_site=self.pipette.pipette_tip_site,
                servo_mode=servo_mode,
                gripper=self._closed_during(),
            )

        quat_wxyz = None
        if servo_mode != "pose":
            _, current_site_quat = site_pose(
                self.model,
                self.data,
                self.mujoco,
                self.pipette.pipette_tip_site,
            )
            quat_wxyz = tuple(
                self.planner.relative_quat_for_world_quat(
                    parent,
                    current_site_quat,
                ).tolist()
            )

        return TaskTarget(
            name=name,
            parent=parent,
            pos=pos,
            euler=self.tips.tip_mount_target_euler,
            quat_wxyz=quat_wxyz,
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode=servo_mode,
            gripper=self._closed_during(),
        )

    def _open_during(self) -> GripperCommand:
        return GripperCommand(
            0.0,
            timing="during",
            steps=self.close_steps,
        )

    def _closed_during(self) -> GripperCommand:
        return GripperCommand(
            255.0,
            timing="during",
            steps=self.close_steps,
        )
