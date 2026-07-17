"""TaskTarget construction for the ADP tip-to-tube workflow. Fully independent of pipette_targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...math3d import normalize_quat, quat_to_mat
from ...planner import TaskTargetPlanner
from ...scene import free_joint_pose, site_pose
from ...task_target import FrameRef, GripperCommand, TaskTarget
from .adp_scene import AdpSceneQuery


# ---------- Configuration dataclasses ----------
@dataclass(frozen=True)
class AdpPipetteModelConfig:
    pipette_tip_site: str = "piptip_site"
    pipette_mount_site: str = "adp_tip_mount_site"
    pipette_joint: str = "pipette_joint"
    pipette_body: str = "pipette_free"


@dataclass(frozen=True)
class AdpTipTargetConfig:
    tip_joint_prefix: str = "pipette_tip_joint_"
    tip_site_prefix: str = "tip"
    tip_mount_site_suffix: str = "mount_site"
    tip_end_site_suffix: str = "tip_end_site"
    tip_pose_servo_enabled: bool = True
    tip_hover_height: float = 0.020
    tip_retract_height: float = 0.100
    tip_length: float = 0.035
    # Offset from the selected tip mount site to the ADP piptip_site in the
    # tip mount-site frame. The tip mount_site is about 4 mm above the visible
    # opening, so -8 mm places the ADP protrusion about 4 mm into the tip.
    tip_mount_offset: tuple[float, float, float] = (0.0, 0.0, -0.008)
    tip_mount_axis_step: float = 0.002
    tip_attach_xy_tolerance: float = 0.0015
    tip_attach_depth_tolerance: float = 0.0015
    tip_axis_xy_tolerance: float = 0.0015


@dataclass(frozen=True)
class AdpTubeTargetConfig:
    tube_joint: str = "centrifuge_50ml_screw_joint_1"
    tube_top_offset: float = 0.115
    tube_hover_height: float = 0.10
    tube_near_height: float = 0.010
    tube_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AdpTrashConfig:
    pass 


# ---------- Target builder ----------
class AdpTargetBuilder:
    """Construct TaskTargets for the ADP workflow: tip mounting, tube hover, tube near, trash, home."""

    def __init__(
        self,
        env: Any,
        planner: TaskTargetPlanner,
        scene: AdpSceneQuery,
        arm_defaults: dict[str, dict[str, Any]],
        *,
        arm: str,
        close_steps: int,
        pipette: AdpPipetteModelConfig,
        tips: AdpTipTargetConfig,
        tube: AdpTubeTargetConfig,
        trash: AdpTrashConfig,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.mujoco = env.mujoco
        self.planner = planner
        self.scene = scene
        self.arm_defaults = arm_defaults
        self.arm = arm
        self.close_steps = close_steps
        self.pipette = pipette
        self.tips = tips
        self.tube = tube
        self.trash = trash

        self.tip_target_info: dict[str, Any] | None = None
        self.tube_target_info: dict[str, Any] | None = None
        self.mounted_piptip_from_tip_end_offset: np.ndarray | None = None

        self.home_pos: np.ndarray | None = None
        self.home_quat: np.ndarray | None = None

        # 固定姿态（竖直），由外部设置
        self.fixed_quat: np.ndarray | None = None

    def set_home_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self.home_pos = pos.copy()
        self.home_quat = quat.copy()

    def set_fixed_quat(self, quat: np.ndarray) -> None:
        self.fixed_quat = normalize_quat(quat.copy())

    def set_mounted_tip_end_offset(
        self,
        piptip_pos: np.ndarray,
        tip_end_pos: np.ndarray,
        *,
        tip_end_site: str | None = None,
    ) -> None:
        offset = np.asarray(piptip_pos, dtype=np.float64) - np.asarray(
            tip_end_pos,
            dtype=np.float64,
        )
        self.mounted_piptip_from_tip_end_offset = offset
        if self.tip_target_info is not None:
            self.tip_target_info["mounted_tip_end_site"] = tip_end_site
            self.tip_target_info["mounted_piptip_pos"] = np.asarray(
                piptip_pos,
                dtype=np.float64,
            ).tolist()
            self.tip_target_info["mounted_tip_end_pos"] = np.asarray(
                tip_end_pos,
                dtype=np.float64,
            ).tolist()
            self.tip_target_info["mounted_piptip_from_tip_end_offset"] = (
                offset.tolist()
            )

    def _ensure_fixed_quat(self) -> None:
        if self.fixed_quat is None:
            raise RuntimeError("Fixed quaternion not set; call set_fixed_quat() first.")

    def _piptip_target_for_tip_end(self, tip_end_target_pos: np.ndarray) -> np.ndarray:
        if self.mounted_piptip_from_tip_end_offset is None:
            return np.asarray(tip_end_target_pos, dtype=np.float64)
        return (
            np.asarray(tip_end_target_pos, dtype=np.float64)
            + self.mounted_piptip_from_tip_end_offset
        )

    # ---------- Tip mounting targets ----------
    def tip_hover_targets(self, arm_name: str) -> tuple[TaskTarget, ...]:
        """Move to the selected tip hover pose while keeping the held ADP attitude."""
        self._ensure_fixed_quat()

        target_tip = self.scene.nearest_active_tip()
        tip_joint = target_tip["joint"]
        tip_pos, _ = free_joint_pose(self.model, self.data, self.mujoco, tip_joint)
        mount_site = target_tip.get("mount_site")
        mount_pos = np.asarray(target_tip["mount_pos"], dtype=np.float64)
        mount_quat = normalize_quat(
            np.asarray(target_tip["mount_quat"], dtype=np.float64)
        )
        mount_mat = quat_to_mat(mount_quat)
        mount_offset = np.asarray(
            self.tips.tip_mount_offset,
            dtype=np.float64,
        )
        hover_offset = np.asarray(
            [0.0, 0.0, self.tips.tip_hover_height],
            dtype=np.float64,
        )
        mount_target_pos = mount_pos + mount_mat @ mount_offset
        target_hover_pos = mount_pos + mount_mat @ hover_offset
        if mount_site:
            target_parent = FrameRef("site", str(mount_site))
            hover_relative_pos = tuple(hover_offset.tolist())
            mount_relative_pos = tuple(mount_offset.tolist())
            fixed_target_quat = self.planner.relative_quat_for_world_quat(
                target_parent,
                self.fixed_quat,
            )
        else:
            target_parent = FrameRef("world")
            hover_relative_pos = tuple(target_hover_pos.tolist())
            mount_relative_pos = tuple(mount_target_pos.tolist())
            fixed_target_quat = self.fixed_quat

        self.tip_target_info = {
            "tip_joint": tip_joint,
            "tip_slot_name": target_tip.get("slot_name"),
            "tip_pos": tip_pos.tolist(),
            "tip_mount_site": mount_site,
            "tip_mount_pos": mount_pos.tolist(),
            "tip_mount_quat": mount_quat.tolist(),
            "tip_end_site": target_tip.get("end_site"),
            "tip_end_pos": target_tip["end_pos"].tolist(),
            "tip_end_quat": target_tip["end_quat"].tolist(),
            "tip_mount_offset": list(self.tips.tip_mount_offset),
            "tip_mount_target_pos": mount_target_pos.tolist(),
            "tip_mount_target_quat": self.fixed_quat.tolist(),
            "tip_mount_target_parent": {
                "kind": target_parent.kind,
                "name": target_parent.name,
            },
            "tip_mount_target_relative_pos": list(mount_relative_pos),
            "tip_mount_target_relative_quat": fixed_target_quat.tolist(),
            "target_tip_hover_pos": target_hover_pos.tolist(),
            "target_tip_hover_parent": {
                "kind": target_parent.kind,
                "name": target_parent.name,
            },
            "target_tip_hover_relative_pos": list(hover_relative_pos),
            "target_tip_hover_relative_quat": fixed_target_quat.tolist(),
            "tip_xy_distance": target_tip["xy_distance"],
        }

        target = TaskTarget(
            name="adp_tip_hover",
            parent=target_parent,
            pos=hover_relative_pos,
            quat_wxyz=tuple(fixed_target_quat.tolist()),
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode="pose",
            gripper=self._closed_during(),
        )
        return (target,)

    def tip_mount_down_targets(self, arm_name: str) -> list[TaskTarget]:
        self._ensure_fixed_quat()
        if self.tip_target_info is None:
            raise RuntimeError("Tip target not selected; call tip_hover_targets first")
        parent_info = self.tip_target_info.get(
            "tip_mount_target_parent",
            {"kind": "world", "name": None},
        )
        target_parent = FrameRef(
            str(parent_info.get("kind", "world")),
            parent_info.get("name"),
        )
        final_pos = np.asarray(
            self.tip_target_info.get(
                "tip_mount_target_relative_pos",
                self.tip_target_info["tip_mount_target_pos"],
            ),
            dtype=np.float64,
        )
        target_quat = np.asarray(
            self.tip_target_info.get(
                "tip_mount_target_relative_quat",
                self.fixed_quat,
            ),
            dtype=np.float64,
        )

        hover_pos = np.asarray(
            self.tip_target_info.get(
                "target_tip_hover_relative_pos",
                [final_pos[0], final_pos[1], self.tips.tip_hover_height],
            ),
            dtype=np.float64,
        )
        z_start = float(hover_pos[2])
        z_final = float(final_pos[2])
        step = max(0.001, float(self.tips.tip_mount_axis_step))
        z_values: list[float] = []
        z = z_start - step
        while z > z_final + step * 0.5:
            z_values.append(z)
            z -= step
        z_values.append(z_final)

        targets: list[TaskTarget] = []
        for index, z_value in enumerate(z_values):
            pos = final_pos.copy()
            pos[2] = z_value
            is_final = index == len(z_values) - 1
            targets.append(
                TaskTarget(
                    name=(
                        "adp_tip_mount_down"
                        if is_final
                        else f"adp_tip_mount_axis_{index + 1:02d}"
                    ),
                    parent=target_parent,
                    pos=tuple(pos.tolist()),
                    quat_wxyz=tuple(target_quat.tolist()),
                    arm=arm_name,
                    controlled_site=self.pipette.pipette_tip_site,
                    servo_mode="pose",
                    gripper=self._closed_during(),
                )
            )
        return targets

    def tip_retract_targets(self, arm_name: str) -> list[TaskTarget]:
        self._ensure_fixed_quat()
        if self.tip_target_info is None:
            raise RuntimeError("Tip target not selected; call tip_hover_targets first")
        mount_pos = np.asarray(self.tip_target_info["tip_mount_target_pos"], dtype=np.float64)
        target_pos = mount_pos + np.asarray(
            [0.0, 0.0, self.tips.tip_retract_height],
            dtype=np.float64,
        )
        target_quat = self.fixed_quat
        return [
            TaskTarget(
                name="adp_tip_retract",
                parent=FrameRef("world"),
                pos=tuple(target_pos.tolist()),
                quat_wxyz=tuple(target_quat.tolist()),
                arm=arm_name,
                controlled_site=self.pipette.pipette_tip_site,
                servo_mode="pose",
                gripper=self._closed_during(),
            )
        ]

    # ---------- Tube targets (world coordinates) ----------
    def tube_hover_targets(self, arm_name: str) -> tuple[TaskTarget, ...]:
        return (self._tube_target(arm_name, "adp_tube_hover", self.tube.tube_hover_height),)

    def tube_near_targets(self, arm_name: str) -> tuple[TaskTarget, ...]:
        return (self._tube_target(arm_name, "adp_tube_probe", self.tube.tube_near_height),)

    def tube_return_hover_targets(self, arm_name: str) -> tuple[TaskTarget, ...]:
        return (self._tube_target(arm_name, "adp_tube_return_hover", self.tube.tube_hover_height),)

    def _tube_target(
        self,
        arm_name: str,
        name: str,
        height: float,
    ) -> TaskTarget:
        self._ensure_fixed_quat()
        active_tube_joint = self.scene.active_tube_joint()
        tube_pos, _ = free_joint_pose(
            self.model, self.data, self.mujoco, active_tube_joint
        )
        tube_target_base = tube_pos + np.asarray(
            self.tube.tube_target_offset,
            dtype=np.float64,
        )
        tube_top = tube_target_base + np.asarray([0.0, 0.0, self.tube.tube_top_offset], dtype=np.float64)
        tip_end_target_pos = tube_top + np.asarray([0.0, 0.0, height], dtype=np.float64)
        target_pos = self._piptip_target_for_tip_end(tip_end_target_pos)

        previous_info = {} if self.tube_target_info is None else self.tube_target_info
        previous_near = previous_info.get("tube_near_tip_site_pos")
        previous_near_tip_end = previous_info.get("tube_near_tip_end_pos")
        self.tube_target_info = {
            "tube_joint": active_tube_joint,
            "tube_pos": tube_pos.tolist(),
            "tube_target_offset": list(self.tube.tube_target_offset),
            "tube_target_base_pos": tube_target_base.tolist(),
            "tube_top_pos": tube_top.tolist(),
            "controlled_site": self.pipette.pipette_tip_site,
            "mounted_piptip_from_tip_end_offset": (
                None
                if self.mounted_piptip_from_tip_end_offset is None
                else self.mounted_piptip_from_tip_end_offset.tolist()
            ),
            "tube_hover_tip_site_pos": (
                target_pos.tolist()
                if height == self.tube.tube_hover_height
                else previous_info.get("tube_hover_tip_site_pos")
            ),
            "tube_hover_tip_end_pos": (
                tip_end_target_pos.tolist()
                if height == self.tube.tube_hover_height
                else previous_info.get("tube_hover_tip_end_pos")
            ),
            "tube_near_tip_site_pos": (
                target_pos.tolist()
                if height == self.tube.tube_near_height
                else previous_near
            ),
            "tube_near_tip_end_pos": (
                tip_end_target_pos.tolist()
                if height == self.tube.tube_near_height
                else previous_near_tip_end
            ),
        }

        return TaskTarget(
            name=name,
            parent=FrameRef("world"),
            pos=tuple(target_pos.tolist()),
            quat_wxyz=tuple(self.fixed_quat.tolist()),
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode="pose",
            gripper=self._closed_during(),
        )

    # ---------- Other targets ----------
    def trash_target(self, arm_name: str) -> TaskTarget:
        """从场景中动态获取垃圾桶位置，移液器尖端悬停于桶口上方"""
        self._ensure_fixed_quat()
        tip_end_target_pos = self.scene.trash_position()
        target_pos = self._piptip_target_for_tip_end(tip_end_target_pos)
        return TaskTarget(
            name="adp_trash_hover",
            parent=FrameRef("world"),
            pos=tuple(target_pos.tolist()),
            quat_wxyz=tuple(self.fixed_quat.tolist()),
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode="pose",
            gripper=self._closed_during(),
        )

    def home_target(self, arm_name: str) -> TaskTarget:
        if self.home_pos is None or self.home_quat is None:
            raise RuntimeError("Home pose not set; call set_home_pose() first.")
        gripper_site = self._gripper_site(arm_name)
        return TaskTarget(
            name="adp_home",
            parent=FrameRef("world"),
            pos=tuple(self.home_pos.tolist()),
            quat_wxyz=tuple(self.home_quat.tolist()),
            arm=arm_name,
            controlled_site=gripper_site,
            servo_mode="pose",
            gripper=self._closed_during(),
        )

    def release_tip_up_target(self, arm_name: str, base_pos: np.ndarray) -> TaskTarget:
        self._ensure_fixed_quat()
        pos = base_pos + np.asarray([0.0, 0.0, 0.06], dtype=np.float64)  # 上升 6cm 确保完全脱离
        return TaskTarget(
            name="adp_release_up",
            parent=FrameRef("world"),
            pos=tuple(pos.tolist()),
            quat_wxyz=tuple(self.fixed_quat.tolist()),
            arm=arm_name,
            controlled_site=self.pipette.pipette_tip_site,
            servo_mode="pose",
            gripper=self._closed_during(),
        )

    def target_tip_joint(self) -> str:
        if self.tip_target_info is None:
            raise RuntimeError("Tip target not selected; call tip_hover_targets first")
        return str(self.tip_target_info["tip_joint"])

    # ---------- Helpers ----------
    def _gripper_site(self, arm_name: str) -> str:
        return str(self.arm_defaults[arm_name]["gripper_site"])

    def _closed_during(self) -> GripperCommand:
        return GripperCommand(255.0, timing="during", steps=self.close_steps)
