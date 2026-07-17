"""ADP tip-to-tube task: mount a visible tip, visit tube hover, drop tip, and home."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...executor import TaskTargetExecutor
from ...math3d import quat_to_mat
from ...motion_context import (
    ExecutionContext,
    ExecutionSettings,
    FixedJointState,
    GripperSettings,
    IKSettings,
    KinematicBinding,
    PlanningContext,
    SiteAttachment,
    VisualServoSettings,
    arm_motion_configs,
)
from ...mujoco_env import EnvConfig
from ...planner import TaskTargetPlanner
from ...recorder import EpisodeRecorder
from ...scene import (
    actuator_id,
    capture_free_joint_state,
    capture_site_attachment,
    restore_free_joint_state,
    site_pose,
)
from ...task import AutoLabTask, TaskConfig
from ...task_target import PlannedTaskTarget, TaskTarget
from ..common import ARM_DEFAULTS
from .adp_scene import AdpSceneQuery
from .adp_targets import (
    AdpPipetteModelConfig,
    AdpTipTargetConfig,
    AdpTrashConfig,
    AdpTubeTargetConfig,
    AdpTargetBuilder,
)
from .adp_metadata import AdpMetadataBuilder


# ---------- Configuration dataclasses ----------
@dataclass(frozen=True)
class AdpTimingConfig:
    initial_static_steps: int = 20
    settle_steps: int = 8
    tool_stabilize_steps: int = 12
    steps_per_segment: int = 12
    tip_hover_steps: int = 36
    close_steps: int = 2
    tip_mount_settle_steps: int = 8
    hold_steps_5s: int = 60
    release_wait_steps: int = 8


@dataclass(frozen=True)
class AdpIKConfig:
    ik_max_iters: int = 400
    ik_pos_tol: float = 0.0008
    ik_rot_tol: float = 0.02
    ik_damping: float = 0.03


@dataclass(frozen=True)
class AdpWaypointSettleConfig:
    waypoint_settle_steps: int = 1
    waypoint_settle_pos_tol: float = 0.0008


@dataclass(frozen=True)
class AdpVisualServoConfig:
    visual_servo_enabled: bool = True
    visual_servo_max_iters: int = 14
    visual_servo_steps: int = 6
    visual_servo_pos_tol: float = 0.00015
    visual_servo_rot_tol: float = 0.03
    visual_servo_gain: float = 0.85
    visual_servo_integral_gain: float = 0.2
    visual_servo_max_correction: float = 0.012


@dataclass(frozen=True)
class AdpGripperSettings:
    open_value: float = 0.0
    close_value: float = 255.0


# ---------- Helper classes ----------
@dataclass(frozen=True)
class ToolAttachment:
    arm: str
    attachment: SiteAttachment

    @property
    def gripper_site(self) -> str:
        return self.attachment.parent_site

    def follow(self) -> tuple[SiteAttachment, ...]:
        return (self.attachment,)

    def planning_context(self, controlled_site: str) -> PlanningContext:
        return PlanningContext(
            (
                KinematicBinding(
                    arm=self.arm,
                    actuator_site=self.gripper_site,
                    controlled_site=controlled_site,
                    attachments=(self.attachment,),
                ),
            )
        )


@dataclass(frozen=True)
class TipMountResult:
    mount_down_plan: list[PlannedTaskTarget]
    tip_joint: str
    tip_attachment: SiteAttachment


class _DiscardingRecorder:
    def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# ---------- Main task ----------
@dataclass(frozen=True)
class AdpTipToTubeTaskConfig:
    env: EnvConfig
    out_dir: Path
    episode_index: int
    seed: int
    cameras: tuple[str, ...] = ("overview_camera", "wrist_cam", "wrist_cam1")
    with_images: bool = False

    arm: str = "second"
    timing: AdpTimingConfig = field(default_factory=AdpTimingConfig)
    ik: AdpIKConfig = field(default_factory=AdpIKConfig)
    waypoint: AdpWaypointSettleConfig = field(default_factory=AdpWaypointSettleConfig)
    visual_servo: AdpVisualServoConfig = field(default_factory=AdpVisualServoConfig)
    gripper: AdpGripperSettings = field(default_factory=AdpGripperSettings)

    pipette: AdpPipetteModelConfig = field(default_factory=AdpPipetteModelConfig)
    tips: AdpTipTargetConfig = field(default_factory=AdpTipTargetConfig)
    tube: AdpTubeTargetConfig = field(default_factory=AdpTubeTargetConfig)
    trash: AdpTrashConfig = field(default_factory=AdpTrashConfig)


class AdpTipToTubeTask(AutoLabTask):
    name = "adp_tip_to_tube"

    def __init__(self, config: AdpTipToTubeTaskConfig) -> None:
        self.runtime = config
        super().__init__(
            TaskConfig(
                env=config.env,
                with_images=config.with_images,
                cameras=config.cameras,
            )
        )

        self.arm_configs = arm_motion_configs(ARM_DEFAULTS)
        self.ik_settings = IKSettings(
            max_iters=config.ik.ik_max_iters,
            pos_tol=config.ik.ik_pos_tol,
            rot_tol=config.ik.ik_rot_tol,
            damping=config.ik.ik_damping,
        )
        self.gripper_settings = GripperSettings(
            open_value=config.gripper.open_value,
            close_value=config.gripper.close_value,
        )

        self.planner = TaskTargetPlanner(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.arm_configs,
            self.ik_settings,
            self.gripper_settings,
        )
        self.executor = TaskTargetExecutor(
            self.env,
            self.manager,
            self.arm_configs,
            self.ik_settings,
            self.gripper_settings,
            ExecutionSettings(
                steps_per_segment=config.timing.steps_per_segment,
                waypoint_settle_steps=config.waypoint.waypoint_settle_steps,
                waypoint_settle_pos_tol=config.waypoint.waypoint_settle_pos_tol,
                visual_servo=VisualServoSettings(
                    enabled=config.visual_servo.visual_servo_enabled,
                    max_iters=config.visual_servo.visual_servo_max_iters,
                    steps=config.visual_servo.visual_servo_steps,
                    pos_tol=config.visual_servo.visual_servo_pos_tol,
                    rot_tol=config.visual_servo.visual_servo_rot_tol,
                    gain=config.visual_servo.visual_servo_gain,
                    integral_gain=config.visual_servo.visual_servo_integral_gain,
                    max_correction=config.visual_servo.visual_servo_max_correction,
                ),
            ),
        )

        self.scene = AdpSceneQuery(
            self.env,
            pipette_tip_site=config.pipette.pipette_tip_site,
            tip_joint_prefix=config.tips.tip_joint_prefix,
            tip_site_prefix=config.tips.tip_site_prefix,
            tip_mount_site_suffix=config.tips.tip_mount_site_suffix,
            tip_end_site_suffix=config.tips.tip_end_site_suffix,
            fallback_tube_joint=config.tube.tube_joint,
        )

        self.target_builder = AdpTargetBuilder(
            self.env,
            self.planner,
            self.scene,
            ARM_DEFAULTS,
            arm=config.arm,
            close_steps=config.timing.close_steps,
            pipette=config.pipette,
            tips=config.tips,
            tube=config.tube,
            trash=config.trash,
        )
        self.metadata_builder = AdpMetadataBuilder(self.env, self.runtime)

        self.execution_site_errors = self.executor.execution_site_errors
        self.visual_servo_events = self.executor.visual_servo_events

    def run(self) -> dict[str, Any]:
        obs = self.reset()
        action = np.asarray(obs["ctrl"], dtype=np.float64).copy()
        action[self._gripper_id(self.runtime.arm)] = self.runtime.gripper.close_value

        gripper_site = self._gripper_site(self.runtime.arm)
        init_pos, init_quat = site_pose(
            self.env.model, self.env.data, self.env.mujoco, gripper_site
        )
        self.target_builder.set_home_pose(init_pos, init_quat)

        recorder = EpisodeRecorder(
            cameras=self.runtime.cameras,
            with_images=self.runtime.with_images,
        )

        self._record_kinematic_hold(
            recorder,
            action,
            self.runtime.timing.initial_static_steps,
            "initial_static_hold",
            ExecutionContext(),
        )

        # 1. Keep the reset pose visible without advancing dynamics. A dynamic
        # hold here creates a small gravity/controller sag in both arms that
        # shows up as a sudden downward posture at the beginning of the video.
        self._record_kinematic_hold(
            recorder,
            action,
            self.runtime.timing.settle_steps,
            "settle",
            ExecutionContext(),
        )

        _, tip_quat = site_pose(
            self.env.model, self.env.data, self.env.mujoco,
            self.runtime.pipette.pipette_tip_site
        )
        self.target_builder.set_fixed_quat(tip_quat)

        tool_holder = self._capture_tool_attachment(self.runtime.arm)
        self._record_kinematic_hold(
            recorder,
            action,
            self.runtime.timing.tool_stabilize_steps,
            "tool_stabilize",
            self._execution_context(attachments=tool_holder.follow()),
        )

        # 3. Tip hover
        tip_hover_targets = self.target_builder.tip_hover_targets(self.runtime.arm)
        target_tip_joint = self.target_builder.target_tip_joint()
        target_tip_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            target_tip_joint,
        )
        plan_hover = self._plan_targets(
            tip_hover_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        for item in plan_hover:
            item.steps = max(
                int(self.runtime.timing.tip_hover_steps),
                int(self.runtime.timing.steps_per_segment),
            )
        self.executor.execute(
            recorder,
            plan_hover,
            "tip_hover",
            self._execution_context(
                fixed_joint_states=((target_tip_joint, target_tip_state),),
                attachments=tool_holder.follow(),
            ),
        )
        self._validate_tip_hover_alignment()

        # 4. Tip mount (down)
        mount_targets = self.target_builder.tip_mount_down_targets(self.runtime.arm)
        plan_mount = self._plan_targets(
            mount_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        for item in plan_mount:
            item.steps = max(1, int(self.runtime.timing.steps_per_segment))
        mount_context = self._execution_context(
            fixed_joint_states=((target_tip_joint, target_tip_state),),
            attachments=tool_holder.follow(),
        )
        if self.runtime.visual_servo.visual_servo_enabled:
            self._execute_tip_mount_axis_servo(
                recorder,
                plan_mount,
                mount_context,
            )
        else:
            self.executor.execute(
                recorder,
                plan_mount,
                "tip_mount",
                mount_context,
            )
        self.executor.hold_action(
            recorder,
            np.asarray(plan_mount[-1].action, dtype=np.float64),
            self.runtime.timing.tip_mount_settle_steps,
            "tip_mount_settle",
            mount_context,
        )
        self._validate_tip_mount_alignment()
        tip_mount = self._capture_tip_attachment(tool_holder)

        # 5. Tip retract (up)
        retract_targets = self.target_builder.tip_retract_targets(self.runtime.arm)
        plan_retract = self._plan_targets(
            retract_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        self.executor.execute(
            recorder,
            plan_retract,
            "tip_retract",
            self._execution_context(
                attachments=tool_holder.follow() + (tip_mount.tip_attachment,)
            ),
        )
        self._update_mounted_tip_end_offset()

        # 6. Move to tube hover
        tube_hover_targets = self.target_builder.tube_hover_targets(self.runtime.arm)
        plan_tube_hover = self._plan_targets(
            tube_hover_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        self.executor.execute(
            recorder,
            plan_tube_hover,
            "tube_hover",
            self._execution_context(
                attachments=tool_holder.follow() + (tip_mount.tip_attachment,)
            ),
        )

        # 7. Hold above the tube for 3 seconds, probe into the tube, then lift
        # back to the same hover height before moving to trash.
        hold_action = np.asarray(plan_tube_hover[-1].action, dtype=np.float64).copy()
        hold_steps = self.runtime.timing.hold_steps_5s
        self.executor.hold_action(
            recorder,
            hold_action,
            hold_steps,
            "hold_tube_3s",
            self._execution_context(
                attachments=tool_holder.follow() + (tip_mount.tip_attachment,)
            ),
        )

        tube_near_targets = self.target_builder.tube_near_targets(self.runtime.arm)
        plan_tube_near = self._plan_targets(
            tube_near_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        mounted_tip_ctx = self._execution_context(
            attachments=tool_holder.follow() + (tip_mount.tip_attachment,)
        )
        self.executor.execute(
            recorder,
            plan_tube_near,
            "tube_probe",
            mounted_tip_ctx,
        )

        tube_return_targets = self.target_builder.tube_return_hover_targets(
            self.runtime.arm
        )
        plan_tube_return = self._plan_targets(
            tube_return_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        self.executor.execute(
            recorder,
            plan_tube_return,
            "tube_return_hover",
            mounted_tip_ctx,
        )

        # 8. Move to trash
        trash_target = self.target_builder.trash_target(self.runtime.arm)
        plan_trash = self._plan_targets(
            (trash_target,),
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        self.executor.execute(
            recorder,
            plan_trash,
            "move_to_trash",
            self._execution_context(
                attachments=tool_holder.follow() + (tip_mount.tip_attachment,)
            ),
        )

        # 10. Release tip directly over trash.
        release_tip_qpos, release_tip_qvel = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            tip_mount.tip_joint,
        )
        release_tip_qpos = release_tip_qpos.copy()
        release_tip_state = (
            release_tip_qpos,
            np.zeros_like(release_tip_qvel),
        )
        restore_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            tip_mount.tip_joint,
            release_tip_state,
        )
        plan_release: list[PlannedTaskTarget] = []
        no_tip_ctx = self._execution_context(attachments=tool_holder.follow())
        # 等待更长时间确保吸头受重力下落
        self.executor.hold_action(
            recorder,
            np.asarray(plan_trash[-1].action, dtype=np.float64),
            self.runtime.timing.release_wait_steps,
            "wait_tip_drop",
            no_tip_ctx,
        )

        # 11. Return home
        home_target = self.target_builder.home_target(self.runtime.arm)
        plan_home = self._plan_targets(
            (home_target,),
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        self.executor.execute(
            recorder,
            plan_home,
            "return_home",
            no_tip_ctx,
        )

        arrays = self._prepare_lerobot_arrays(recorder.to_arrays())
        metadata = self.metadata_builder.build(
            tip_hover_targets=tuple(plan_hover),
            mount_plan=plan_mount,
            retract_plan=plan_retract,
            tube_hover_plan=plan_tube_hover,
            tube_near_plan=plan_tube_near,
            tube_return_plan=plan_tube_return,
            trash_plan=plan_trash,
            release_plan=plan_release,
            home_plan=plan_home,
            tip_attachment=tip_mount.tip_attachment.to_mapping(),
            tip_target_info=self.target_builder.tip_target_info,
            tube_target_info=self.target_builder.tube_target_info,
            visual_servo_events=self.visual_servo_events,
            execution_site_errors=self.execution_site_errors,
            episode_arrays=self._episode_array_metadata(arrays),
            lerobot_conversion=self._lerobot_conversion_metadata(arrays),
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    # ---------- Helpers ----------
    def _prepare_lerobot_arrays(
        self,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Normalize ADP episode arrays for the LeRobot ACT converter."""

        prepared = dict(arrays)
        steps = int(prepared["action"].shape[0])
        prepared["time"] = (
            np.arange(steps, dtype=np.float64) * float(self.env.control_dt)
        )
        return prepared

    def _episode_array_metadata(
        self,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
            }
            for key, value in arrays.items()
        }

    def _lerobot_conversion_metadata(
        self,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        camera_keys = [
            f"image_{camera}"
            for camera in self.runtime.cameras
            if f"image_{camera}" in arrays
        ]
        return {
            "converter": "scripts/convert_autolabsim_to_lerobot_act.py",
            "state_key": "ctrl",
            "action_key": "action",
            "action_offset": 1,
            "time_key": "time",
            "control_dt": float(self.env.control_dt),
            "fps": int(round(1.0 / float(self.env.control_dt))),
            "time_aligned_to_control_dt": True,
            "camera_keys": camera_keys,
            "requires_images": True,
        }

    def _update_mounted_tip_end_offset(self) -> None:
        if self.target_builder.tip_target_info is None:
            return
        tip_end_site = self.target_builder.tip_target_info.get("tip_end_site")
        if not tip_end_site:
            return
        piptip_pos, _ = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_tip_site,
        )
        tip_end_pos, _ = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            str(tip_end_site),
        )
        self.target_builder.set_mounted_tip_end_offset(
            piptip_pos,
            tip_end_pos,
            tip_end_site=str(tip_end_site),
        )

    def _capture_tool_attachment(self, arm_name: str) -> ToolAttachment:
        gripper_site = self._gripper_site(arm_name)
        return ToolAttachment(
            arm=arm_name,
            attachment=SiteAttachment.from_mapping(
                self.runtime.pipette.pipette_joint,
                gripper_site,
                capture_site_attachment(
                    self.env.model,
                    self.env.data,
                    self.env.mujoco,
                    self.runtime.pipette.pipette_joint,
                    gripper_site,
                ),
            ),
        )

    def _capture_tip_attachment(self, tool_holder: ToolAttachment) -> TipMountResult:
        tip_joint = self.target_builder.target_tip_joint()
        tip_attachment = SiteAttachment.from_mapping(
            tip_joint,
            self.runtime.pipette.pipette_tip_site,
            capture_site_attachment(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                tip_joint,
                self.runtime.pipette.pipette_tip_site,
            ),
        )
        return TipMountResult(
            mount_down_plan=[],
            tip_joint=tip_joint,
            tip_attachment=tip_attachment,
        )

    def _execute_tip_mount_axis_servo(
        self,
        recorder: EpisodeRecorder,
        plan: list[PlannedTaskTarget],
        context: ExecutionContext,
    ) -> None:
        current_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        current_action[self._gripper_id(self.runtime.arm)] = (
            self.runtime.gripper.close_value
        )
        site_name = self.runtime.pipette.pipette_tip_site
        silent_recorder = _DiscardingRecorder()
        record_steps = max(1, int(self.runtime.timing.steps_per_segment) // 4)
        for item in plan:
            phase = f"tip_mount:{item.name}"
            target_pos = item.debug_target_pos
            if target_pos is None:
                target_pos = item.resolved.pos
            current_action = self.executor.visual_servo_site_to_target(
                silent_recorder,
                current_action,
                phase,
                site_name,
                target_pos,
                item.debug_target_quat,
                arm_name=str(item.target.arm),
                context=context,
            )
            current_action[self._gripper_id(str(item.target.arm))] = (
                self.runtime.gripper.close_value
            )
            item.action = current_action.copy()
            self.executor.hold_action(
                recorder,
                current_action,
                record_steps,
                phase,
                context,
            )
            self.executor.record_site_target_error(phase, site_name, target_pos)

    def _validate_tip_hover_alignment(self) -> dict[str, Any]:
        alignment = self._tip_axis_alignment()
        hover_height = float(self.runtime.tips.tip_hover_height)
        height_error = abs(float(alignment["piptip_minus_mount"][2]) - hover_height)
        alignment["expected_hover_height"] = hover_height
        alignment["height_error"] = height_error
        self.target_builder.tip_target_info[
            "tip_hover_alignment"
        ] = alignment
        if (
            float(alignment["piptip_xy_error"]) > self.runtime.tips.tip_attach_xy_tolerance
            or float(alignment["pipette_mount_xy_error"]) > self.runtime.tips.tip_attach_xy_tolerance
            or float(alignment["axis_xy_drift"]) > self.runtime.tips.tip_axis_xy_tolerance
        ):
            raise RuntimeError(
                "ADP tip is not centered above the selected tip: "
                f"piptip_xy={float(alignment['piptip_xy_error']) * 1000.0:.2f}mm, "
                f"mount_xy={float(alignment['pipette_mount_xy_error']) * 1000.0:.2f}mm, "
                f"axis_drift={float(alignment['axis_xy_drift']) * 1000.0:.2f}mm"
            )
        return alignment

    def _validate_tip_mount_alignment(self) -> dict[str, Any]:
        alignment = self._tip_axis_alignment()
        actual_depth = float(-alignment["piptip_minus_mount"][2])
        expected_depth = max(
            0.0,
            -float(self.runtime.tips.tip_mount_offset[2]),
        )
        depth_error = abs(actual_depth - expected_depth)
        alignment.update(
            {
                "actual_insert_depth": actual_depth,
                "expected_insert_depth": expected_depth,
                "depth_error": depth_error,
            }
        )
        self.target_builder.tip_target_info[
            "tip_mount_alignment_before_attachment"
        ] = alignment
        if (
            float(alignment["piptip_xy_error"]) > self.runtime.tips.tip_attach_xy_tolerance
            or float(alignment["pipette_mount_xy_error"]) > self.runtime.tips.tip_attach_xy_tolerance
            or float(alignment["axis_xy_drift"]) > self.runtime.tips.tip_axis_xy_tolerance
            or depth_error > self.runtime.tips.tip_attach_depth_tolerance
            or actual_depth <= 0.0
        ):
            raise RuntimeError(
                "Refusing to attach tip before the ADP end is centered and inserted: "
                f"piptip_xy={float(alignment['piptip_xy_error']) * 1000.0:.2f}mm, "
                f"mount_xy={float(alignment['pipette_mount_xy_error']) * 1000.0:.2f}mm, "
                f"axis_drift={float(alignment['axis_xy_drift']) * 1000.0:.2f}mm, "
                f"depth_error={depth_error * 1000.0:.2f}mm, "
                f"actual_depth={actual_depth * 1000.0:.2f}mm"
            )
        return alignment

    def _tip_axis_alignment(self) -> dict[str, Any]:
        if self.target_builder.tip_target_info is None:
            raise RuntimeError("Tip target not selected before attachment")
        mount_site = self.target_builder.tip_target_info.get("tip_mount_site")
        if not mount_site:
            raise RuntimeError("Selected tip has no mount site for attachment check")

        mount_pos, mount_quat = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            str(mount_site),
        )
        piptip_pos, _ = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_tip_site,
        )
        pipette_mount_pos, _ = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_mount_site,
        )
        mount_mat = quat_to_mat(mount_quat)
        delta_world = np.asarray(piptip_pos, dtype=np.float64) - np.asarray(
            mount_pos,
            dtype=np.float64,
        )
        pipette_mount_delta_world = np.asarray(
            pipette_mount_pos,
            dtype=np.float64,
        ) - np.asarray(mount_pos, dtype=np.float64)
        nozzle_axis_world = np.asarray(piptip_pos, dtype=np.float64) - np.asarray(
            pipette_mount_pos,
            dtype=np.float64,
        )
        delta = mount_mat.T @ delta_world
        pipette_mount_delta = mount_mat.T @ pipette_mount_delta_world
        nozzle_axis = mount_mat.T @ nozzle_axis_world
        return {
            "pipette_tip_site": self.runtime.pipette.pipette_tip_site,
            "pipette_mount_site": self.runtime.pipette.pipette_mount_site,
            "tip_mount_site": str(mount_site),
            "piptip_pos": piptip_pos.tolist(),
            "pipette_mount_pos": pipette_mount_pos.tolist(),
            "tip_mount_pos": mount_pos.tolist(),
            "tip_mount_quat": mount_quat.tolist(),
            "piptip_minus_mount": delta.tolist(),
            "piptip_minus_mount_world": delta_world.tolist(),
            "pipette_mount_minus_tip_mount": pipette_mount_delta.tolist(),
            "pipette_mount_minus_tip_mount_world": (
                pipette_mount_delta_world.tolist()
            ),
            "nozzle_axis": nozzle_axis.tolist(),
            "nozzle_axis_world": nozzle_axis_world.tolist(),
            "piptip_xy_error": float(np.linalg.norm(delta[:2])),
            "pipette_mount_xy_error": float(np.linalg.norm(pipette_mount_delta[:2])),
            "axis_xy_drift": float(np.linalg.norm(nozzle_axis[:2])),
            "xy_tolerance": self.runtime.tips.tip_attach_xy_tolerance,
            "depth_tolerance": self.runtime.tips.tip_attach_depth_tolerance,
            "axis_xy_tolerance": self.runtime.tips.tip_axis_xy_tolerance,
        }

    def _plan_targets(
        self,
        targets: tuple[TaskTarget, ...] | list[TaskTarget],
        tool_holder: ToolAttachment,
        *,
        default_gripper_value: float,
    ) -> list[PlannedTaskTarget]:
        context = tool_holder.planning_context(
            self.runtime.pipette.pipette_tip_site
        )
        return self.planner.plan(
            targets,
            context,
            default_gripper_value=default_gripper_value,
        )

    def _record_kinematic_hold(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        steps: int,
        phase: str,
        context: ExecutionContext,
    ) -> None:
        """Record a hold without advancing dynamics, preserving exact mount pose."""

        hold_action = np.asarray(action, dtype=np.float64).copy()
        for _ in range(max(0, int(steps))):
            self.env.data.qvel[:] = 0.0
            self.env.data.ctrl[:] = hold_action
            obs = self.executor.apply_constraints(context) or self.env.get_observation()
            recorder.record(obs, hold_action, phase)

    def _execution_context(
        self,
        *,
        fixed_joint_states: tuple[
            tuple[str, tuple[np.ndarray, np.ndarray]],
            ...,
        ] = (),
        attachments: tuple[SiteAttachment, ...] = (),
    ) -> ExecutionContext:
        return ExecutionContext(
            fixed_joint_states=tuple(
                FixedJointState(joint_name=name, state=state)
                for name, state in fixed_joint_states
            ),
            attachments=attachments,
        )

    def _gripper_site(self, arm_name: str) -> str:
        return str(ARM_DEFAULTS[arm_name]["gripper_site"])

    def _gripper_id(self, arm_name: str) -> int:
        return actuator_id(
            self.env.model,
            self.env.mujoco,
            str(ARM_DEFAULTS[arm_name]["gripper_actuator"]),
        )
