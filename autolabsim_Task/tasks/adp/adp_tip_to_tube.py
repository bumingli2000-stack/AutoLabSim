"""ADP tip-to-tube task: mount a visible tip, visit tube hover, drop tip, and home."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...executor import TaskTargetExecutor
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
    joint_qpos_ids,
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

        # 4. Tip mount (down)
        mount_targets = self.target_builder.tip_mount_down_targets(self.runtime.arm)
        plan_mount = self._plan_targets(
            mount_targets,
            tool_holder,
            default_gripper_value=self.runtime.gripper.close_value,
        )
        mount_context = self._execution_context(
            fixed_joint_states=((target_tip_joint, target_tip_state),),
            attachments=tool_holder.follow(),
        )
        self._execute_kinematic_mount_plan(
            recorder,
            plan_mount,
            "tip_mount",
            mount_context,
        )
        self._record_kinematic_hold(
            recorder,
            np.asarray(plan_mount[-1].action, dtype=np.float64),
            self.runtime.timing.tip_mount_settle_steps,
            "tip_mount_settle",
            mount_context,
        )
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

        arrays = recorder.to_arrays()
        metadata = self.metadata_builder.build(
            tip_hover_targets=tip_hover_targets,
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
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    # ---------- Helpers ----------
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

    def _execute_kinematic_mount_plan(
        self,
        recorder: EpisodeRecorder,
        plan: list[PlannedTaskTarget],
        phase_prefix: str,
        context: ExecutionContext,
    ) -> None:
        """Record the short tip-mount insertion using exact planned IK states."""

        for item in plan:
            phase = f"{phase_prefix}:{item.name}"
            qpos_start = self.env.data.qpos.copy()
            qpos_target = qpos_start.copy()
            qpos_ids = joint_qpos_ids(
                self.env.model,
                self.env.mujoco,
                item.arm_joint_names,
            )
            qpos_target[qpos_ids] = np.asarray(item.arm_qpos, dtype=np.float64)
            action = np.asarray(item.action, dtype=np.float64).copy()
            action[self._gripper_id(str(item.target.arm))] = self.runtime.gripper.close_value
            steps = max(1, min(4, self.runtime.timing.steps_per_segment))
            for step in range(1, steps + 1):
                alpha = step / steps
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                self.env.data.qpos[:] = (1.0 - alpha) * qpos_start + alpha * qpos_target
                self.env.data.qvel[:] = 0.0
                self.env.data.ctrl[:] = action
                obs = self.executor.apply_constraints(context) or self.env.get_observation()
                recorder.record(obs, action, phase)
            if item.debug_target_pos is not None:
                self.executor.record_site_target_error(
                    phase,
                    self.runtime.pipette.pipette_tip_site,
                    item.debug_target_pos,
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
