"""Dual-arm pipette grasp, tip mounting, handoff, and tube approach task.

The task module owns workflow stages and attachment lifecycle. Task-space target
construction, scene discovery, and metadata serialization live in sibling
modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..executor import TaskTargetExecutor
from ..motion_context import (
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
from ..mujoco_env import EnvConfig
from ..planner import TaskTargetPlanner
from ..recorder import EpisodeRecorder
from ..scene import (
    actuator_id,
    capture_free_joint_state,
    capture_site_attachment,
    equality_id,
)
from ..task import AutoLabTask, TaskConfig
from ..task_target import PlannedTaskTarget, TaskTarget
from .common import ARM_DEFAULTS
from .pipette_metadata import PipetteMetadataBuilder
from .pipette_scene import PipetteSceneQuery
from .pipette_targets import (
    PipetteHandleGraspConfig,
    PipetteModelConfig,
    PipetteTargetBuilder,
    PipetteTipTargetConfig,
    PipetteTubeTargetConfig,
)


@dataclass(frozen=True)
class PipetteRobotConfig:
    arm: str = "first"
    open_gripper: float = 0.0
    close_gripper: float = 255.0


@dataclass(frozen=True)
class PipetteTimingConfig:
    settle_steps: int = 20
    free_settle_steps: int = 20
    steps_per_segment: int = 50
    close_steps: int = 12
    hold_steps: int = 20
    grasp_hold_steps: int = 8


@dataclass(frozen=True)
class IKConfig:
    ik_max_iters: int = 800
    ik_pos_tol: float = 0.0005
    ik_rot_tol: float = 0.02
    ik_damping: float = 0.01


@dataclass(frozen=True)
class WaypointSettleConfig:
    waypoint_settle_steps: int = 10
    waypoint_settle_pos_tol: float = 0.0005


@dataclass(frozen=True)
class VisualServoConfig:
    visual_servo_enabled: bool = True
    visual_servo_max_iters: int = 12
    visual_servo_steps: int = 10
    visual_servo_pos_tol: float = 0.0001
    visual_servo_rot_tol: float = 0.02
    visual_servo_gain: float = 0.8
    visual_servo_integral_gain: float = 0.25
    visual_servo_max_correction: float = 0.02


@dataclass(frozen=True)
class PipetteGraspTaskConfig:
    env: EnvConfig
    out_dir: Path
    episode_index: int
    seed: int
    cameras: tuple[str, ...] = ("overview_camera",)
    with_images: bool = False
    robot: PipetteRobotConfig = field(default_factory=PipetteRobotConfig)
    timing: PipetteTimingConfig = field(default_factory=PipetteTimingConfig)
    grasp: PipetteHandleGraspConfig = field(
        default_factory=PipetteHandleGraspConfig
    )
    pipette: PipetteModelConfig = field(default_factory=PipetteModelConfig)
    tips: PipetteTipTargetConfig = field(
        default_factory=PipetteTipTargetConfig
    )
    tube: PipetteTubeTargetConfig = field(
        default_factory=PipetteTubeTargetConfig
    )
    ik: IKConfig = field(default_factory=IKConfig)
    waypoint: WaypointSettleConfig = field(
        default_factory=WaypointSettleConfig
    )
    visual_servo: VisualServoConfig = field(
        default_factory=VisualServoConfig
    )


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
class PipetteHandoffResult:
    holder: ToolAttachment
    middle_grasp_plan: list[PlannedTaskTarget]
    first_retreat_plan: list[PlannedTaskTarget]


@dataclass(frozen=True)
class TipMountResult:
    mount_down_plan: list[PlannedTaskTarget]
    tip_joint: str
    tip_attachment: SiteAttachment


class PipetteGraspTask(AutoLabTask):
    name = "pipette_grasp"

    def __init__(self, config: PipetteGraspTaskConfig) -> None:
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
            open_value=config.robot.open_gripper,
            close_value=config.robot.close_gripper,
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
                waypoint_settle_steps=(
                    config.waypoint.waypoint_settle_steps
                ),
                waypoint_settle_pos_tol=(
                    config.waypoint.waypoint_settle_pos_tol
                ),
                visual_servo=VisualServoSettings(
                    enabled=config.visual_servo.visual_servo_enabled,
                    max_iters=config.visual_servo.visual_servo_max_iters,
                    steps=config.visual_servo.visual_servo_steps,
                    pos_tol=config.visual_servo.visual_servo_pos_tol,
                    rot_tol=config.visual_servo.visual_servo_rot_tol,
                    gain=config.visual_servo.visual_servo_gain,
                    integral_gain=(
                        config.visual_servo.visual_servo_integral_gain
                    ),
                    max_correction=(
                        config.visual_servo.visual_servo_max_correction
                    ),
                ),
            ),
        )
        self.scene_query = PipetteSceneQuery(
            self.env,
            pipette_tip_site=config.pipette.pipette_tip_site,
            tip_joint_prefix=config.tips.tip_joint_prefix,
            tip_site_prefix=config.tips.tip_site_prefix,
            tip_mount_site_suffix=config.tips.tip_mount_site_suffix,
            tip_end_site_suffix=config.tips.tip_end_site_suffix,
            fallback_tube_joint=config.tube.tube_joint,
        )
        self.target_builder = PipetteTargetBuilder(
            self.env,
            self.planner,
            self.scene_query,
            ARM_DEFAULTS,
            primary_arm=config.robot.arm,
            close_steps=config.timing.close_steps,
            grasp=config.grasp,
            pipette=config.pipette,
            tips=config.tips,
            tube=config.tube,
        )
        self.metadata_builder = PipetteMetadataBuilder(self.env, config)

        self.execution_site_errors = self.executor.execution_site_errors
        self.visual_servo_events = self.executor.visual_servo_events

    def run(self) -> dict[str, Any]:
        """Run the scripted pipette workflow."""

        obs = self.reset()
        action = self._open_initial_grippers(
            np.asarray(obs["ctrl"], dtype=np.float64)
        )
        for _ in range(max(0, int(self.runtime.timing.free_settle_steps))):
            obs, *_ = self.manager.step(action)

        initial_pipette_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_joint,
        )
        recorder = EpisodeRecorder(
            cameras=self.runtime.cameras,
            with_images=self.runtime.with_images,
        )
        self.executor.hold_action(
            recorder,
            action,
            self.runtime.timing.settle_steps,
            "settle",
            self._execution_context(
                fixed_joint_states=(
                    (
                        self.runtime.pipette.pipette_joint,
                        initial_pipette_state,
                    ),
                ),
            ),
        )

        grasp_plan = self._stage_primary_grasp(
            recorder,
            initial_pipette_state,
        )
        self._set_equality_active_if_exists(
            self.runtime.pipette.parking_weld,
            False,
        )

        tool_holder = self._capture_tool_attachment(self.runtime.robot.arm)
        lift_plan = self._stage_move_tool_to_tip_hover(
            recorder,
            tool_holder,
        )
        tip_mount = self._stage_mount_tip(recorder, tool_holder)
        tip_retract_plan = self._stage_retract_mounted_tip(
            recorder,
            tool_holder,
            tip_mount,
        )
        tube_hover_plan = self._stage_move_tip_to_tube_hover(
            recorder,
            tool_holder,
            tip_mount,
        )
        handoff = self._stage_handoff_to_middle_arm(
            recorder,
            tool_holder,
            extra_follow=self._tip_follow(tip_mount),
        )
        tool_holder = handoff.holder
        tube_near_plan = self._stage_move_tip_near_tube(
            recorder,
            tool_holder,
            tip_mount,
        )
        tube_plan = tube_hover_plan + tube_near_plan

        arrays = recorder.to_arrays()
        metadata = self.metadata_builder.build(
            grasp_plan,
            handoff.middle_grasp_plan,
            handoff.first_retreat_plan,
            lift_plan,
            tip_mount.mount_down_plan,
            tip_retract_plan,
            tube_plan,
            handoff_attachment=handoff.holder.attachment.to_mapping(),
            tip_attachment=tip_mount.tip_attachment.to_mapping(),
            tip_target_info=self.target_builder.tip_target_info,
            tube_target_info=self.target_builder.tube_target_info,
            visual_servo_events=self.visual_servo_events,
            execution_site_errors=self.execution_site_errors,
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    def _stage_primary_grasp(
        self,
        recorder: EpisodeRecorder,
        initial_pipette_state: tuple[np.ndarray, np.ndarray],
    ) -> list[PlannedTaskTarget]:
        fixed_pipette = (
            (
                self.runtime.pipette.pipette_joint,
                initial_pipette_state,
            ),
        )
        context = self._execution_context(
            fixed_joint_states=fixed_pipette
        )
        plan = self._plan_targets(
            self.target_builder.primary_grasp_targets(),
            default_gripper_value=self.runtime.robot.open_gripper,
        )
        self.executor.execute(
            recorder,
            plan,
            "move_to_pipette",
            context,
        )

        grasp_action = np.asarray(
            plan[-1].action,
            dtype=np.float64,
        ).copy()
        self.executor.hold_action(
            recorder,
            grasp_action,
            self.runtime.timing.grasp_hold_steps,
            "hold_at_grasp",
            context,
        )

        close_action = grasp_action.copy()
        close_action[self._gripper_id(self.runtime.robot.arm)] = (
            self.runtime.robot.close_gripper
        )
        self.executor.move_action(
            recorder,
            close_action,
            self.runtime.timing.close_steps,
            "close_gripper",
            context,
        )
        return plan

    def _stage_move_tool_to_tip_hover(
        self,
        recorder: EpisodeRecorder,
        tool_holder: ToolAttachment,
    ) -> list[PlannedTaskTarget]:
        built = self.target_builder.tip_hover_targets(tool_holder.arm)
        plan = self._plan_targets(
            built.targets,
            default_gripper_value=self.runtime.robot.close_gripper,
            planning_context=tool_holder.planning_context(
                self.runtime.pipette.pipette_tip_site
            ),
        )
        tip_info = self.target_builder.target_tip_info()
        plan[0].put_extra("tip_target", tip_info)
        plan[0].put_extra("keeps_current_pipette_quat", True)
        plan[1].put_extra("tip_target", tip_info)
        plan[1].put_extra(
            "aligns_pipette_quat",
            built.aligned_joint_quat.tolist(),
        )

        context = self._execution_context(
            attachments=tool_holder.follow()
        )
        self.executor.execute(
            recorder,
            plan,
            "move_pipette_to_tip_hover",
            context,
        )
        self.executor.hold_action(
            recorder,
            np.asarray(plan[-1].action, dtype=np.float64),
            self.runtime.timing.hold_steps,
            "hold_lifted",
            context,
        )
        self.executor.record_site_target_error(
            "hold_over_tip",
            self.runtime.pipette.pipette_tip_site,
            tip_info["target_tip_hover_pos"],
        )
        return plan

    def _stage_handoff_to_middle_arm(
        self,
        recorder: EpisodeRecorder,
        current_holder: ToolAttachment,
        *,
        extra_follow: tuple[SiteAttachment, ...] = (),
    ) -> PipetteHandoffResult:
        middle_arm = self.runtime.grasp.middle_grasp_arm
        current_follow = current_holder.follow() + tuple(extra_follow)
        current_context = self._execution_context(
            attachments=current_follow
        )

        middle_grasp_plan = self._plan_targets(
            self.target_builder.middle_grasp_targets(),
            default_gripper_value=self.runtime.robot.open_gripper,
        )
        self.executor.execute(
            recorder,
            middle_grasp_plan,
            "move_second_arm_to_pipette_middle",
            current_context,
        )

        middle_grasp_action = np.asarray(
            middle_grasp_plan[-1].action,
            dtype=np.float64,
        ).copy()
        self.executor.hold_action(
            recorder,
            middle_grasp_action,
            self.runtime.timing.grasp_hold_steps,
            "hold_second_arm_at_middle_grasp",
            current_context,
        )

        middle_close_action = middle_grasp_action.copy()
        middle_close_action[self._gripper_id(middle_arm)] = (
            self.runtime.robot.close_gripper
        )
        self.executor.move_action(
            recorder,
            middle_close_action,
            self.runtime.timing.close_steps,
            "close_second_gripper_on_pipette",
            current_context,
        )

        next_holder = self._capture_tool_attachment(middle_arm)
        next_follow = next_holder.follow() + tuple(extra_follow)
        next_context = self._execution_context(attachments=next_follow)

        first_release_action = middle_close_action.copy()
        first_release_action[self._gripper_id(self.runtime.robot.arm)] = (
            self.runtime.robot.open_gripper
        )
        self.executor.move_action(
            recorder,
            first_release_action,
            self.runtime.timing.close_steps,
            "release_first_gripper_after_handoff",
            next_context,
        )

        first_retreat_plan = self._plan_targets(
            self.target_builder.first_retreat_after_handoff_targets(),
            default_gripper_value=self.runtime.robot.open_gripper,
        )
        self.executor.execute(
            recorder,
            first_retreat_plan,
            "retreat_first_gripper_after_handoff",
            next_context,
        )

        first_close_action = np.asarray(
            first_retreat_plan[-1].action,
            dtype=np.float64,
        ).copy()
        first_close_action[self._gripper_id(self.runtime.robot.arm)] = (
            self.runtime.robot.close_gripper
        )
        self.executor.move_action(
            recorder,
            first_close_action,
            self.runtime.timing.close_steps,
            "close_first_gripper_after_retreat",
            next_context,
        )
        return PipetteHandoffResult(
            holder=next_holder,
            middle_grasp_plan=middle_grasp_plan,
            first_retreat_plan=first_retreat_plan,
        )

    def _stage_mount_tip(
        self,
        recorder: EpisodeRecorder,
        tool_holder: ToolAttachment,
    ) -> TipMountResult:
        target_tip_joint = self.target_builder.target_tip_joint()
        target_tip_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            target_tip_joint,
        )
        plan = self._plan_targets(
            self.target_builder.tip_mount_down_targets(tool_holder.arm),
            default_gripper_value=self.runtime.robot.close_gripper,
            planning_context=tool_holder.planning_context(
                self.runtime.pipette.pipette_tip_site
            ),
        )
        plan[0].put_extra(
            "tip_target",
            self.target_builder.target_tip_info(),
        )
        plan[0].put_extra(
            "mount_tip_site_pos",
            plan[0].resolved.pos.tolist(),
        )
        context = self._execution_context(
            fixed_joint_states=((target_tip_joint, target_tip_state),),
            attachments=tool_holder.follow(),
        )
        self.executor.execute(
            recorder,
            plan,
            "mount_pipette_tip",
            context,
        )

        tip_attachment = SiteAttachment.from_mapping(
            target_tip_joint,
            self.runtime.pipette.pipette_tip_site,
            capture_site_attachment(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                target_tip_joint,
                self.runtime.pipette.pipette_tip_site,
            ),
        )
        return TipMountResult(
            mount_down_plan=plan,
            tip_joint=target_tip_joint,
            tip_attachment=tip_attachment,
        )

    def _stage_retract_mounted_tip(
        self,
        recorder: EpisodeRecorder,
        tool_holder: ToolAttachment,
        tip_mount: TipMountResult,
    ) -> list[PlannedTaskTarget]:
        plan = self._plan_targets(
            self.target_builder.tip_retract_targets(tool_holder.arm),
            default_gripper_value=self.runtime.robot.close_gripper,
            planning_context=tool_holder.planning_context(
                self.runtime.pipette.pipette_tip_site
            ),
        )
        plan[0].put_extra(
            "tip_target",
            self.target_builder.target_tip_info(),
        )
        plan[0].put_extra(
            "retract_tip_site_pos",
            plan[0].resolved.pos.tolist(),
        )
        self.executor.execute(
            recorder,
            plan,
            "retract_mounted_tip",
            self._execution_context(
                attachments=self._mounted_tip_follow(
                    tool_holder,
                    tip_mount,
                )
            ),
        )
        return plan

    def _stage_move_tip_to_tube_hover(
        self,
        recorder: EpisodeRecorder,
        tool_holder: ToolAttachment,
        tip_mount: TipMountResult,
    ) -> list[PlannedTaskTarget]:
        plan = self._plan_targets(
            self.target_builder.tube_hover_targets(tool_holder.arm),
            default_gripper_value=self.runtime.robot.close_gripper,
            planning_context=tool_holder.planning_context(
                self.runtime.pipette.pipette_tip_site
            ),
        )
        self._annotate_tube_plan(plan)
        self.executor.execute(
            recorder,
            plan,
            "move_tip_to_tube_hover",
            self._execution_context(
                attachments=self._mounted_tip_follow(
                    tool_holder,
                    tip_mount,
                )
            ),
        )
        return plan

    def _stage_move_tip_near_tube(
        self,
        recorder: EpisodeRecorder,
        tool_holder: ToolAttachment,
        tip_mount: TipMountResult,
    ) -> list[PlannedTaskTarget]:
        plan = self._plan_targets(
            self.target_builder.tube_near_targets(tool_holder.arm),
            default_gripper_value=self.runtime.robot.close_gripper,
            planning_context=tool_holder.planning_context(
                self.runtime.pipette.pipette_tip_site
            ),
        )
        self._annotate_tube_plan(plan)
        mounted_context = self._execution_context(
            attachments=self._mounted_tip_follow(
                tool_holder,
                tip_mount,
            )
        )
        self.executor.execute(
            recorder,
            plan,
            "move_tip_near_tube",
            mounted_context,
        )
        self.executor.hold_action(
            recorder,
            np.asarray(plan[-1].action, dtype=np.float64),
            self.runtime.timing.hold_steps,
            "hold_tip_near_tube",
            mounted_context,
        )
        tube_info = self.target_builder.tube_target_info
        if tube_info is None:
            raise RuntimeError("Tube target information was not generated")
        self.executor.record_site_target_error(
            "hold_tip_near_tube",
            self.runtime.pipette.pipette_tip_site,
            tube_info["tube_near_tip_site_pos"],
        )
        return plan

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

    def _tip_follow(
        self,
        tip_mount: TipMountResult,
    ) -> tuple[SiteAttachment, ...]:
        if tip_mount.tip_attachment.joint_name != tip_mount.tip_joint:
            raise ValueError(
                "Tip attachment joint mismatch: "
                f"{tip_mount.tip_attachment.joint_name!r} != "
                f"{tip_mount.tip_joint!r}"
            )
        return (tip_mount.tip_attachment,)

    def _mounted_tip_follow(
        self,
        tool_holder: ToolAttachment,
        tip_mount: TipMountResult,
    ) -> tuple[SiteAttachment, ...]:
        return tool_holder.follow() + self._tip_follow(tip_mount)

    def _plan_targets(
        self,
        targets: list[TaskTarget] | tuple[TaskTarget, ...],
        *,
        default_gripper_value: float,
        planning_context: PlanningContext | None = None,
    ) -> list[PlannedTaskTarget]:
        return self.planner.plan(
            targets,
            planning_context or PlanningContext(),
            default_gripper_value=default_gripper_value,
        )

    def _annotate_tube_plan(
        self,
        plan: list[PlannedTaskTarget],
    ) -> None:
        for item in plan:
            item.put_extra(
                "tube_target",
                self.target_builder.tube_target_info,
            )

    def _open_initial_grippers(self, ctrl: np.ndarray) -> np.ndarray:
        action = np.asarray(ctrl, dtype=np.float64).copy()
        arm_names = {
            self.runtime.robot.arm,
            self.runtime.grasp.middle_grasp_arm,
        }
        for arm_name in arm_names:
            action[self._gripper_id(arm_name)] = (
                self.runtime.robot.open_gripper
            )
        return action

    @staticmethod
    def _execution_context(
        *,
        fixed_joint_states: tuple[
            tuple[str, tuple[np.ndarray, np.ndarray]],
            ...,
        ] = (),
        attachments: tuple[SiteAttachment, ...] = (),
    ) -> ExecutionContext:
        return ExecutionContext(
            fixed_joint_states=tuple(
                FixedJointState(
                    joint_name=joint_name,
                    state=state,
                )
                for joint_name, state in fixed_joint_states
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

    def _set_equality_active_if_exists(
        self,
        name: str,
        active: bool,
    ) -> None:
        try:
            eq_id = equality_id(self.env.model, self.env.mujoco, name)
        except ValueError:
            return
        self.env.data.eq_active[eq_id] = 1 if active else 0
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
