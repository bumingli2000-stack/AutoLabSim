"""Specialized execution for the bimanual screw-cap task.

The shared :class:`TaskTargetPlanner` and :class:`TaskTargetExecutor` handle
ordinary robot motion.  This module contains only the behavior that is unique
to screw-cap opening:

- build the ratchet-style unscrew sequence;
- associate each rotational waypoint with a cumulative ``twist_angle``;
- interpolate ``ScrewCapSystem.commanded_twist`` while the robot moves;
- refresh the scripted cap pose after fixed-joint/attachment constraints;
- optionally execute ordinary multi-waypoint plans through TOPPRA.

The workflow task remains responsible for the *lifecycle* of the screw system:
creating it, calling ``engage()``, switching to follow-after-release, and
releasing that follow relation after the cap is placed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from autolabsim.executor import TaskTargetExecutor
from autolabsim.math3d import gripper_quat_from_axes, unit
from autolabsim.motion_context import ExecutionContext, PlanningContext
from autolabsim.planner import TaskTargetPlanner
from autolabsim.scene import actuator_id, joint_qpos_ids, site_pose
from autolabsim.screw import ScrewCapSystem
from autolabsim.task_target import PlannedTaskTarget, TaskTarget
from autolabsim.topp import Topp, ToppConfig
from autolabsim.tasks.screw_cap.screw_cap_targets import ScrewCapTargetBuilder


ArmDefaults = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ScrewCapExecutionSettings:
    """Execution parameters needed by the screw-specific controller.

    This type intentionally contains only parameters used by this module.  It
    can currently be created from the existing flat
    ``BimanualUnscrewTaskConfig`` through :meth:`from_runtime`.  When the task
    configuration is later split into nested dataclasses, only that conversion
    method needs to change.
    """

    cap_arm: str
    cap_approach_axis: tuple[float, float, float]
    cap_tool_roll: float

    open_gripper: float
    cap_close_gripper: float
    close_steps: int
    steps_per_segment: int

    release_angle: float
    release_lift: float
    thread_pitch: float
    ratchet_angle: float

    waypoint_settle_steps: int
    waypoint_settle_pos_tol: float

    use_topp: bool
    topp_vel: float
    topp_acc: float

    @classmethod
    def from_runtime(cls, runtime: Any) -> "ScrewCapExecutionSettings":
        """Create settings from the current task configuration object."""

        return cls(
            cap_arm=str(runtime.cap_arm),
            cap_approach_axis=tuple(
                float(value) for value in runtime.cap_approach_axis
            ),
            cap_tool_roll=float(runtime.cap_tool_roll),
            open_gripper=float(runtime.open_gripper),
            cap_close_gripper=float(runtime.cap_close_gripper),
            close_steps=int(runtime.close_steps),
            steps_per_segment=int(runtime.steps_per_segment),
            release_angle=float(runtime.release_angle),
            release_lift=float(runtime.release_lift),
            thread_pitch=float(runtime.thread_pitch),
            ratchet_angle=float(runtime.ratchet_angle),
            waypoint_settle_steps=int(runtime.waypoint_settle_steps),
            waypoint_settle_pos_tol=float(runtime.waypoint_settle_pos_tol),
            use_topp=bool(runtime.use_topp),
            topp_vel=float(runtime.topp_vel),
            topp_acc=float(runtime.topp_acc),
        )


class ScrewCapExecutionController:
    """Build and execute the task-specific screw/ratchet motion.

    Parameters
    ----------
    env, manager:
        Active environment and simulation manager.
    planner:
        Shared task-space planner used to solve the rotational gripper targets.
    executor:
        Shared executor used for normal plans and common error recording.
    target_builder:
        Shared screw-cap target builder.  Its public ``world_gripper_target``
        method is reused so ordinary and rotational targets follow identical
        conventions.
    screw_system:
        Scripted screw behavior system that turns cumulative rotation into cap
        rotation and axial lift.
    arm_defaults:
        Raw arm mapping, normally ``ARM_DEFAULTS``.  The raw mapping is needed
        for the cap gripper closing axis.
    settings:
        Screw-specific execution settings.
    """

    def __init__(
        self,
        env: Any,
        manager: Any,
        planner: TaskTargetPlanner,
        executor: TaskTargetExecutor,
        target_builder: ScrewCapTargetBuilder,
        screw_system: ScrewCapSystem,
        arm_defaults: ArmDefaults,
        settings: ScrewCapExecutionSettings,
    ) -> None:
        self.env = env
        self.manager = manager
        self.planner = planner
        self.executor = executor
        self.target_builder = target_builder
        self.screw_system = screw_system
        self.arm_defaults = arm_defaults
        self.settings = settings

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def build_unscrew_plan(
        self,
        grasp_pos: np.ndarray,
        start_action: np.ndarray,
    ) -> list[PlannedTaskTarget]:
        """Build the complete ratchet-style unscrew plan.

        Each cycle consists of:

        1. rotate while gripping the cap;
        2. open the cap gripper;
        3. rewind the wrist to the reference orientation;
        4. close the gripper again.

        The final cycle stops after the rotation because the accumulated twist
        has reached ``release_angle``.  ``PlannedTaskTarget.twist_angle`` stores
        cumulative screw rotation; the wrist orientation itself only rotates by
        one ratchet segment before being rewound.
        """

        cap_arm = self._raw_arm(self.settings.cap_arm)
        approach = unit(
            np.asarray(
                self.settings.cap_approach_axis,
                dtype=np.float64,
            ),
            "cap_unscrew_axis",
        )
        closing = unit(
            cap_arm["closing_axis"],
            "cap_closing_axis",
        )
        base_pos = np.asarray(grasp_pos, dtype=np.float64).copy()
        gripper_id = actuator_id(
            self.env.model,
            self.env.mujoco,
            str(cap_arm["gripper_actuator"]),
        )

        ratchet_angle = max(1e-6, float(self.settings.ratchet_angle))
        release_angle = max(0.0, float(self.settings.release_angle))
        loop_count = max(1, int(np.ceil(release_angle / ratchet_angle)))

        plan: list[PlannedTaskTarget] = []
        current_action = np.asarray(start_action, dtype=np.float64).copy()
        accumulated_twist = 0.0

        for loop_id in range(1, loop_count + 1):
            remaining = max(0.0, release_angle - accumulated_twist)
            segment_angle = min(ratchet_angle, remaining)
            twist_angle = accumulated_twist + segment_angle
            target_pos = self._position_for_twist(base_pos, twist_angle)

            # The physical wrist rotates only one ratchet segment.  The screw
            # system receives the cumulative angle through ``twist_angle``.
            rotate_quat = gripper_quat_from_axes(
                self.env.mujoco,
                approach,
                closing,
                self.settings.cap_tool_roll - segment_angle,
            )
            rotate_target = self.target_builder.world_gripper_target(
                f"cap_ratchet_twist_{loop_id:02d}",
                target_pos,
                rotate_quat,
                self.settings.cap_arm,
                self.settings.cap_close_gripper,
            )
            rotate_item = self._plan_single_target(
                rotate_target,
                self.settings.cap_close_gripper,
            )
            rotate_item.twist_angle = float(twist_angle)
            plan.append(rotate_item)
            current_action = np.asarray(
                rotate_item.action,
                dtype=np.float64,
            ).copy()
            accumulated_twist = twist_angle

            if twist_angle >= release_angle - 1e-9:
                break

            # Open without changing arm joints.  This is a manual action item,
            # so ``arm_qpos`` is empty and the complete mixed plan cannot be
            # incorrectly passed to TOPPRA.
            open_action = current_action.copy()
            open_action[gripper_id] = self.settings.open_gripper
            open_target = self.target_builder.world_gripper_target(
                f"cap_ratchet_open_{loop_id:02d}",
                target_pos,
                rotate_quat,
                self.settings.cap_arm,
                self.settings.open_gripper,
            )
            plan.append(
                self._manual_action_item(
                    open_target,
                    open_action,
                    steps=self.settings.close_steps,
                    twist_angle=twist_angle,
                )
            )

            # Rewind the wrist while the gripper is open.  The commanded screw
            # angle remains unchanged, so the cap does not script itself back
            # onto the tube during the wrist reset.
            rewind_quat = gripper_quat_from_axes(
                self.env.mujoco,
                approach,
                closing,
                self.settings.cap_tool_roll,
            )
            rewind_target = self.target_builder.world_gripper_target(
                f"cap_ratchet_rewind_{loop_id:02d}",
                target_pos,
                rewind_quat,
                self.settings.cap_arm,
                self.settings.open_gripper,
            )
            rewind_item = self._plan_single_target(
                rewind_target,
                self.settings.open_gripper,
            )
            rewind_item.twist_angle = float(twist_angle)
            plan.append(rewind_item)

            # Regrip without changing arm joints, ready for the next segment.
            close_action = np.asarray(
                rewind_item.action,
                dtype=np.float64,
            ).copy()
            close_action[gripper_id] = self.settings.cap_close_gripper
            regrip_target = self.target_builder.world_gripper_target(
                f"cap_ratchet_regrip_{loop_id:02d}",
                target_pos,
                rewind_quat,
                self.settings.cap_arm,
                self.settings.cap_close_gripper,
            )
            plan.append(
                self._manual_action_item(
                    regrip_target,
                    close_action,
                    steps=self.settings.close_steps,
                    twist_angle=twist_angle,
                )
            )
            current_action = close_action

        return plan

    # ------------------------------------------------------------------
    # Execution entry points
    # ------------------------------------------------------------------

    def execute(
        self,
        recorder: Any,
        plan: Sequence[PlannedTaskTarget],
        phase_prefix: str,
        context: ExecutionContext = ExecutionContext(),
        *,
        allow_topp: bool = True,
    ) -> None:
        """Execute a plan through the appropriate execution path.

        Selection order:

        - a plan containing ``twist_angle`` uses screw-aware execution;
        - an eligible ordinary plan may use TOPPRA;
        - all other plans are delegated to the shared executor.
        """

        items = list(plan)
        if not items:
            return
        if any(item.twist_angle is not None for item in items):
            self.execute_unscrew_plan(
                recorder,
                items,
                phase_prefix,
                context,
            )
            return
        if allow_topp and self._can_use_topp(items):
            self._execute_topp_plan(
                recorder,
                items,
                phase_prefix,
                context,
            )
            return
        self.executor.execute(
            recorder,
            items,
            phase_prefix,
            context,
        )

    def execute_unscrew_plan(
        self,
        recorder: Any,
        plan: Sequence[PlannedTaskTarget],
        phase_prefix: str,
        context: ExecutionContext,
    ) -> None:
        """Execute a ratchet plan while maintaining cumulative screw twist."""

        for item in plan:
            if item.twist_angle is None:
                raise ValueError(
                    f"Unscrew plan item {item.name!r} has no twist_angle"
                )
            phase = f"{phase_prefix}:{item.name}"

            self._apply_target_gripper_command(
                recorder,
                item,
                "before",
                f"{phase}:gripper_before",
                context,
            )
            self.move_action(
                recorder,
                np.asarray(item.action, dtype=np.float64),
                int(
                    item.steps
                    if item.steps is not None
                    else self.settings.steps_per_segment
                ),
                phase,
                context,
                twist_target=float(item.twist_angle),
            )

            site_name = self._controlled_site(item)
            if item.servo_mode != "none" and item.debug_target_pos is not None:
                self.settle_until_site_reached(
                    recorder,
                    np.asarray(item.action, dtype=np.float64),
                    phase,
                    site_name,
                    item.debug_target_pos,
                    context,
                    twist_target=float(item.twist_angle),
                )
                self.executor.record_site_target_error(
                    phase,
                    site_name,
                    item.debug_target_pos,
                )

            self._apply_target_gripper_command(
                recorder,
                item,
                "after",
                f"{phase}:gripper_after",
                context,
            )

    # ------------------------------------------------------------------
    # Screw-aware motion primitives
    # ------------------------------------------------------------------

    def move_action(
        self,
        recorder: Any,
        target_action: np.ndarray,
        steps: int,
        phase: str,
        context: ExecutionContext,
        *,
        twist_target: float,
    ) -> None:
        """Interpolate an action and cumulative screw twist together."""

        start_action = np.asarray(
            self.env.data.ctrl,
            dtype=np.float64,
        ).copy()
        target = np.asarray(target_action, dtype=np.float64)
        denom = max(1, int(steps))
        start_twist = float(self.screw_system.progress.twist_angle)

        for step in range(1, denom + 1):
            alpha = step / denom
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            action = (1.0 - alpha) * start_action + alpha * target
            commanded_twist = (
                start_twist + alpha * (float(twist_target) - start_twist)
            )
            self.screw_system.set_commanded_twist(commanded_twist)

            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints_and_refresh(
                action,
                obs,
                context,
            )
            recorder.record(obs, action, phase)

        self.screw_system.set_commanded_twist(float(twist_target))

    def hold_action(
        self,
        recorder: Any,
        action: np.ndarray,
        steps: int,
        phase: str,
        context: ExecutionContext,
        *,
        twist_target: float,
    ) -> None:
        """Hold an action while preserving a fixed commanded screw angle."""

        target_action = np.asarray(action, dtype=np.float64)
        self.screw_system.set_commanded_twist(float(twist_target))
        for _ in range(max(0, int(steps))):
            obs, *_ = self.manager.step(target_action)
            obs = self._apply_constraints_and_refresh(
                target_action,
                obs,
                context,
            )
            recorder.record(obs, target_action, phase)

    def settle_until_site_reached(
        self,
        recorder: Any,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        context: ExecutionContext,
        *,
        twist_target: float,
    ) -> None:
        """Settle a rotational waypoint without losing screw state."""

        target = np.asarray(target_pos, dtype=np.float64)
        settle_phase = f"{phase}:settle"
        target_action = np.asarray(action, dtype=np.float64)
        self.screw_system.set_commanded_twist(float(twist_target))

        for _ in range(max(0, self.settings.waypoint_settle_steps)):
            actual, _ = site_pose(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                site_name,
            )
            if (
                float(np.linalg.norm(target - actual))
                <= self.settings.waypoint_settle_pos_tol
            ):
                return

            obs, *_ = self.manager.step(target_action)
            obs = self._apply_constraints_and_refresh(
                target_action,
                obs,
                context,
            )
            recorder.record(obs, target_action, settle_phase)

    # ------------------------------------------------------------------
    # Optional TOPP execution for ordinary plans
    # ------------------------------------------------------------------

    def _can_use_topp(self, plan: Sequence[PlannedTaskTarget]) -> bool:
        if not self.settings.use_topp or len(plan) < 2:
            return False
        first_joint_names = tuple(plan[0].arm_joint_names)
        return (
            bool(first_joint_names)
            and all(tuple(item.arm_joint_names) == first_joint_names for item in plan)
            and all(item.arm_qpos.size > 0 for item in plan)
            and all(item.steps is None for item in plan)
            and all(item.twist_angle is None for item in plan)
            and all(
                item.gripper_command is None
                or item.gripper_command.timing == "during"
                for item in plan
            )
        )

    def _execute_topp_plan(
        self,
        recorder: Any,
        plan: list[PlannedTaskTarget],
        phase_prefix: str,
        context: ExecutionContext,
    ) -> None:
        """Execute a homogeneous ordinary plan through TOPPRA."""

        joint_names = tuple(str(name) for name in plan[0].arm_joint_names)
        qpos_ids = joint_qpos_ids(
            self.env.model,
            self.env.mujoco,
            joint_names,
        )
        action_ids = [
            actuator_id(self.env.model, self.env.mujoco, joint_name)
            for joint_name in joint_names
        ]
        start_q = np.asarray(
            [self.env.data.qpos[qpos_id] for qpos_id in qpos_ids],
            dtype=np.float64,
        )
        q_waypoints = np.vstack(
            [
                start_q,
                *(
                    np.asarray(item.arm_qpos, dtype=np.float64)
                    for item in plan
                ),
            ]
        )

        topp = Topp(
            ToppConfig(
                dof=len(joint_names),
                qc_vel=self.settings.topp_vel,
                qc_acc=self.settings.topp_acc,
            )
        )
        trajectory = topp.jnt_traj(q_waypoints)
        duration = float(trajectory.duration)
        steps = max(1, int(np.ceil(duration / self.env.control_dt)))
        segment_edges = np.linspace(0.0, duration, len(plan) + 1)

        for step in range(1, steps + 1):
            t = duration * step / steps
            q = topp.query(trajectory, t)
            item_index = int(
                np.searchsorted(
                    segment_edges[1:],
                    t,
                    side="left",
                )
            )
            item_index = min(item_index, len(plan) - 1)
            item = plan[item_index]
            action = np.asarray(item.action, dtype=np.float64).copy()
            for action_id, value in zip(action_ids, q, strict=True):
                action[action_id] = value

            phase = f"{phase_prefix}:{item.name}"
            obs, *_ = self.manager.step(action)
            obs = self.executor.apply_constraints(context) or obs
            recorder.record(obs, action, phase)

        # Preserve the original task behavior: after a TOPP path only the final
        # key target receives an explicit settle/error check.
        final_item = plan[-1]
        site_name = self._controlled_site(final_item)
        if (
            final_item.servo_mode != "none"
            and final_item.debug_target_pos is not None
        ):
            phase = f"{phase_prefix}:{final_item.name}"
            self.executor.settle_until_site_reached(
                recorder,
                np.asarray(final_item.action, dtype=np.float64),
                phase,
                site_name,
                final_item.debug_target_pos,
                context,
            )
            self.executor.record_site_target_error(
                phase,
                site_name,
                final_item.debug_target_pos,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _position_for_twist(
        self,
        base_pos: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Return the cap gripper position for a cumulative screw angle."""

        lifted = np.asarray(base_pos, dtype=np.float64).copy()
        lifted[2] += min(
            self.settings.release_lift,
            self.settings.thread_pitch * (float(angle) / (2.0 * np.pi)),
        )
        return lifted

    def _plan_single_target(
        self,
        target: TaskTarget,
        default_gripper_value: float,
    ) -> PlannedTaskTarget:
        return self.planner.plan(
            [target],
            PlanningContext(),
            default_gripper_value=default_gripper_value,
        )[0]

    def _manual_action_item(
        self,
        target: TaskTarget,
        action: np.ndarray,
        *,
        steps: int,
        twist_angle: float,
    ) -> PlannedTaskTarget:
        """Wrap a gripper-only action in a planned-target record."""

        resolved = self.planner.resolve(target)
        arm = self.planner.arm_configs[target.arm]
        gripper_value = self.planner.target_gripper_value(target.gripper)
        return PlannedTaskTarget(
            target=target,
            resolved=resolved,
            ik_site_pos=np.asarray(resolved.pos, dtype=np.float64),
            ik_site_quat=np.asarray(resolved.quat, dtype=np.float64),
            action=np.asarray(action, dtype=np.float64).copy(),
            ik_success=True,
            ik_pos_error=0.0,
            ik_rot_error=0.0,
            arm_joint_names=tuple(str(name) for name in arm.joint_names),
            # Empty qpos marks this as a non-IK gripper-only item and prevents
            # a mixed ratchet plan from being accepted by TOPPRA.
            arm_qpos=np.zeros(0, dtype=np.float64),
            gripper_value=gripper_value,
            steps=int(steps),
            twist_angle=float(twist_angle),
        )

    def _apply_target_gripper_command(
        self,
        recorder: Any,
        item: PlannedTaskTarget,
        timing: str,
        phase: str,
        context: ExecutionContext,
    ) -> None:
        command = item.gripper_command
        if command is None or command.timing != timing:
            return

        gripper_value = item.gripper_value
        if gripper_value is None:
            gripper_value = self.planner.target_gripper_value(command)
        if gripper_value is None:
            raise RuntimeError(
                f"Unable to resolve gripper command for {item.name!r}"
            )

        arm = self.planner.arm_configs[item.target.arm]
        gripper_id = actuator_id(
            self.env.model,
            self.env.mujoco,
            arm.gripper_actuator,
        )
        base = self.env.data.ctrl if timing == "before" else item.action
        action = np.asarray(base, dtype=np.float64).copy()
        action[gripper_id] = float(gripper_value)
        self.move_action(
            recorder,
            action,
            int(command.steps),
            phase,
            context,
            twist_target=float(item.twist_angle or 0.0),
        )
        item.action = action

    def _apply_constraints_and_refresh(
        self,
        action: np.ndarray,
        obs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        """Apply ordinary constraints, then recompute the scripted cap pose.

        ``Manager.step`` already calls ``ScrewCapSystem.after_step``.  However,
        restoring a fixed tube joint or applying an attachment changes the
        parent pose *after* that system callback.  The screw pose must therefore
        be refreshed once more from the constrained tube pose.
        """

        constrained_obs = self.executor.apply_constraints(context)
        if constrained_obs is None:
            return obs

        progress = self.screw_system.progress

        if not progress.engaged:
            return constrained_obs

        if progress.released:
            # 最后一次旋转刚达到释放角时，试管随后会被
            # ExecutionContext 恢复。这里立即根据恢复后的试管
            # 位姿重新同步瓶盖，避免阶段切换时闪跳。
            self.screw_system.synchronize_pose_from_current_tube(
                self.env
            )
        else:
            # 未释放时重新运行螺纹更新，使瓶盖位姿基于
            # 已经应用 fixed joint/attachment 后的试管位姿。
            self.screw_system.after_step(
                self.env,
                action,
                constrained_obs,
            )

        return self.env.get_observation()

    def _controlled_site(self, item: PlannedTaskTarget) -> str:
        if item.target.controlled_site:
            return item.target.controlled_site
        return self.planner.arm_configs[item.target.arm].actuator_site

    def _raw_arm(self, arm_name: str) -> Mapping[str, Any]:
        try:
            return self.arm_defaults[arm_name]
        except KeyError as exc:
            raise KeyError(f"Unknown arm name: {arm_name!r}") from exc