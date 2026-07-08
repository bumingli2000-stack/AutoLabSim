"""Execute planned task targets with interpolation, servo, and constraints."""
from __future__ import annotations

from typing import Any

import numpy as np

from .ik import solve_site_ik
from .math3d import normalize_quat, quat_conjugate, quat_multiply
from .motion_context import ArmMotionConfig, ExecutionContext, ExecutionSettings, GripperSettings, IKSettings
from .scene import actuator_id, apply_site_attachment, joint_qpos_ids, restore_free_joint_state, site_pose
from .task_target import PlannedTaskTarget, gripper_command_to_actuator


class TaskTargetExecutor:
    def __init__(
        self,
        env: Any,
        manager: Any,
        arm_configs: dict[str, ArmMotionConfig],
        ik: IKSettings,
        gripper: GripperSettings,
        settings: ExecutionSettings,
    ):
        self.env = env
        self.manager = manager
        self.arm_configs = arm_configs
        self.ik = ik
        self.gripper = gripper
        self.settings = settings
        self.execution_site_errors: list[dict[str, Any]] = []
        self.visual_servo_events: list[dict[str, Any]] = []

    def execute(
        self,
        recorder: Any,
        plan: list[PlannedTaskTarget],
        phase_prefix: str,
        context: ExecutionContext = ExecutionContext(),
    ) -> None:
        for item in plan:
            phase = f"{phase_prefix}:{item.name}"
            self.apply_target_gripper_command(
                recorder,
                item,
                "before",
                f"{phase}:gripper_before",
                context,
            )
            self.move_action(
                recorder,
                np.asarray(item.action, dtype=np.float64),
                int(item.steps if item.steps is not None else self.settings.steps_per_segment),
                phase,
                context,
            )
            site_name = self._controlled_site(item)
            if item.servo_mode != "none" and item.debug_target_pos is not None:
                if self.settings.visual_servo.enabled:
                    item.action = self.visual_servo_site_to_target(
                        recorder,
                        np.asarray(item.action, dtype=np.float64),
                        phase,
                        site_name,
                        item.debug_target_pos,
                        item.debug_target_quat,
                        arm_name=item.target.arm,
                        context=context,
                    )
                else:
                    self.settle_until_site_reached(
                        recorder,
                        np.asarray(item.action, dtype=np.float64),
                        phase,
                        site_name,
                        item.debug_target_pos,
                        context,
                    )
                self.record_site_target_error(phase, site_name, item.debug_target_pos)
            self.apply_target_gripper_command(
                recorder,
                item,
                "after",
                f"{phase}:gripper_after",
                context,
            )

    def apply_target_gripper_command(
        self,
        recorder: Any,
        item: PlannedTaskTarget,
        timing: str,
        phase: str,
        context: ExecutionContext = ExecutionContext(),
    ) -> None:
        command = item.gripper_command
        if command is None or command.timing != timing:
            return
        gripper_value = item.gripper_value
        if gripper_value is None:
            gripper_value = gripper_command_to_actuator(
                command,
                self.gripper.open_value,
                self.gripper.close_value,
            )
        arm = self.arm_configs[str(item.target.arm)]
        gripper_id = actuator_id(self.env.model, self.env.mujoco, arm.gripper_actuator)
        action = np.asarray(self.env.data.ctrl if timing == "before" else item.action, dtype=np.float64).copy()
        action[gripper_id] = float(gripper_value)
        self.move_action(recorder, action, int(command.steps), phase, context)
        item.action = action

    def move_action(
        self,
        recorder: Any,
        target_action: np.ndarray,
        steps: int,
        phase: str,
        context: ExecutionContext = ExecutionContext(),
    ) -> None:
        start = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        denom = max(1, int(steps))
        for step in range(1, denom + 1):
            alpha = step / denom
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            action = (1.0 - alpha) * start + alpha * target_action
            obs, *_ = self.manager.step(action)
            obs = self.apply_constraints(context) or obs
            recorder.record(obs, action, phase)

    def hold_action(
        self,
        recorder: Any,
        action: np.ndarray,
        steps: int,
        phase: str,
        context: ExecutionContext = ExecutionContext(),
    ) -> None:
        for _ in range(max(0, int(steps))):
            obs, *_ = self.manager.step(action)
            obs = self.apply_constraints(context) or obs
            recorder.record(obs, action, phase)

    def settle_until_site_reached(
        self,
        recorder: Any,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        context: ExecutionContext = ExecutionContext(),
    ) -> None:
        target = np.asarray(target_pos, dtype=np.float64)
        settle_phase = f"{phase}:settle"
        for _ in range(max(0, int(self.settings.waypoint_settle_steps))):
            actual, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
            if float(np.linalg.norm(target - actual)) <= float(self.settings.waypoint_settle_pos_tol):
                return
            obs, *_ = self.manager.step(action)
            obs = self.apply_constraints(context) or obs
            recorder.record(obs, action, settle_phase)

    def visual_servo_site_to_target(
        self,
        recorder: Any,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        target_quat: Any | None = None,
        *,
        arm_name: str,
        context: ExecutionContext = ExecutionContext(),
    ) -> np.ndarray:
        arm = self.arm_configs[arm_name]
        target = np.asarray(target_pos, dtype=np.float64)
        target_quat_arr = None if target_quat is None else normalize_quat(np.asarray(target_quat, dtype=np.float64))
        current_action = np.asarray(action, dtype=np.float64).copy()
        servo_phase = f"{phase}:visual_servo"
        integral_error = np.zeros(3, dtype=np.float64)
        settings = self.settings.visual_servo
        for iteration in range(max(0, int(settings.max_iters))):
            actual, actual_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
            error = target - actual
            error_norm = float(np.linalg.norm(error))
            rot_error_norm = 0.0
            if target_quat_arr is not None:
                rot_error_norm = self.quat_error_norm(target_quat_arr, actual_quat)
            if error_norm <= float(settings.pos_tol) and (
                target_quat_arr is None or rot_error_norm <= float(settings.rot_tol)
            ):
                self.visual_servo_events.append(
                    {
                        "phase": phase,
                        "site": site_name,
                        "iterations": iteration,
                        "final_error_norm": error_norm,
                        "final_rot_error_norm": rot_error_norm,
                        "converged": True,
                        "integral_gain": settings.integral_gain,
                    }
                )
                return current_action

            gripper_pos, gripper_quat = site_pose(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                arm.actuator_site,
            )
            target_gripper_quat = gripper_quat
            if target_quat_arr is not None:
                local_site_quat = normalize_quat(quat_multiply(quat_conjugate(gripper_quat), actual_quat))
                target_gripper_quat = normalize_quat(quat_multiply(target_quat_arr, quat_conjugate(local_site_quat)))
            integral_error += error
            correction = error * float(settings.gain) + integral_error * float(settings.integral_gain)
            correction_norm = float(np.linalg.norm(correction))
            max_correction = float(settings.max_correction)
            if max_correction > 0.0 and correction_norm > max_correction:
                correction *= max_correction / correction_norm
            correction_action = self.solve_gripper_servo_action(
                gripper_pos + correction,
                target_gripper_quat,
                current_action,
                arm_name=arm_name,
            )
            self.move_action(recorder, correction_action, settings.steps, servo_phase, context)
            current_action = correction_action

        actual, actual_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
        final_error_norm = float(np.linalg.norm(target - actual))
        final_rot_error_norm = 0.0 if target_quat_arr is None else self.quat_error_norm(target_quat_arr, actual_quat)
        self.visual_servo_events.append(
            {
                "phase": phase,
                "site": site_name,
                "iterations": int(settings.max_iters),
                "final_error_norm": final_error_norm,
                "final_rot_error_norm": final_rot_error_norm,
                "converged": final_error_norm <= float(settings.pos_tol)
                and (target_quat_arr is None or final_rot_error_norm <= float(settings.rot_tol)),
                "integral_gain": settings.integral_gain,
            }
        )
        return current_action

    def solve_gripper_servo_action(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        base_action: np.ndarray,
        *,
        arm_name: str,
    ) -> np.ndarray:
        arm = self.arm_configs[arm_name]
        joint_names = tuple(arm.joint_names)
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]

        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        try:
            result = solve_site_ik(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                arm.actuator_site,
                joint_names,
                target_pos,
                target_quat,
                max_iters=self.ik.max_iters,
                pos_tol=min(float(self.ik.pos_tol), float(self.settings.visual_servo.pos_tol)),
                rot_tol=self.ik.rot_tol,
                damping=self.ik.damping,
            )
            action = np.asarray(base_action, dtype=np.float64).copy()
            for action_id, qpos_id in zip(arm_actuator_ids, qpos_ids, strict=True):
                action[action_id] = result.qpos[qpos_id]
            return action
        finally:
            self.env.data.qpos[:] = start_qpos
            self.env.data.qvel[:] = start_qvel
            self.env.data.ctrl[:] = start_ctrl
            self.env.mujoco.mj_forward(self.env.model, self.env.data)

    def apply_constraints(self, context: ExecutionContext) -> dict[str, Any] | None:
        constrained = False
        for fixed in context.fixed_joint_states:
            restore_free_joint_state(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                fixed.joint_name,
                fixed.state,
            )
            constrained = True
        for attachment in context.attachments:
            apply_site_attachment(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                attachment.joint_name,
                attachment.parent_site,
                attachment.to_mapping(),
            )
            constrained = True
        return self.env.get_observation() if constrained else None

    def record_site_target_error(self, phase: str, site_name: str, target_pos: Any) -> None:
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

    @staticmethod
    def quat_error_norm(target_quat: np.ndarray, actual_quat: np.ndarray) -> float:
        delta = normalize_quat(quat_multiply(target_quat, quat_conjugate(actual_quat)))
        vec_norm = float(np.linalg.norm(delta[1:]))
        if vec_norm < 1e-12:
            return 0.0
        return float(2.0 * np.arctan2(vec_norm, abs(float(delta[0]))))

    def _controlled_site(self, item: PlannedTaskTarget) -> str:
        if item.target.controlled_site:
            return item.target.controlled_site
        return self.arm_configs[item.target.arm].actuator_site
