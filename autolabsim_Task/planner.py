"""TaskTarget planner: resolve targets, walk attachment chains, and solve IK."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .ik import solve_site_ik
from .math3d import normalize_quat, quat_conjugate, quat_multiply, quat_to_mat
from .motion_context import (
    ArmMotionConfig,
    GripperSettings,
    IKSettings,
    KinematicBinding,
    PlanningContext,
    SiteAttachment,
)
from .scene import actuator_id, free_joint_pose, joint_qpos_ids, site_pose
from .task_target import (
    GripperCommand,
    PlannedTaskTarget,
    ResolvedTaskTarget,
    TaskTarget,
    TaskTargetResolver,
    gripper_command_to_actuator,
)


class IKPlanningError(RuntimeError):
    """Raised when a TaskTarget cannot be converted into a valid IK solution."""

    def __init__(
        self,
        *,
        target_name: str,
        arm: str,
        position_error: float,
        rotation_error: float,
    ) -> None:
        self.target_name = target_name
        self.arm = arm
        self.position_error = float(position_error)
        self.rotation_error = float(rotation_error)
        super().__init__(
            f"IK failed for target {target_name!r} on arm {arm!r}: "
            f"position_error={self.position_error:.6g}, "
            f"rotation_error={self.rotation_error:.6g}"
        )


def joint_pose_for_site_target(
    model: Any,
    data: Any,
    mujoco: Any,
    joint_name: str,
    site_name: str,
    target_site_pos: np.ndarray,
    target_site_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a free-joint pose that places one of its sites at a target pose."""

    joint_pos, joint_quat = free_joint_pose(model, data, mujoco, joint_name)
    site_pos, site_quat = site_pose(model, data, mujoco, site_name)
    joint_mat = quat_to_mat(joint_quat)
    local_site_pos = joint_mat.T @ (site_pos - joint_pos)
    local_site_quat = normalize_quat(quat_multiply(quat_conjugate(joint_quat), site_quat))
    target_joint_quat = normalize_quat(
        quat_multiply(target_site_quat, quat_conjugate(local_site_quat))
    )
    target_joint_pos = (
        np.asarray(target_site_pos, dtype=np.float64)
        - quat_to_mat(target_joint_quat) @ local_site_pos
    )
    return target_joint_pos, target_joint_quat


def parent_site_pose_for_joint_pose(
    target_joint_pos: np.ndarray,
    target_joint_quat: np.ndarray,
    attachment: SiteAttachment,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a captured site attachment to obtain the parent-site target pose."""

    local_quat = normalize_quat(np.asarray(attachment.local_quat, dtype=np.float64))
    parent_site_quat = normalize_quat(
        quat_multiply(target_joint_quat, quat_conjugate(local_quat))
    )
    parent_site_pos = (
        np.asarray(target_joint_pos, dtype=np.float64)
        - quat_to_mat(parent_site_quat)
        @ np.asarray(attachment.local_pos, dtype=np.float64)
    )
    return parent_site_pos, parent_site_quat


def site_quat_for_joint_quat(
    model: Any,
    data: Any,
    mujoco: Any,
    joint_name: str,
    site_name: str,
    target_joint_quat: np.ndarray,
) -> np.ndarray:
    """Return the site orientation produced by a requested free-joint orientation."""

    _, joint_quat = free_joint_pose(model, data, mujoco, joint_name)
    _, site_quat = site_pose(model, data, mujoco, site_name)
    local_site_quat = normalize_quat(
        quat_multiply(quat_conjugate(joint_quat), site_quat)
    )
    return normalize_quat(quat_multiply(target_joint_quat, local_site_quat))

'''规划器核心职责：
    TaskTarget
        ↓
    解析世界坐标目标
        ↓
    处理 attachment
        ↓
    计算机械臂末端目标
        ↓
    求解 IK
        ↓
    PlannedTaskTarget
'''
class TaskTargetPlanner:
    def __init__(
        self,
        model: Any,
        data: Any,
        mujoco: Any,
        arm_configs: dict[str, ArmMotionConfig],
        ik: IKSettings,
        gripper: GripperSettings,
    ) -> None:
        self.model = model
        self.data = data
        self.mujoco = mujoco
        self.arm_configs = arm_configs
        self.ik = ik
        self.gripper = gripper
        self.resolver = TaskTargetResolver(model, data, mujoco)

    def resolve(self, target: TaskTarget) -> ResolvedTaskTarget:
        return self.resolver.resolve(target)

    def relative_quat_for_world_quat(self, parent: Any, world_quat: np.ndarray) -> np.ndarray:
        return self.resolver.relative_quat_for_world_quat(parent, world_quat)

    def plan(
        self,
        targets: Sequence[TaskTarget],
        context: PlanningContext,
        *,
        default_gripper_value: float,
    ) -> list[PlannedTaskTarget]:
        """Resolve and solve targets while preserving prior planned actuator commands."""

        start_qpos = self.data.qpos.copy()
        start_qvel = self.data.qvel.copy()
        start_ctrl = self.data.ctrl.copy()
        working_ctrl = start_ctrl.copy()
        plan: list[PlannedTaskTarget] = []

        try:
            for target in targets:
                arm = self.arm_configs[target.arm]
                joint_names = tuple(arm.joint_names)
                qpos_ids = joint_qpos_ids(self.model, self.mujoco, joint_names)
                arm_actuator_ids = [
                    actuator_id(self.model, self.mujoco, name) for name in joint_names
                ]
                gripper_id = actuator_id(
                    self.model,
                    self.mujoco,
                    arm.gripper_actuator,
                )

                resolved, ik_pos, ik_quat, binding = self.target_to_ik_target(
                    target,
                    context,
                )
                result = solve_site_ik(
                    self.model,
                    self.data,
                    self.mujoco,
                    binding.actuator_site,
                    joint_names,
                    ik_pos,
                    ik_quat,
                    max_iters=self.ik.max_iters,
                    pos_tol=self.ik.pos_tol,
                    rot_tol=self.ik.rot_tol,
                    damping=self.ik.damping,
                )
                if not result.success:
                    raise IKPlanningError(
                        target_name=target.name,
                        arm=target.arm,
                        position_error=result.pos_error,
                        rotation_error=result.rot_error,
                    )

                action = working_ctrl.copy()
                for action_id, qpos_id in zip(
                    arm_actuator_ids,
                    qpos_ids,
                    strict=True,
                ):
                    action[action_id] = result.qpos[qpos_id]

                target_gripper_value = self.target_gripper_value(target.gripper)
                action[gripper_id] = float(
                    target_gripper_value
                    if target_gripper_value is not None
                    else default_gripper_value
                )

                plan.append(
                    PlannedTaskTarget(
                        target=target,
                        resolved=resolved,
                        ik_site_pos=np.asarray(ik_pos, dtype=np.float64),
                        ik_site_quat=np.asarray(ik_quat, dtype=np.float64),
                        action=action,
                        ik_success=True,
                        ik_pos_error=float(result.pos_error),
                        ik_rot_error=float(result.rot_error),
                        arm_joint_names=tuple(str(name) for name in joint_names),
                        arm_qpos=np.asarray(result.qpos[qpos_ids], dtype=np.float64),
                        gripper_value=target_gripper_value,
                    )
                )
                working_ctrl = action
        finally:
            self.data.qpos[:] = start_qpos
            self.data.qvel[:] = start_qvel
            self.data.ctrl[:] = start_ctrl
            self.mujoco.mj_forward(self.model, self.data)

        return plan

    def target_to_ik_target(
        self,
        target: TaskTarget,
        context: PlanningContext,
    ) -> tuple[ResolvedTaskTarget, np.ndarray, np.ndarray, KinematicBinding]:
        resolved = self.resolve(target)
        arm = self.arm_configs[target.arm]
        controlled_site = target.controlled_site or arm.actuator_site
        binding = context.binding_for(
            target.arm,
            controlled_site,
            arm.actuator_site,
        )
        if binding.actuator_site != arm.actuator_site:
            raise ValueError(
                f"Binding for {target.name!r} uses "
                f"actuator_site={binding.actuator_site!r}, but arm "
                f"{target.arm!r} is configured with "
                f"actuator_site={arm.actuator_site!r}"
            )

        site_name = controlled_site
        target_pos = np.asarray(resolved.pos, dtype=np.float64)
        target_quat = normalize_quat(np.asarray(resolved.quat, dtype=np.float64))

        for attachment in reversed(binding.attachments):
            target_joint_pos, target_joint_quat = joint_pose_for_site_target(
                self.model,
                self.data,
                self.mujoco,
                attachment.joint_name,
                site_name,
                target_pos,
                target_quat,
            )
            target_pos, target_quat = parent_site_pose_for_joint_pose(
                target_joint_pos,
                target_joint_quat,
                attachment,
            )
            site_name = attachment.parent_site

        if site_name != binding.actuator_site:
            raise ValueError(
                f"Attachment chain for target {target.name!r} ended at site "
                f"{site_name!r}, expected actuator site "
                f"{binding.actuator_site!r}"
            )

        return resolved, target_pos, target_quat, binding

    def target_gripper_value(self, command: GripperCommand | None) -> float | None:
        if command is None:
            return None
        return gripper_command_to_actuator(
            command,
            self.gripper.open_value,
            self.gripper.close_value,
        )