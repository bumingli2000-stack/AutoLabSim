"""Task-space target construction for the bimanual screw-cap task.

This module describes *where* the tube and cap grippers should move during the
non-rotational stages of the workflow. It deliberately does not:

- solve inverse kinematics;
- execute trajectories;
- maintain rigid attachments;
- update :class:`ScrewCapSystem`;
- construct the ratchet/unscrew sequence;
- serialize episode metadata.

The dedicated screw/ratchet target generation will remain in the rotational
execution module because it is coupled to cumulative twist and release state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from autolabsim.math3d import gripper_quat_from_axes, normalize_quat, unit
from autolabsim.planner import TaskTargetPlanner
from autolabsim.scene import site_pose
from autolabsim.task_target import FrameRef, GripperCommand, TaskTarget


ArmDefaults = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class CapLiftTargets:
    """Sparse cap-grasp targets before Cartesian densification.

    The order is fixed and matches the physical workflow:

    ``pregrasp -> grasp -> post``

    ``post`` lifts the cap and tube assembly after the cap gripper has closed.
    """

    pregrasp: TaskTarget
    grasp: TaskTarget
    post: TaskTarget

    def ordered(self) -> tuple[TaskTarget, ...]:
        return (self.pregrasp, self.grasp, self.post)


@dataclass(frozen=True)
class TubeGraspTargets:
    """Sparse tube-side-grasp targets.

    The current task uses ``pregrasp`` and ``grasp`` while the cap arm is still
    holding the tube-cap assembly. ``lift`` is preserved because the original
    target builder exposed it and it may be useful for later task variants.
    """

    pregrasp: TaskTarget
    grasp: TaskTarget
    lift: TaskTarget

    def approach(self) -> tuple[TaskTarget, ...]:
        """Return only the targets used by the current side-grasp stage."""

        return (self.pregrasp, self.grasp)

    def ordered(self) -> tuple[TaskTarget, ...]:
        return (self.pregrasp, self.grasp, self.lift)


@dataclass(frozen=True)
class CapPlaceTargets:
    """Targets used to move a released cap to the table."""

    clearance: TaskTarget
    preplace: TaskTarget
    place: TaskTarget

    def ordered(self) -> tuple[TaskTarget, ...]:
        return (self.clearance, self.preplace, self.place)


@dataclass(frozen=True)
class TubeReturnTargets:
    """Targets used to return the tube to its original rack slot."""

    preplace: TaskTarget
    place: TaskTarget

    def ordered(self) -> tuple[TaskTarget, ...]:
        return (self.preplace, self.place)


class ScrewCapTargetBuilder:
    """Build non-rotational ``TaskTarget`` objects for the screw-cap task.

    Parameters
    ----------
    env:
        Active MuJoCo environment. It is read only when the current gripper
        pose is required for Cartesian densification or tube return.
    planner:
        Shared task-target planner. Only ``resolve`` is used here to transform
        sparse targets into world poses before interpolation; no IK is solved
        by this class.
    arm_defaults:
        Project arm configuration mapping, normally ``ARM_DEFAULTS``.
    runtime:
        Current ``BimanualUnscrewTaskConfig`` instance. ``Any`` is used here
        temporarily to avoid importing the task module and creating a circular
        dependency while the refactor is still in progress.

    Notes
    -----
    This builder preserves the original task geometry. The first refactor pass
    changes module boundaries only; it does not convert every world-space
    target into object-local coordinates. That geometric cleanup can be done
    separately after regression behavior is stable.
    """

    def __init__(
        self,
        env: Any,
        planner: TaskTargetPlanner,
        arm_defaults: ArmDefaults,
        runtime: Any,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.mujoco = env.mujoco
        self.planner = planner
        self.arm_defaults = arm_defaults
        self.runtime = runtime

    # ------------------------------------------------------------------
    # Public target groups
    # ------------------------------------------------------------------

    def cap_lift_targets(self, cap_pos: np.ndarray) -> CapLiftTargets:
        """Construct cap pregrasp, grasp, and post-lift targets.

        The cap gripper approaches along ``cap_approach_axis``. The grasp point
        is the cap body position plus ``cap_offset``. After closing, ``post``
        moves the gripper by ``cap_post_offset`` so the tube-cap assembly is
        lifted out of the rack.
        """

        approach = unit(
            np.asarray(self.runtime.cap_approach_axis, dtype=np.float64),
            "cap_approach_axis",
        )
        closing = unit(
            self._arm(self.runtime.cap_arm)["closing_axis"],
            "cap_closing_axis",
        )
        quat = gripper_quat_from_axes(
            self.mujoco,
            approach,
            closing,
            self.runtime.cap_tool_roll,
        )

        grasp_pos = (
            np.asarray(cap_pos, dtype=np.float64)
            + np.asarray(self.runtime.cap_offset, dtype=np.float64)
        )
        pregrasp_pos = (
            grasp_pos - approach * float(self.runtime.cap_pregrasp_distance)
        )
        post_pos = (
            grasp_pos
            + np.asarray(self.runtime.cap_post_offset, dtype=np.float64)
        )

        return CapLiftTargets(
            pregrasp=self.world_gripper_target(
                "cap_pregrasp",
                pregrasp_pos,
                quat,
                self.runtime.cap_arm,
                self.runtime.open_gripper,
                servo_mode="none",
            ),
            grasp=self.world_gripper_target(
                "cap_grasp",
                grasp_pos,
                quat,
                self.runtime.cap_arm,
                self.runtime.open_gripper,
                servo_mode="pose",
            ),
            post=self.world_gripper_target(
                "cap_post",
                post_pos,
                quat,
                self.runtime.cap_arm,
                self.runtime.cap_close_gripper,
                servo_mode="pose",
            ),
        )

    def tube_grasp_targets(self, tube_pos: np.ndarray) -> TubeGraspTargets:
        """Construct side pregrasp, grasp, and optional lift targets.

        The approach and closing axes come from the configured tube arm. The
        forward/outward offsets allow the gripper center to be tuned without
        changing the object position or arm definition.
        """

        tube_arm = self._arm(self.runtime.tube_arm)
        approach = unit(
            tube_arm["approach_axis"],
            "tube_approach_axis",
        )
        closing = unit(
            tube_arm["closing_axis"],
            "tube_closing_axis",
        )
        quat = gripper_quat_from_axes(
            self.mujoco,
            approach,
            closing,
            self.runtime.tube_tool_roll,
        )

        grasp_pos = (
            np.asarray(tube_pos, dtype=np.float64)
            + np.asarray(
                [0.0, 0.0, self.runtime.tube_grasp_height],
                dtype=np.float64,
            )
            + approach * float(self.runtime.tube_pinch_forward_offset)
            - approach * float(self.runtime.tube_grasp_outward_offset)
        )
        pregrasp_pos = (
            grasp_pos - approach * float(self.runtime.tube_pregrasp_distance)
        )
        lift_pos = (
            grasp_pos
            + np.asarray(self.runtime.tube_lift_offset, dtype=np.float64)
        )

        return TubeGraspTargets(
            pregrasp=self.world_gripper_target(
                "tube_pregrasp",
                pregrasp_pos,
                quat,
                self.runtime.tube_arm,
                self.runtime.open_gripper,
                servo_mode="none",
            ),
            grasp=self.world_gripper_target(
                "tube_grasp",
                grasp_pos,
                quat,
                self.runtime.tube_arm,
                self.runtime.open_gripper,
                servo_mode="none",
            ),
            lift=self.world_gripper_target(
                "tube_lift",
                lift_pos,
                quat,
                self.runtime.tube_arm,
                self.runtime.tube_close_gripper,
                servo_mode="none",
            ),
        )

    def cap_place_targets(
        self,
        current_cap_gripper_pos: np.ndarray,
        current_cap_gripper_quat: np.ndarray,
    ) -> CapPlaceTargets:
        """Construct clearance, preplace, and final table-place targets.

        ``current_cap_gripper_pos`` is the cap gripper site position immediately
        after unscrewing. The clearance target first moves vertically upward to
        avoid sweeping the released cap across nearby geometry.
        """

        approach = unit(
            np.asarray(self.runtime.cap_approach_axis, dtype=np.float64),
            "cap_place_approach_axis",
        )
        closing = unit(
            self._arm(self.runtime.cap_arm)["closing_axis"],
            "cap_place_closing_axis",
        )
        place_quat = gripper_quat_from_axes(
            self.mujoco,
            approach,
            closing,
            self.runtime.cap_tool_roll,
        )

        place_pos = np.asarray(
            self.runtime.cap_place_pos,
            dtype=np.float64,
        )
        current_quat = normalize_quat(
            np.asarray(
                current_cap_gripper_quat,
                dtype=np.float64,
            )
        )
        preplace_pos = place_pos + np.asarray(
            [0.0, 0.0, self.runtime.cap_place_lift],
            dtype=np.float64,
        )
        clearance_pos = np.asarray(
            current_cap_gripper_pos,
            dtype=np.float64,
        ) + np.asarray(
            [0.0, 0.0, self.runtime.cap_clearance_lift],
            dtype=np.float64,
        )

        return CapPlaceTargets(
            clearance=self.world_gripper_target(
                "cap_clearance_lift",
                clearance_pos,
                current_quat,
                self.runtime.cap_arm,
                self.runtime.cap_close_gripper,
                servo_mode="pose",
            ),
            preplace=self.world_gripper_target(
                "cap_preplace",
                preplace_pos,
                place_quat,
                self.runtime.cap_arm,
                self.runtime.cap_close_gripper,
                servo_mode="none",
            ),
            place=self.world_gripper_target(
                "cap_place",
                place_pos,
                place_quat,
                self.runtime.cap_arm,
                self.runtime.cap_close_gripper,
                servo_mode="none",
            ),
        )

    def tube_return_targets(self, slot_pos: np.ndarray) -> TubeReturnTargets:
        """Construct targets for placing the tube back in its original slot.

        The final gripper position is intentionally derived from the same tube
        grasp geometry used during pickup. This preserves the original task's
        behavior: when the tube free-joint origin returns to ``slot_pos``, the
        tube gripper should return to the corresponding side-grasp position.

        The current gripper orientation is retained during return, matching the
        original implementation and avoiding an unnecessary wrist reorientation.
        """

        return_geometry = self.tube_grasp_targets(
            np.asarray(slot_pos, dtype=np.float64)
        )
        return_pos = np.asarray(
            return_geometry.grasp.pos,
            dtype=np.float64,
        )
        _, current_quat = site_pose(
            self.model,
            self.data,
            self.mujoco,
            self.gripper_site(self.runtime.tube_arm),
        )
        preplace_pos = return_pos + np.asarray(
            [0.0, 0.0, 0.10],
            dtype=np.float64,
        )

        return TubeReturnTargets(
            preplace=self.world_gripper_target(
                "tube_return_preplace",
                preplace_pos,
                current_quat,
                self.runtime.tube_arm,
                self.runtime.tube_close_gripper,
                servo_mode="none",
            ),
            place=self.world_gripper_target(
                "tube_return_place",
                return_pos,
                current_quat,
                self.runtime.tube_arm,
                self.runtime.tube_close_gripper,
                servo_mode="none",
            ),
        )

    # ------------------------------------------------------------------
    # Cartesian densification
    # ------------------------------------------------------------------

    def densify_from_current_site(
        self,
        arm_name: str,
        waypoints: Sequence[TaskTarget],
    ) -> list[TaskTarget]:
        """Densify sparse targets starting at the arm's current gripper pose."""

        start_pos, start_quat = site_pose(
            self.model,
            self.data,
            self.mujoco,
            self.gripper_site(arm_name),
        )
        return self.densify_cartesian_waypoints(
            start_pos,
            start_quat,
            waypoints,
        )

    def densify_cartesian_waypoints(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        waypoints: Sequence[TaskTarget],
    ) -> list[TaskTarget]:
        """Interpolate sparse task-space targets into shorter Cartesian steps.

        Each sparse target is first resolved to a world pose. Position uses
        linear interpolation and orientation uses quaternion SLERP. Intermediate
        targets receive ``_<name>_cart_XX`` names, while the final sample keeps
        the original target name so stage code can still identify key points.

        The number of samples is based on translation distance, preserving the
        original implementation. A pure orientation change still receives at
        least ``cartesian_min_steps`` samples.
        """

        dense: list[TaskTarget] = []
        current_pos = np.asarray(start_pos, dtype=np.float64)
        current_quat = normalize_quat(
            np.asarray(start_quat, dtype=np.float64)
        )

        for waypoint in waypoints:
            resolved = self.planner.resolve(waypoint)
            target_pos = np.asarray(resolved.pos, dtype=np.float64)
            target_quat = normalize_quat(
                np.asarray(resolved.quat, dtype=np.float64)
            )

            distance = float(np.linalg.norm(target_pos - current_pos))
            quaternion_delta = 1.0 - abs(
                float(np.dot(current_quat, target_quat))
            )

            if distance < 1e-8 and quaternion_delta < 1e-8:
                steps = 1
            else:
                steps = max(
                    int(self.runtime.cartesian_min_steps),
                    int(
                        np.ceil(
                            distance
                            / max(
                                1e-6,
                                float(self.runtime.cartesian_step_size),
                            )
                        )
                    ),
                )

            for step in range(1, steps + 1):
                alpha = step / steps
                name = (
                    waypoint.name
                    if step == steps
                    else f"{waypoint.name}_cart_{step:02d}"
                )
                dense.append(
                    TaskTarget(
                        name=name,
                        parent=FrameRef("world"),
                        pos=tuple(
                            (
                                (1.0 - alpha) * current_pos
                                + alpha * target_pos
                            ).tolist()
                        ),
                        quat_wxyz=tuple(
                            self.quat_slerp(
                                current_quat,
                                target_quat,
                                alpha,
                            ).tolist()
                        ),
                        arm=waypoint.arm,
                        controlled_site=waypoint.controlled_site,
                        servo_mode=waypoint.servo_mode,
                        gripper=waypoint.gripper,
                    )
                )

            current_pos = target_pos
            current_quat = target_quat

        return dense

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def world_gripper_target(
        self,
        name: str,
        pos: np.ndarray,
        quat: np.ndarray,
        arm_name: str,
        gripper_value: float,
        *,
        servo_mode: str = "pose",
    ) -> TaskTarget:
        """Create a world-frame target for one arm's gripper site."""

        return TaskTarget(
            name=name,
            parent=FrameRef("world"),
            pos=tuple(np.asarray(pos, dtype=np.float64).tolist()),
            quat_wxyz=tuple(
                normalize_quat(
                    np.asarray(quat, dtype=np.float64)
                ).tolist()
            ),
            arm=arm_name,
            controlled_site=self.gripper_site(arm_name),
            servo_mode=servo_mode,
            gripper=GripperCommand(
                float(gripper_value),
                timing="during",
                steps=int(self.runtime.close_steps),
            ),
        )

    def gripper_site(self, arm_name: str) -> str:
        """Return the configured gripper site name for an arm."""

        return str(self._arm(arm_name)["gripper_site"])

    @staticmethod
    def quat_slerp(
        start_quat: np.ndarray,
        target_quat: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Spherically interpolate two ``wxyz`` quaternions."""

        start = normalize_quat(
            np.asarray(start_quat, dtype=np.float64)
        )
        target = normalize_quat(
            np.asarray(target_quat, dtype=np.float64)
        )

        dot = float(np.dot(start, target))
        if dot < 0.0:
            # q and -q represent the same rotation. Flipping the target chooses
            # the shorter interpolation arc and avoids an unnecessary full turn.
            target = -target
            dot = -dot

        if dot > 0.9995:
            # Very close rotations are numerically better handled by normalized
            # linear interpolation than by dividing by a very small sin(theta).
            return normalize_quat(
                (1.0 - alpha) * start + alpha * target
            )

        theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta = float(np.sin(theta))
        lhs = np.sin((1.0 - alpha) * theta) / sin_theta
        rhs = np.sin(alpha * theta) / sin_theta
        return normalize_quat(lhs * start + rhs * target)

    def _arm(self, arm_name: str) -> Mapping[str, Any]:
        try:
            return self.arm_defaults[arm_name]
        except KeyError as exc:
            raise KeyError(f"Unknown arm name: {arm_name!r}") from exc