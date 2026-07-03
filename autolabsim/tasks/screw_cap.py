'''
双臂旋拧开盖任务
'''
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..ik import solve_site_ik
from ..math3d import gripper_quat_from_axes, normalize_quat, unit
from ..mujoco_env import EnvConfig
from ..recorder import EpisodeRecorder
from ..scene_profile import active_joint_fallback
from ..scene import (
    apply_site_attachment,
    actuator_id,
    body_pos,
    capture_free_joint_state,
    capture_site_attachment,
    free_joint_pos,
    joint_qpos_ids,
    restore_free_joint_state,
    set_free_joint_pose,
    site_pose,
)
from ..screw import ScrewCapSystem
from ..task import AutoLabTask, TaskConfig
from .common import (
    ARM_DEFAULTS,
    cap_body_from_tube_joint,
    cap_joint_from_tube_joint,
    cap_weld_from_tube_joint,
    json_safe,
    random_reset_info,
)


@dataclass(frozen=True)
class BimanualUnscrewTaskConfig:
    env: EnvConfig
    out_dir: Path
    episode_index: int
    seed: int
    cameras: tuple[str, ...] = ("overview_camera",)
    with_images: bool = False
    tube_arm: str = "second"
    cap_arm: str = "first"
    open_gripper: float = 0.0
    close_gripper: float = 255.0
    settle_steps: int = 20
    steps_per_segment: int = 20
    grasp_hold_steps: int = 10
    hold_steps: int = 10
    close_steps: int = 12
    cap_hold_steps: int = 12
    tube_grasp_height: float = 0.09
    tube_pregrasp_distance: float = 0.10
    tube_lift_offset: tuple[float, float, float] = (0.25, 0.0, 0.12)
    tube_pinch_forward_offset: float = 0.02
    tube_grasp_outward_offset: float = 0.02
    tube_tool_roll: float = float(np.pi)
    cap_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cap_pregrasp_distance: float = 0.10
    cap_post_offset: tuple[float, float, float] = (0.0, 0.0, 0.08)
    cap_place_pos: tuple[float, float, float] = (0.18, -3.06, 0.43)
    cap_place_lift: float = 0.08
    cap_clearance_lift: float = 0.1
    cap_tool_roll: float = 0.0
    release_angle: float = float(np.pi * 1.5)
    release_lift: float = 0.008
    thread_pitch: float = 0.008
    unscrew_steps: int = 24
    ratchet_angle: float = float(np.pi / 2.0)
    return_tube_to_rack: bool = True
    ik_max_iters: int = 500
    ik_pos_tol: float = 0.003
    ik_rot_tol: float = 0.05
    ik_damping: float = 0.08


class BimanualUnscrewTask(AutoLabTask):
    name = "bimanual_unscrew_cap"

    def __init__(self, config: BimanualUnscrewTaskConfig):
        self.runtime = config
        self.tube_arm = ARM_DEFAULTS[config.tube_arm]
        self.cap_arm = ARM_DEFAULTS[config.cap_arm]
        self.screw_system: ScrewCapSystem | None = None
        super().__init__(
            TaskConfig(
                env=config.env,
                with_images=config.with_images,
                cameras=config.cameras,
            )
        )

    def run(self) -> dict[str, Any]:
        obs = self.reset()
        reset_info = dict(self.env.last_reset_info)
        random_info = random_reset_info(reset_info)
        active_joint = str(random_info["active_joint"]) if random_info else active_joint_fallback()
        active_cap_joint = cap_joint_from_tube_joint(active_joint)
        active_cap_body = cap_body_from_tube_joint(active_joint)
        active_cap_weld = cap_weld_from_tube_joint(active_joint)
        self.screw_system = ScrewCapSystem(
            tube_joint=active_joint,
            cap_joint=active_cap_joint,
            cap_site=str(self.cap_arm["gripper_site"]),
            weld_name=active_cap_weld,
            release_angle=self.runtime.release_angle,
            thread_pitch=self.runtime.thread_pitch,
            max_lift=self.runtime.release_lift,
        )
        self.manager.systems = [self.screw_system]
        self.screw_system.on_reset(self.env)

        tube_gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.tube_arm["gripper_actuator"]))
        cap_gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.cap_arm["gripper_actuator"]))

        settle_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        settle_action[tube_gripper_id] = self.runtime.open_gripper
        settle_action[cap_gripper_id] = self.runtime.open_gripper
        for _ in range(self.runtime.settle_steps):
            obs, *_ = self.manager.step(settle_action)

        cap_pos = body_pos(self.env.model, self.env.data, self.env.mujoco, active_cap_body)
        recorder = EpisodeRecorder(self.runtime.cameras, self.runtime.with_images)

        cap_waypoints = self._make_cap_waypoints(cap_pos)
        cap_plan = self._plan_arm(self.cap_arm, cap_waypoints, self.runtime.open_gripper)

        recorder.record(self.env.get_observation(), np.asarray(self.env.data.ctrl).copy(), "start")

        initial_tube_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint)
        initial_cap_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_cap_joint)
        initial_object_states = [(active_joint, initial_tube_state), (active_cap_joint, initial_cap_state)]
        self._execute_plan(
            recorder,
            cap_plan[:2],
            "cap_move_to_lift_grasp",
            fixed_joint_states=initial_object_states,
        )
        close_cap = np.asarray(cap_plan[1]["action"]).copy()
        close_cap[cap_gripper_id] = self.runtime.close_gripper
        self._move_action(
            recorder,
            close_cap,
            self.runtime.close_steps,
            "cap_close_for_lift",
            fixed_joint_states=initial_object_states,
        )

        cap_lift_attachments = self._attachments_to_site(
            str(self.cap_arm["gripper_site"]),
            (active_cap_joint, active_joint),
        )
        cap_lift_action = np.asarray(cap_plan[2]["action"]).copy()
        cap_lift_action[cap_gripper_id] = self.runtime.close_gripper
        self._move_action(
            recorder,
            cap_lift_action,
            self.runtime.steps_per_segment,
            "cap_lift_tube_out",
            follow_attachments=cap_lift_attachments,
        )
        self._hold_action(
            recorder,
            cap_lift_action,
            self.runtime.hold_steps,
            "cap_hold_tube_lifted",
            follow_attachments=cap_lift_attachments,
        )

        lifted_tube_pos = free_joint_pos(self.env.model, self.env.data, self.env.mujoco, active_joint)
        tube_waypoints = self._make_tube_waypoints(lifted_tube_pos)
        tube_plan = self._plan_arm(self.tube_arm, tube_waypoints[:2], self.runtime.open_gripper)

        self._execute_plan(
            recorder,
            tube_plan,
            "tube_move_to_side_grasp",
            follow_attachments=cap_lift_attachments,
        )
        close_tube = np.asarray(tube_plan[-1]["action"]).copy()
        close_tube[tube_gripper_id] = self.runtime.close_gripper
        self._move_action(
            recorder,
            close_tube,
            self.runtime.close_steps,
            "tube_close_side_grip",
            follow_attachments=cap_lift_attachments,
        )

        tube_grip_attachment = self._attachments_to_site(str(self.tube_arm["gripper_site"]), (active_joint,))
        held_tube_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint)

        self.screw_system.engage(self.env)
        cap_site_pos, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, str(self.cap_arm["gripper_site"]))
        unscrew_start_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        unscrew_plan = self._plan_unscrew({"pos": cap_site_pos}, unscrew_start_action)
        self._execute_plan(recorder, unscrew_plan, "cap_unscrew", hold_joint=active_joint, held_state=held_tube_state)
        if self.screw_system is not None:
            self.screw_system.start_follow_after_release(self.env)

        cap_place_waypoints = self._make_cap_place_waypoints(cap_site_pos)
        cap_place_plan = self._plan_arm(self.cap_arm, cap_place_waypoints, self.runtime.close_gripper)
        self._execute_plan(recorder, cap_place_plan, "cap_place_on_table", hold_joint=active_joint, held_state=held_tube_state)
        if self.screw_system is not None:
            self.screw_system.release_follow()
        placed_cap_state = capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_cap_joint)
        open_cap = np.asarray(cap_place_plan[-1]["action"]).copy()
        open_cap[cap_gripper_id] = self.runtime.open_gripper
        self._move_action(
            recorder,
            open_cap,
            self.runtime.close_steps,
            "cap_release_on_table",
            hold_joint=active_joint,
            held_state=held_tube_state,
            fixed_joint_states=[(active_cap_joint, placed_cap_state)],
        )

        if self.runtime.return_tube_to_rack:
            slot_pose = random_info.get("pose") if random_info else None
            slot_pos = np.asarray(slot_pose["pos"], dtype=np.float64) if isinstance(slot_pose, dict) else free_joint_pos(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                active_joint,
            )
            slot_quat = np.asarray(slot_pose.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64) if isinstance(slot_pose, dict) else None
            return_waypoints = self._make_tube_return_waypoints(slot_pos)
            return_plan = self._plan_arm(self.tube_arm, return_waypoints, self.runtime.close_gripper)
            self._execute_plan(
                recorder,
                return_plan,
                "tube_return_to_rack",
                follow_attachments=tube_grip_attachment,
                fixed_joint_states=[(active_cap_joint, placed_cap_state)],
            )
            if slot_quat is not None:
                set_free_joint_pose(self.env.model, self.env.data, self.env.mujoco, active_joint, slot_pos, slot_quat)
            open_tube = np.asarray(return_plan[-1]["action"]).copy()
            open_tube[tube_gripper_id] = self.runtime.open_gripper
            self._move_action(
                recorder,
                open_tube,
                self.runtime.close_steps,
                "tube_release_in_rack",
                fixed_joint_states=[(active_cap_joint, placed_cap_state)],
            )
            self._hold_action(
                recorder,
                open_tube,
                self.runtime.hold_steps,
                "tube_hold_released",
                fixed_joint_states=[(active_cap_joint, placed_cap_state)],
            )

        arrays = recorder.to_arrays()
        arrays["final_state"] = self.env.get_observation()["state"]
        metadata = self._make_metadata(
            reset_info,
            random_info,
            active_joint,
            active_cap_joint,
            active_cap_body,
            tube_plan,
            cap_plan,
            unscrew_plan,
            num_steps=arrays["qpos"].shape[0],
        )
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    def _make_tube_waypoints(self, tube_pos: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(self.tube_arm["approach_axis"], "tube_approach_axis")
        closing = unit(self.tube_arm["closing_axis"], "tube_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.tube_tool_roll)
        grasp_pos = (
            tube_pos
            + np.asarray([0.0, 0.0, self.runtime.tube_grasp_height], dtype=np.float64)
            + approach * self.runtime.tube_pinch_forward_offset
            - approach * self.runtime.tube_grasp_outward_offset
        )
        pregrasp_pos = grasp_pos - approach * self.runtime.tube_pregrasp_distance
        lift_pos = grasp_pos + np.asarray(self.runtime.tube_lift_offset, dtype=np.float64)
        return [
            {"name": "tube_pregrasp", "pos": pregrasp_pos, "quat": quat},
            {"name": "tube_grasp", "pos": grasp_pos, "quat": quat},
            {"name": "tube_lift", "pos": lift_pos, "quat": quat},
        ]

    def _make_cap_waypoints(self, cap_pos: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(np.asarray([0.0, 0.0, -1.0], dtype=np.float64), "cap_approach_axis")
        closing = unit(self.cap_arm["closing_axis"], "cap_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.cap_tool_roll)
        grasp_pos = cap_pos + np.asarray(self.runtime.cap_offset, dtype=np.float64)
        pregrasp_pos = grasp_pos - approach * self.runtime.cap_pregrasp_distance
        post_pos = grasp_pos + np.asarray(self.runtime.cap_post_offset, dtype=np.float64)
        return [
            {"name": "cap_pregrasp", "pos": pregrasp_pos, "quat": quat},
            {"name": "cap_grasp", "pos": grasp_pos, "quat": quat},
            {"name": "cap_post", "pos": post_pos, "quat": quat},
        ]

    def _make_cap_place_waypoints(self, current_cap_grasp_pos: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(np.asarray([0.0, 0.0, -1.0], dtype=np.float64), "cap_place_approach_axis")
        closing = unit(self.cap_arm["closing_axis"], "cap_place_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.cap_tool_roll)
        place_pos = np.asarray(self.runtime.cap_place_pos, dtype=np.float64)
        preplace_pos = place_pos + np.asarray([0.0, 0.0, self.runtime.cap_place_lift], dtype=np.float64)
        clear_pos = np.asarray(current_cap_grasp_pos, dtype=np.float64) + np.asarray([0.0, 0.0, self.runtime.cap_clearance_lift], dtype=np.float64)
        return [
            {"name": "cap_clearance_lift", "pos": clear_pos, "quat": quat},
            {"name": "cap_preplace", "pos": preplace_pos, "quat": quat},
            {"name": "cap_place", "pos": place_pos, "quat": quat},
        ]

    def _make_tube_return_waypoints(self, slot_pos: np.ndarray) -> list[dict[str, Any]]:
        return_pos = self._make_tube_waypoints(np.asarray(slot_pos, dtype=np.float64))[1]["pos"]
        _, current_quat = site_pose(self.env.model, self.env.data, self.env.mujoco, str(self.tube_arm["gripper_site"]))
        preplace_pos = return_pos + np.asarray([0.0, 0.0, 0.10], dtype=np.float64)
        return [
            {"name": "tube_return_preplace", "pos": preplace_pos, "quat": current_quat},
            {"name": "tube_return_place", "pos": return_pos, "quat": current_quat},
        ]

    def _attachments_to_site(self, site_name: str, joint_names: tuple[str, ...]) -> list[tuple[str, str, dict[str, np.ndarray]]]:
        return [
            (
                joint_name,
                site_name,
                capture_site_attachment(self.env.model, self.env.data, self.env.mujoco, joint_name, site_name),
            )
            for joint_name in joint_names
        ]

    def _apply_follow_attachments(self, attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None) -> None:
        if not attachments:
            return
        for joint_name, site_name, attachment in attachments:
            apply_site_attachment(self.env.model, self.env.data, self.env.mujoco, joint_name, site_name, attachment)

    def _restore_fixed_joint_states(self, states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None) -> None:
        if not states:
            return
        for joint_name, state in states:
            restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, joint_name, state)

    def _refresh_screw_pose_after_constraints(self, action: np.ndarray, obs: dict[str, Any]) -> dict[str, Any]:
        if self.screw_system is None or not self.screw_system.progress.engaged or self.screw_system.progress.released:
            return obs
        self.screw_system.after_step(self.env, action, obs)
        return self.env.get_observation()

    def _plan_arm(self, arm: dict[str, Any], waypoints: list[dict[str, Any]], gripper_value: float) -> list[dict[str, Any]]:
        joint_names = tuple(arm["joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(arm["gripper_actuator"]))

        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        plan: list[dict[str, Any]] = []
        for waypoint in waypoints:
            result = solve_site_ik(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                str(arm["gripper_site"]),
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
            plan.append(
                {
                    "name": waypoint["name"],
                    "target_pos": waypoint["pos"].tolist(),
                    "target_quat_wxyz": waypoint["quat"].tolist(),
                    "action": action,
                    "ik_success": result.success,
                    "ik_pos_error": result.pos_error,
                    "ik_rot_error": result.rot_error,
                }
            )
        self.env.data.qpos[:] = start_qpos
        self.env.data.qvel[:] = start_qvel
        self.env.data.ctrl[:] = start_ctrl
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        return plan

    def _plan_unscrew(self, grasp_waypoint: dict[str, Any], start_action: np.ndarray) -> list[dict[str, Any]]:
        approach = unit(np.asarray([0.0, 0.0, -1.0], dtype=np.float64), "cap_unscrew_axis")
        closing = unit(self.cap_arm["closing_axis"], "cap_closing_axis")
        pos = np.asarray(grasp_waypoint["pos"], dtype=np.float64)
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.cap_arm["gripper_actuator"]))
        ratchet_angle = max(1e-6, float(self.runtime.ratchet_angle))
        loops = max(1, int(np.ceil(self.runtime.release_angle / ratchet_angle)))
        plan: list[dict[str, Any]] = []
        current_action = np.asarray(start_action, dtype=np.float64).copy()
        accumulated_twist = 0.0

        for loop_id in range(1, loops + 1):
            segment_angle = min(ratchet_angle, self.runtime.release_angle - accumulated_twist)
            twist_angle = accumulated_twist + segment_angle
            rotate_quat = gripper_quat_from_axes(
                self.env.mujoco,
                approach,
                closing,
                self.runtime.cap_tool_roll - segment_angle,
            )
            rotate_plan = self._plan_arm(
                self.cap_arm,
                [
                    {
                        "name": f"cap_ratchet_twist_{loop_id:02d}",
                        "pos": pos,
                        "quat": normalize_quat(rotate_quat),
                    }
                ],
                self.runtime.close_gripper,
            )[0]
            rotate_plan["twist_angle"] = float(twist_angle)
            plan.append(rotate_plan)
            current_action = np.asarray(rotate_plan["action"], dtype=np.float64).copy()
            accumulated_twist = twist_angle

            if twist_angle >= self.runtime.release_angle - 1e-9:
                continue

            open_action = current_action.copy()
            open_action[gripper_id] = self.runtime.open_gripper
            plan.append(
                {
                    "name": f"cap_ratchet_open_{loop_id:02d}",
                    "target_pos": pos.tolist(),
                    "target_quat_wxyz": rotate_quat.tolist(),
                    "action": open_action,
                    "ik_success": True,
                    "ik_pos_error": 0.0,
                    "ik_rot_error": 0.0,
                    "twist_angle": float(twist_angle),
                    "steps": self.runtime.close_steps,
                }
            )

            rewind_quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.cap_tool_roll)
            rewind_plan = self._plan_arm(
                self.cap_arm,
                [
                    {
                        "name": f"cap_ratchet_rewind_{loop_id:02d}",
                        "pos": pos,
                        "quat": normalize_quat(rewind_quat),
                    }
                ],
                self.runtime.open_gripper,
            )[0]
            rewind_plan["twist_angle"] = float(twist_angle)
            plan.append(rewind_plan)

            close_action = np.asarray(rewind_plan["action"], dtype=np.float64).copy()
            close_action[gripper_id] = self.runtime.close_gripper
            plan.append(
                {
                    "name": f"cap_ratchet_regrip_{loop_id:02d}",
                    "target_pos": pos.tolist(),
                    "target_quat_wxyz": rewind_quat.tolist(),
                    "action": close_action,
                    "ik_success": True,
                    "ik_pos_error": 0.0,
                    "ik_rot_error": 0.0,
                    "twist_angle": float(twist_angle),
                    "steps": self.runtime.close_steps,
                }
            )
            current_action = close_action
        return plan

    def _execute_plan(
        self,
        recorder: EpisodeRecorder,
        plan: list[dict[str, Any]],
        phase_prefix: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    ) -> None:
        for item in plan:
            self._move_action(
                recorder,
                np.asarray(item["action"]),
                int(item.get("steps", self.runtime.steps_per_segment)),
                f"{phase_prefix}:{item['name']}",
                hold_joint=hold_joint,
                held_state=held_state,
                follow_attachments=follow_attachments,
                fixed_joint_states=fixed_joint_states,
                twist_target=item.get("twist_angle"),
            )

    def _move_action(
        self,
        recorder: EpisodeRecorder,
        target_action: np.ndarray,
        steps: int,
        phase: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        twist_target: float | None = None,
    ) -> None:
        start = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        denom = max(1, steps)
        start_twist = self.screw_system.progress.twist_angle if self.screw_system is not None else 0.0
        for step in range(1, denom + 1):
            alpha = step / denom
            action = (1.0 - alpha) * start + alpha * target_action
            if self.screw_system is not None and twist_target is not None:
                self.screw_system.set_commanded_twist(start_twist + alpha * (twist_target - start_twist))
            obs, *_ = self.manager.step(action)
            constrained = False
            if hold_joint is not None and held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
                constrained = True
            if follow_attachments:
                self._apply_follow_attachments(follow_attachments)
                obs = self.env.get_observation()
                constrained = True
            if fixed_joint_states:
                self._restore_fixed_joint_states(fixed_joint_states)
                obs = self.env.get_observation()
                constrained = True
            if constrained:
                obs = self._refresh_screw_pose_after_constraints(action, obs)
            recorder.record(obs, action, phase)
        if self.screw_system is not None and twist_target is not None:
            self.screw_system.set_commanded_twist(twist_target)

    def _hold_action(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        steps: int,
        phase: str,
        *,
        hold_joint: str | None = None,
        held_state: tuple[np.ndarray, np.ndarray] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    ) -> None:
        for _ in range(max(0, steps)):
            obs, *_ = self.manager.step(action)
            constrained = False
            if hold_joint is not None and held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
                constrained = True
            if follow_attachments:
                self._apply_follow_attachments(follow_attachments)
                obs = self.env.get_observation()
                constrained = True
            if fixed_joint_states:
                self._restore_fixed_joint_states(fixed_joint_states)
                obs = self.env.get_observation()
                constrained = True
            if constrained:
                obs = self._refresh_screw_pose_after_constraints(action, obs)
            recorder.record(obs, action, phase)

    def _make_metadata(
        self,
        reset_info: dict[str, Any],
        random_info: dict[str, Any] | None,
        active_joint: str,
        active_cap_joint: str,
        active_cap_body: str,
        tube_plan: list[dict[str, Any]],
        cap_plan: list[dict[str, Any]],
        unscrew_plan: list[dict[str, Any]],
        *,
        num_steps: int,
    ) -> dict[str, Any]:
        final_obs = self.env.get_observation()
        metadata = {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "task": self.name,
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "tube_arm": self.runtime.tube_arm,
            "cap_arm": self.runtime.cap_arm,
            "active_joint": active_joint,
            "cap_joint": active_cap_joint,
            "cap_body": active_cap_body,
            "slot_index": random_info.get("slot_index") if random_info else None,
            "slot_name": random_info.get("slot_name") if random_info else None,
            "reset_info": reset_info,
            "tube_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in tube_plan],
            "cap_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in cap_plan],
            "unscrew_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in unscrew_plan],
            "screw_progress": {
                "released": bool(self.screw_system.progress.released if self.screw_system else False),
                "twist_angle": float(self.screw_system.progress.twist_angle if self.screw_system else 0.0),
                "lift_distance": float(self.screw_system.progress.lift_distance if self.screw_system else 0.0),
                "release_angle_target": float(self.runtime.release_angle),
            },
            "final_time": float(final_obs["time"]),
            "final_state_summary": json_safe(
                {
                    "tube_pos": free_joint_pos(self.env.model, self.env.data, self.env.mujoco, active_joint),
                    "cap_pos": body_pos(self.env.model, self.env.data, self.env.mujoco, active_cap_body),
                }
            ),
        }
        return metadata
