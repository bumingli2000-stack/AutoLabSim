'''
单臂夹起移液枪任务
'''
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..ik import solve_site_ik
from ..math3d import gripper_quat_from_axes, normalize_quat, quat_conjugate, quat_multiply, quat_to_mat, unit
from ..mujoco_env import EnvConfig
from ..recorder import EpisodeRecorder
from ..scene import (
    actuator_id,
    body_pos,
    capture_free_joint_state,
    capture_site_attachment,
    equality_id,
    free_joint_pose,
    joint_qpos_ids,
    restore_free_joint_state,
    site_pose,
)
from ..task import AutoLabTask, TaskConfig
from .common import ARM_DEFAULTS, json_safe


@dataclass(frozen=True)
class PipetteGraspTaskConfig:
    env: EnvConfig
    out_dir: Path
    episode_index: int
    seed: int
    cameras: tuple[str, ...] = ("overview_camera",)
    with_images: bool = False
    arm: str = "first"
    open_gripper: float = 0.0
    close_gripper: float = 180.0
    settle_steps: int = 20
    steps_per_segment: int = 50
    close_steps: int = 12
    hold_steps: int = 20
    grasp_hold_steps: int = 8
    pregrasp_distance: float = 0.0
    grasp_offset: tuple[float, float, float] = (-0.01, 0.0, 0.17)
    lift_offset: tuple[float, float, float] = (0.0, 0.0, 0.08)
    lift_retry_fractions: tuple[float, ...] = (1.0, 0.75, 0.5)
    tool_roll: float = 0.0
    pipette_joint: str = "pipette_joint"
    pipette_body: str = "pippipette"
    parking_weld: str = "pipette_rack_weld"
    vertical_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    ik_max_iters: int = 800
    ik_pos_tol: float = 0.008
    ik_rot_tol: float = 0.02
    ik_damping: float = 0.01
    waypoint_settle_steps: int = 20
    waypoint_settle_pos_tol: float = 0.008


class PipetteGraspTask(AutoLabTask):
    name = "pipette_grasp"

    def __init__(self, config: PipetteGraspTaskConfig):
        self.runtime = config
        self.arm = ARM_DEFAULTS[config.arm]
        self.execution_site_errors: list[dict[str, Any]] = []
        super().__init__(
            TaskConfig(
                env=config.env,
                with_images=config.with_images,
                cameras=config.cameras,
            )
        )

    def run(self) -> dict[str, Any]:
        obs = self.reset()
        recorder = EpisodeRecorder(cameras=self.runtime.cameras, with_images=self.runtime.with_images)
        action = np.asarray(obs["ctrl"], dtype=np.float64).copy()
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.arm["gripper_actuator"]))
        action[gripper_id] = self.runtime.open_gripper

        initial_pipette_state = capture_free_joint_state(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_joint,
        )
        self._hold_action(
            recorder,
            action,
            self.runtime.settle_steps,
            "settle",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
        )

        grasp_plan = self._plan_grasp_waypoints(self.runtime.open_gripper)
        self._execute_plan(
            recorder,
            grasp_plan,
            "move_to_pipette",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
            debug_site=str(self.arm["gripper_site"]),
        )

        grasp_action = np.asarray(grasp_plan[-1]["action"], dtype=np.float64).copy()
        self._hold_action(
            recorder,
            grasp_action,
            self.runtime.grasp_hold_steps,
            "hold_at_grasp",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
        )

        close_action = grasp_action.copy()
        close_action[gripper_id] = self.runtime.close_gripper
        self._move_action(
            recorder,
            close_action,
            self.runtime.close_steps,
            "close_gripper",
            fixed_joint_states=[(self.runtime.pipette_joint, initial_pipette_state)],
        )
        self._set_equality_active_if_exists(self.runtime.parking_weld, False)

        attachment = capture_site_attachment(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_joint,
            str(self.arm["gripper_site"]),
        )
        lift_plan = self._plan_lift_waypoint(attachment, close_action)
        self._execute_plan(
            recorder,
            lift_plan,
            "lift_pipette_vertical",
            follow_attachments=[(self.runtime.pipette_joint, str(self.arm["gripper_site"]), attachment)],
            debug_site=str(self.arm["gripper_site"]),
        )
        self._hold_action(
            recorder,
            np.asarray(lift_plan[-1]["action"], dtype=np.float64),
            self.runtime.hold_steps,
            "hold_lifted",
            follow_attachments=[(self.runtime.pipette_joint, str(self.arm["gripper_site"]), attachment)],
        )

        arrays = recorder.to_arrays()
        metadata = self._make_metadata(grasp_plan, lift_plan, num_steps=arrays["qpos"].shape[0])
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    def _plan_grasp_waypoints(self, gripper_value: float) -> list[dict[str, Any]]:
        grasp_pos = body_pos(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_body) + np.asarray(
            self.runtime.grasp_offset,
            dtype=np.float64,
        )
        approach = unit(np.asarray(self.arm["approach_axis"], dtype=np.float64), "pipette_approach_axis")
        closing = unit(np.asarray(self.arm["closing_axis"], dtype=np.float64), "pipette_closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.tool_roll)
        waypoints = []
        if self.runtime.pregrasp_distance > 0.0:
            pregrasp_pos = grasp_pos - approach * self.runtime.pregrasp_distance
            waypoints.append({"name": "pipette_pregrasp", "pos": pregrasp_pos, "quat": quat})
        waypoints.append({"name": "pipette_grasp", "pos": grasp_pos, "quat": quat})
        return self._plan_arm(waypoints, gripper_value)

    def _plan_lift_waypoint(self, attachment: dict[str, np.ndarray], start_action: np.ndarray) -> list[dict[str, Any]]:
        joint_pos, _ = free_joint_pose(self.env.model, self.env.data, self.env.mujoco, self.runtime.pipette_joint)
        lift_offset = np.asarray(self.runtime.lift_offset, dtype=np.float64)
        target_joint_quat = normalize_quat(np.asarray(self.runtime.vertical_quat, dtype=np.float64))
        local_quat = normalize_quat(np.asarray(attachment["local_quat"], dtype=np.float64))
        best_plan: list[dict[str, Any]] | None = None
        best_score = float("inf")
        retry_fractions = tuple(float(item) for item in self.runtime.lift_retry_fractions if float(item) > 0.0)
        for fraction in retry_fractions or (1.0,):
            target_joint_pos = joint_pos + lift_offset * fraction
            site_quat = normalize_quat(quat_multiply(target_joint_quat, quat_conjugate(local_quat)))
            site_pos = target_joint_pos - quat_to_mat(site_quat) @ np.asarray(attachment["local_pos"], dtype=np.float64)
            plan = self._plan_arm(
                [{"name": "pipette_lift_vertical", "pos": site_pos, "quat": site_quat}],
                self.runtime.close_gripper,
            )
            plan[0]["action"] = np.asarray(plan[0]["action"], dtype=np.float64)
            plan[0]["lift_fraction"] = fraction
            score = float(plan[0]["ik_pos_error"]) + 0.1 * float(plan[0]["ik_rot_error"])
            if best_plan is None or score < best_score:
                best_plan = plan
                best_score = score
            if plan[0]["ik_success"]:
                return plan

        if best_plan is None:
            raise RuntimeError("No pipette lift waypoint could be planned")
        return best_plan

    def _plan_arm(self, waypoints: list[dict[str, Any]], gripper_value: float) -> list[dict[str, Any]]:
        joint_names = tuple(self.arm["joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.arm["gripper_actuator"]))

        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        plan: list[dict[str, Any]] = []
        for waypoint in waypoints:
            result = solve_site_ik(
                self.env.model,
                self.env.data,
                self.env.mujoco,
                str(self.arm["gripper_site"]),
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
                    "target_pos": np.asarray(waypoint["pos"], dtype=np.float64).tolist(),
                    "target_quat_wxyz": np.asarray(waypoint["quat"], dtype=np.float64).tolist(),
                    "action": action,
                    "ik_success": result.success,
                    "ik_pos_error": result.pos_error,
                    "ik_rot_error": result.rot_error,
                    "arm_joint_names": list(joint_names),
                    "arm_qpos": result.qpos[qpos_ids].tolist(),
                }
            )

        self.env.data.qpos[:] = start_qpos
        self.env.data.qvel[:] = start_qvel
        self.env.data.ctrl[:] = start_ctrl
        self.env.mujoco.mj_forward(self.env.model, self.env.data)
        return plan

    def _execute_plan(
        self,
        recorder: EpisodeRecorder,
        plan: list[dict[str, Any]],
        phase_prefix: str,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
        debug_site: str | None = None,
    ) -> None:
        for item in plan:
            phase = f"{phase_prefix}:{item['name']}"
            self._move_action(
                recorder,
                np.asarray(item["action"], dtype=np.float64),
                self.runtime.steps_per_segment,
                phase,
                fixed_joint_states=fixed_joint_states,
                follow_attachments=follow_attachments,
            )
            if debug_site is not None:
                self._settle_until_site_reached(
                    recorder,
                    np.asarray(item["action"], dtype=np.float64),
                    phase,
                    debug_site,
                    item["target_pos"],
                    fixed_joint_states=fixed_joint_states,
                    follow_attachments=follow_attachments,
                )
                self._record_site_target_error(phase, debug_site, item["target_pos"])

    def _move_action(
        self,
        recorder: EpisodeRecorder,
        target_action: np.ndarray,
        steps: int,
        phase: str,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> None:
        start = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        denom = max(1, int(steps))
        for step in range(1, denom + 1):
            alpha = step / denom
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            action = (1.0 - alpha) * start + alpha * target_action
            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints(fixed_joint_states, follow_attachments) or obs
            recorder.record(obs, action, phase)

    def _hold_action(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        steps: int,
        phase: str,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> None:
        for _ in range(max(0, int(steps))):
            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints(fixed_joint_states, follow_attachments) or obs
            recorder.record(obs, action, phase)

    def _settle_until_site_reached(
        self,
        recorder: EpisodeRecorder,
        action: np.ndarray,
        phase: str,
        site_name: str,
        target_pos: Any,
        *,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None = None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None = None,
    ) -> None:
        target = np.asarray(target_pos, dtype=np.float64)
        settle_phase = f"{phase}:settle"
        for _ in range(max(0, int(self.runtime.waypoint_settle_steps))):
            actual, _ = site_pose(self.env.model, self.env.data, self.env.mujoco, site_name)
            if float(np.linalg.norm(target - actual)) <= float(self.runtime.waypoint_settle_pos_tol):
                return
            obs, *_ = self.manager.step(action)
            obs = self._apply_constraints(fixed_joint_states, follow_attachments) or obs
            recorder.record(obs, action, settle_phase)

    def _apply_constraints(
        self,
        fixed_joint_states: list[tuple[str, tuple[np.ndarray, np.ndarray]]] | None,
        follow_attachments: list[tuple[str, str, dict[str, np.ndarray]]] | None,
    ) -> dict[str, Any] | None:
        constrained = False
        if fixed_joint_states:
            for joint_name, state in fixed_joint_states:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, joint_name, state)
            constrained = True
        if follow_attachments:
            from ..scene import apply_site_attachment

            for joint_name, site_name, attachment in follow_attachments:
                apply_site_attachment(self.env.model, self.env.data, self.env.mujoco, joint_name, site_name, attachment)
            constrained = True
        return self.env.get_observation() if constrained else None

    def _set_equality_active_if_exists(self, name: str, active: bool) -> None:
        try:
            eq_id = equality_id(self.env.model, self.env.mujoco, name)
        except ValueError:
            return
        self.env.data.eq_active[eq_id] = 1 if active else 0
        self.env.mujoco.mj_forward(self.env.model, self.env.data)

    def _record_site_target_error(self, phase: str, site_name: str, target_pos: Any) -> None:
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

    def _make_metadata(self, grasp_plan: list[dict[str, Any]], lift_plan: list[dict[str, Any]], *, num_steps: int) -> dict[str, Any]:
        final_obs = self.env.get_observation()
        pipette_pos, pipette_quat = free_joint_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette_joint,
        )
        return {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "task": self.name,
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "arm": self.runtime.arm,
            "pipette_joint": self.runtime.pipette_joint,
            "reset_info": self.env.last_reset_info,
            "slot_index": self.env.last_reset_info.get("random_single_free_joint", {}).get("slot_index"),
            "slot_name": self.env.last_reset_info.get("random_single_free_joint", {}).get("slot_name"),
            "grasp_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in grasp_plan],
            "lift_waypoints": [{k: v for k, v in item.items() if k != "action"} for item in lift_plan],
            "execution_site_errors": json_safe(self.execution_site_errors),
            "final_time": float(final_obs["time"]),
            "final_state_summary": {
                "pipette_pos": pipette_pos.tolist(),
                "pipette_quat": pipette_quat.tolist(),
            },
        }
