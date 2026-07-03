'''
单臂抓管任务
'''
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..ik import solve_site_ik
from ..math3d import gripper_quat_from_axes, unit
from ..mujoco_env import EnvConfig
from ..recorder import EpisodeRecorder
from ..scene_profile import active_joint_fallback
from ..scene import actuator_id, capture_free_joint_state, free_joint_pos, joint_qpos_ids, restore_free_joint_state
from ..task import AutoLabTask, TaskConfig
from .common import ARM_DEFAULTS, json_safe, random_reset_info


@dataclass(frozen=True)
class TubeGraspTaskConfig:
    env: EnvConfig
    out_dir: Path
    episode_index: int
    seed: int
    cameras: tuple[str, ...] = ("overview_camera",)
    with_images: bool = False
    arm: str = "second"
    open_gripper: float = 0.0
    close_gripper: float = 255.0
    settle_steps: int = 20
    steps_per_segment: int = 30
    grasp_hold_steps: int = 8
    close_steps: int = 12
    hold_steps: int = 20
    grasp_height: float = 0.09
    pregrasp_distance: float = 0.10
    lift_offset: tuple[float, float, float] = (0.25, 0.0, 0.12)
    pinch_forward_offset: float = 0.02
    grasp_outward_offset: float = 0.02
    tool_roll: float = float(np.pi)
    hold_active_tube_until_grasp: bool = True
    ik_max_iters: int = 500
    ik_pos_tol: float = 0.003
    ik_rot_tol: float = 0.05
    ik_damping: float = 0.08


class TubeGraspTask(AutoLabTask):
    name = "tube_grasp"

    def __init__(self, config: TubeGraspTaskConfig):
        self.runtime = config
        self.arm = ARM_DEFAULTS[config.arm]
        super().__init__(TaskConfig(env=config.env, with_images=config.with_images, cameras=config.cameras))

    def run(self) -> dict:
        self.reset()
        reset_info = dict(self.env.last_reset_info)
        random_info = random_reset_info(reset_info)
        active_joint = str(random_info["active_joint"]) if random_info else active_joint_fallback()
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.arm["gripper_actuator"]))
        held_tube_state = (
            capture_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint)
            if self.runtime.hold_active_tube_until_grasp
            else None
        )

        settle_action = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        settle_action[gripper_id] = self.runtime.open_gripper
        for _ in range(self.runtime.settle_steps):
            obs, *_ = self.manager.step(settle_action)
            if held_tube_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, active_joint, held_tube_state)
                obs = self.env.get_observation()

        plan = self._plan(free_joint_pos(self.env.model, self.env.data, self.env.mujoco, active_joint))
        recorder = EpisodeRecorder(self.runtime.cameras, self.runtime.with_images)
        recorder.record(self.env.get_observation(), np.asarray(self.env.data.ctrl).copy(), "start")
        self._move(recorder, np.asarray(plan[0]["action"]), self.runtime.steps_per_segment, "move_pregrasp", active_joint, held_tube_state)
        self._move(recorder, np.asarray(plan[1]["action"]), self.runtime.steps_per_segment, "move_grasp", active_joint, held_tube_state)
        self._hold(recorder, np.asarray(plan[1]["action"]), self.runtime.grasp_hold_steps, "hold_grasp_open", active_joint, held_tube_state)

        close_grasp = np.asarray(plan[1]["action"]).copy()
        close_grasp[gripper_id] = self.runtime.close_gripper
        self._move(recorder, close_grasp, self.runtime.close_steps, "close_gripper", active_joint, held_tube_state)

        close_lift = np.asarray(plan[2]["action"]).copy()
        close_lift[gripper_id] = self.runtime.close_gripper
        self._move(recorder, close_lift, self.runtime.steps_per_segment, "lift", active_joint, None)
        self._hold(recorder, close_lift, self.runtime.hold_steps, "hold_lift", active_joint, None)

        arrays = recorder.to_arrays()
        arrays["final_state"] = self.env.get_observation()["state"]
        metadata = {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(arrays["qpos"].shape[0]),
            "task": self.name,
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "arm": self.runtime.arm,
            "active_joint": active_joint,
            "slot_index": random_info.get("slot_index") if random_info else None,
            "slot_name": random_info.get("slot_name") if random_info else None,
            "reset_info": reset_info,
            "waypoints": [{k: v for k, v in item.items() if k != "action"} for item in plan],
            "ik_all_waypoints_solved": all(item["ik_success"] for item in plan),
            "hold_active_tube_until_grasp": bool(self.runtime.hold_active_tube_until_grasp),
            "script_args": json_safe(self.runtime.__dict__),
            "final_tube_pos": free_joint_pos(self.env.model, self.env.data, self.env.mujoco, active_joint).tolist(),
        }
        self.save_episode(self.runtime.out_dir, metadata, arrays)
        return metadata

    def _plan(self, tube_pos: np.ndarray) -> list[dict]:
        approach = unit(self.arm["approach_axis"], "approach_axis")
        closing = unit(self.arm["closing_axis"], "closing_axis")
        quat = gripper_quat_from_axes(self.env.mujoco, approach, closing, self.runtime.tool_roll)
        grasp_pos = (
            tube_pos
            + np.asarray([0.0, 0.0, self.runtime.grasp_height], dtype=np.float64)
            + approach * self.runtime.pinch_forward_offset
            - approach * self.runtime.grasp_outward_offset
        )
        pregrasp_pos = grasp_pos - approach * self.runtime.pregrasp_distance
        lift_pos = grasp_pos + np.asarray(self.runtime.lift_offset, dtype=np.float64)
        waypoints = [
            {"name": "pregrasp", "pos": pregrasp_pos, "quat": quat},
            {"name": "grasp", "pos": grasp_pos, "quat": quat},
            {"name": "lift", "pos": lift_pos, "quat": quat},
        ]

        joint_names = tuple(self.arm["joint_names"])
        qpos_ids = joint_qpos_ids(self.env.model, self.env.mujoco, joint_names)
        arm_actuator_ids = [actuator_id(self.env.model, self.env.mujoco, name) for name in joint_names]
        gripper_id = actuator_id(self.env.model, self.env.mujoco, str(self.arm["gripper_actuator"]))
        start_qpos = self.env.data.qpos.copy()
        start_qvel = self.env.data.qvel.copy()
        start_ctrl = self.env.data.ctrl.copy()
        plan = []
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
            action[gripper_id] = self.runtime.open_gripper
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

    def _move(self, recorder: EpisodeRecorder, target_action: np.ndarray, steps: int, phase: str, hold_joint: str, held_state):
        start = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        denom = max(1, steps)
        for step in range(1, denom + 1):
            alpha = step / denom
            action = (1.0 - alpha) * start + alpha * target_action
            obs, *_ = self.manager.step(action)
            if held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
            recorder.record(obs, action, phase)

    def _hold(self, recorder: EpisodeRecorder, action: np.ndarray, steps: int, phase: str, hold_joint: str, held_state):
        for _ in range(max(0, steps)):
            obs, *_ = self.manager.step(action)
            if held_state is not None:
                restore_free_joint_state(self.env.model, self.env.data, self.env.mujoco, hold_joint, held_state)
                obs = self.env.get_observation()
            recorder.record(obs, action, phase)
