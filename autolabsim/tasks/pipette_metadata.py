"""Episode metadata construction for the pipette workflow."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..scene import free_joint_pose, site_pose
from ..task_target import PlannedTaskTarget
from .common import json_safe


class PipetteMetadataBuilder:
    """Build the stable metadata payload emitted by the pipette task."""

    def __init__(self, env: Any, runtime: Any) -> None:
        self.env = env
        self.runtime = runtime

    def build(
        self,
        grasp_plan: list[PlannedTaskTarget],
        middle_grasp_plan: list[PlannedTaskTarget],
        first_retreat_plan: list[PlannedTaskTarget],
        lift_plan: list[PlannedTaskTarget],
        mount_down_plan: list[PlannedTaskTarget],
        tip_retract_plan: list[PlannedTaskTarget],
        tube_plan: list[PlannedTaskTarget],
        *,
        handoff_attachment: dict[str, np.ndarray] | None,
        tip_attachment: dict[str, np.ndarray] | None,
        tip_target_info: dict[str, Any] | None,
        tube_target_info: dict[str, Any] | None,
        visual_servo_events: list[dict[str, Any]],
        execution_site_errors: list[dict[str, Any]],
        num_steps: int,
    ) -> dict[str, Any]:
        final_obs = self.env.get_observation()
        pipette_pos, pipette_quat = free_joint_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_joint,
        )
        final_tip_pos, final_tip_quat = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_tip_site,
        )
        return {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "task": "pipette_grasp",
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "arm": self.runtime.robot.arm,
            "pipette_joint": self.runtime.pipette.pipette_joint,
            "local_grasp_pose": {
                "body": self.runtime.pipette.pipette_body,
                "pos": list(self.runtime.grasp.handle_grasp_offset),
                "euler_xyz": list(self.runtime.grasp.handle_grasp_euler),
                "grasp_to_gripper_pos": list(
                    self.runtime.grasp.grasp_to_gripper_offset
                ),
                "grasp_to_gripper_euler_xyz": list(
                    self.runtime.grasp.grasp_to_gripper_euler
                ),
            },
            "middle_grasp_pose": {
                "body": self.runtime.pipette.pipette_body,
                "arm": self.runtime.grasp.middle_grasp_arm,
                "pos": list(self.runtime.grasp.middle_grasp_offset),
                "euler_xyz": list(self.runtime.grasp.middle_grasp_euler),
                "grasp_to_gripper_pos": list(
                    self.runtime.grasp.middle_grasp_to_gripper_offset
                ),
                "grasp_to_gripper_euler_xyz": list(
                    self.runtime.grasp.middle_grasp_to_gripper_euler
                ),
                "pregrasp_distance": self.runtime.grasp.middle_pregrasp_distance,
                "first_retreat_after_handoff_offset": list(
                    self.runtime.grasp.first_retreat_after_handoff_offset
                ),
            },
            "reset_info": self.env.last_reset_info,
            "slot_index": self.env.last_reset_info.get(
                "random_single_free_joint",
                {},
            ).get("slot_index"),
            "slot_name": self.env.last_reset_info.get(
                "random_single_free_joint",
                {},
            ).get("slot_name"),
            "grasp_waypoints": [item.to_metadata() for item in grasp_plan],
            "middle_grasp_waypoints": [
                item.to_metadata() for item in middle_grasp_plan
            ],
            "first_retreat_after_handoff_waypoints": [
                item.to_metadata() for item in first_retreat_plan
            ],
            "lift_waypoints": [item.to_metadata() for item in lift_plan],
            "tip_mount_waypoints": [
                item.to_metadata() for item in mount_down_plan
            ],
            "tip_retract_waypoints": [
                item.to_metadata() for item in tip_retract_plan
            ],
            "tube_waypoints": [item.to_metadata() for item in tube_plan],
            "tip_target": json_safe(tip_target_info),
            "handoff_attachment": json_safe(handoff_attachment),
            "tip_attachment": json_safe(tip_attachment),
            "tube_target": json_safe(tube_target_info),
            "tip_mount": {
                "tip_mount_offset": list(
                    self.runtime.tips.tip_mount_offset
                ),
                "tip_mount_target_euler_xyz": list(
                    self.runtime.tips.tip_mount_target_euler
                ),
                "tip_site_prefix": self.runtime.tips.tip_site_prefix,
                "tip_mount_site_suffix": (
                    self.runtime.tips.tip_mount_site_suffix
                ),
                "tip_end_site_suffix": self.runtime.tips.tip_end_site_suffix,
                "tip_pose_servo_enabled": (
                    self.runtime.tips.tip_pose_servo_enabled
                ),
                "tube_joint": self.runtime.tube.tube_joint,
                "tube_top_offset": self.runtime.tube.tube_top_offset,
                "tube_hover_height": self.runtime.tube.tube_hover_height,
                "tube_near_height": self.runtime.tube.tube_near_height,
                "tube_target_offset": list(
                    self.runtime.tube.tube_target_offset
                ),
            },
            "visual_servo": {
                "enabled": self.runtime.visual_servo.visual_servo_enabled,
                "max_iters": (
                    self.runtime.visual_servo.visual_servo_max_iters
                ),
                "steps": self.runtime.visual_servo.visual_servo_steps,
                "pos_tol": self.runtime.visual_servo.visual_servo_pos_tol,
                "rot_tol": self.runtime.visual_servo.visual_servo_rot_tol,
                "gain": self.runtime.visual_servo.visual_servo_gain,
                "integral_gain": (
                    self.runtime.visual_servo.visual_servo_integral_gain
                ),
                "max_correction": (
                    self.runtime.visual_servo.visual_servo_max_correction
                ),
                "events": json_safe(visual_servo_events),
            },
            "execution_site_errors": json_safe(execution_site_errors),
            "final_time": float(final_obs["time"]),
            "final_state_summary": {
                "pipette_pos": pipette_pos.tolist(),
                "pipette_quat": pipette_quat.tolist(),
                "pipette_tip_site_pos": final_tip_pos.tolist(),
                "pipette_tip_site_quat": final_tip_quat.tolist(),
            },
        }
