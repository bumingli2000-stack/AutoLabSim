"""Episode metadata construction for the ADP tip-to-tube workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from ...scene import free_joint_pose, site_pose
from ...task_target import PlannedTaskTarget
from ..common import json_safe


class AdpMetadataBuilder:
    """Build episode metadata for the ADP tip-to-tube task.

    This class extracts and summarizes relevant information from the completed
    trajectory, including waypoints, attachments, target selection, and final state.
    """

    def __init__(self, env: Any, runtime: Any) -> None:
        """Initialize the metadata builder.

        Args:
            env: The MuJoCo environment instance.
            runtime: The task configuration object (AdpTipToTubeTaskConfig).
        """
        self.env = env
        self.runtime = runtime

    def build(
        self,
        tip_hover_targets: tuple,
        mount_plan: list[PlannedTaskTarget],
        retract_plan: list[PlannedTaskTarget],
        tube_hover_plan: list[PlannedTaskTarget],
        tube_near_plan: list[PlannedTaskTarget],
        tube_return_plan: list[PlannedTaskTarget],
        trash_plan: list[PlannedTaskTarget],
        release_plan: list[PlannedTaskTarget],
        home_plan: list[PlannedTaskTarget],
        *,
        tip_attachment: dict[str, np.ndarray] | None,
        tip_target_info: dict[str, Any] | None,
        tube_target_info: dict[str, Any] | None,
        visual_servo_events: list[dict[str, Any]],
        execution_site_errors: list[dict[str, Any]],
        episode_arrays: dict[str, dict[str, Any]],
        lerobot_conversion: dict[str, Any],
        num_steps: int,
    ) -> dict[str, Any]:
        """Build the complete metadata dictionary.

        Args:
            tip_hover_targets: The two TaskTarget objects used for tip hover.
            mount_plan: Planned targets for the tip mount (downward) phase.
            retract_plan: Planned targets for tip retract (upward) phase.
            tube_hover_plan: Planned targets for tube hover phase.
            tube_near_plan: Planned targets for tube near (lower to opening) phase.
            tube_return_plan: Planned targets for lifting back above the tube.
            trash_plan: Planned targets for trash hover phase.
            release_plan: Planned targets for tip release (upward move).
            home_plan: Planned targets for returning home.
            tip_attachment: Mapping of tip attachment (joint->site) if captured.
            tip_target_info: Information about the selected tip.
            tube_target_info: Information about the selected tube (including near position).
            visual_servo_events: Closed-loop correction events collected during execution.
            execution_site_errors: Site target errors collected during execution.
            episode_arrays: NPZ array keys, shapes, and dtypes written for the episode.
            lerobot_conversion: Conversion hints matching convert_autolabsim_to_lerobot_act.py.
            num_steps: Total number of control steps in the episode.

        Returns:
            A dictionary containing the complete metadata.
        """
        final_obs = self.env.get_observation()

        # Get final pipette free joint pose
        pipette_pos, pipette_quat = free_joint_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_joint,
        )

        # Get final pipette tip site pose
        final_tip_pos, final_tip_quat = site_pose(
            self.env.model,
            self.env.data,
            self.env.mujoco,
            self.runtime.pipette.pipette_tip_site,
        )

        planned_waypoints = (
            list(tip_hover_targets)
            + list(mount_plan)
            + list(retract_plan)
            + list(tube_hover_plan)
            + list(tube_near_plan)
            + list(tube_return_plan)
            + list(trash_plan)
            + list(release_plan)
            + list(home_plan)
        )
        ik_success_values = [
            bool(getattr(item, "ik_success"))
            for item in planned_waypoints
            if hasattr(item, "ik_success")
        ]
        success = (
            bool(ik_success_values)
            and all(ik_success_values)
            and tip_attachment is not None
            and tip_target_info is not None
            and tube_target_info is not None
        )
        conversion_metadata = dict(lerobot_conversion)
        conversion_metadata.update(
            self._lerobot_compatibility_metadata(
                episode_arrays,
                conversion_metadata,
                success=bool(success),
            )
        )

        # Helper to convert TaskTarget to metadata dict if it has to_metadata
        def target_to_metadata(obj):
            if hasattr(obj, "to_metadata"):
                return obj.to_metadata()
            elif isinstance(obj, dict):
                return obj
            else:
                return str(obj)

        # Build metadata
        metadata = {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "success": bool(success),
            "task_success": bool(success),
            "completed": bool(success),
            "task": "adp_tip_to_tube",
            "task_text": (
                "Use the ADP pipette to mount a tip, visit the centrifuge tube, "
                "drop the tip over trash, and return home."
            ),
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "with_images": bool(self.runtime.with_images),
            "cameras": list(self.runtime.cameras),
            "episode_arrays": json_safe(episode_arrays),
            "lerobot_conversion": json_safe(conversion_metadata),
            "arm": self.runtime.arm,
            "pipette_joint": self.runtime.pipette.pipette_joint,
            "pipette_tip_site": self.runtime.pipette.pipette_tip_site,
            "reset_info": self.env.last_reset_info,
            "slot_index": self.env.last_reset_info.get(
                "random_single_free_joint", {}
            ).get("slot_index"),
            "slot_name": self.env.last_reset_info.get(
                "random_single_free_joint", {}
            ).get("slot_name"),
            # Waypoints
            "tip_hover_waypoints": [
                target_to_metadata(t) for t in tip_hover_targets
            ],
            "tip_mount_waypoints": [item.to_metadata() for item in mount_plan],
            "tip_retract_waypoints": [item.to_metadata() for item in retract_plan],
            "tube_hover_waypoints": [item.to_metadata() for item in tube_hover_plan],
            "tube_near_waypoints": [item.to_metadata() for item in tube_near_plan],
            "tube_return_waypoints": [
                item.to_metadata() for item in tube_return_plan
            ],
            "trash_waypoints": [item.to_metadata() for item in trash_plan],
            "release_waypoints": [item.to_metadata() for item in release_plan],
            "home_waypoints": [item.to_metadata() for item in home_plan],
            # Target info
            "tip_target": json_safe(tip_target_info),
            "tube_target": json_safe(tube_target_info),
            # Attachments
            "tip_attachment": json_safe(tip_attachment),
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
            # Final state
            "final_time": float(final_obs["time"]),
            "final_state_summary": {
                "pipette_pos": pipette_pos.tolist(),
                "pipette_quat": pipette_quat.tolist(),
                "pipette_tip_site_pos": final_tip_pos.tolist(),
                "pipette_tip_site_quat": final_tip_quat.tolist(),
            },
        }

        return metadata

    def _lerobot_compatibility_metadata(
        self,
        episode_arrays: dict[str, dict[str, Any]],
        conversion: dict[str, Any],
        *,
        success: bool,
    ) -> dict[str, Any]:
        """Summarize whether the emitted NPZ matches the ACT converter contract."""

        state_key = str(conversion.get("state_key", "ctrl"))
        action_key = str(conversion.get("action_key", "action"))
        time_key = str(conversion.get("time_key", "time"))
        action_offset = int(conversion.get("action_offset", 1))
        camera_keys = tuple(
            str(key) for key in conversion.get("camera_keys", ()) if key
        )
        requires_images = bool(conversion.get("requires_images", True))
        if requires_images and not camera_keys:
            camera_keys = tuple(f"image_{camera}" for camera in self.runtime.cameras)

        required_keys = (state_key, action_key, time_key, *camera_keys)
        missing_keys = [key for key in required_keys if key not in episode_arrays]

        state_shape = self._array_shape(episode_arrays, state_key)
        action_shape = self._array_shape(episode_arrays, action_key)
        time_shape = self._array_shape(episode_arrays, time_key)
        raw_frames = action_shape[0] if action_shape else None
        converted_frames = (
            int(raw_frames) - action_offset if raw_frames is not None else None
        )

        camera_checks: dict[str, dict[str, Any]] = {}
        for key in camera_keys:
            shape = self._array_shape(episode_arrays, key)
            dtype = self._array_dtype(episode_arrays, key)
            camera_checks[key] = {
                "shape": shape,
                "dtype": dtype,
                "is_4d": len(shape) == 4,
                "channel_count_ok": len(shape) == 4 and shape[-1] in (1, 3, 4),
                "dtype_uint8": self._is_dtype(dtype, np.uint8),
            }

        frame_counts = {
            key: self._array_shape(episode_arrays, key)[0]
            for key in required_keys
            if self._array_shape(episode_arrays, key)
        }
        lengths_match = (
            not missing_keys
            and len(frame_counts) == len(required_keys)
            and len(set(frame_counts.values())) == 1
        )
        checks = {
            "metadata_success_true": bool(success),
            "required_npz_keys_present": not missing_keys,
            "state_is_2d": len(state_shape) == 2,
            "action_is_2d": len(action_shape) == 2,
            "time_is_1d": len(time_shape) == 1,
            "frame_counts_match": lengths_match,
            "has_frames_after_action_offset": (
                converted_frames is not None and converted_frames > 0
            ),
            "images_present_when_required": (
                (not requires_images) or bool(camera_keys)
            ),
            "camera_arrays_ok": all(
                item["is_4d"]
                and item["channel_count_ok"]
                and item["dtype_uint8"]
                for item in camera_checks.values()
            ),
            "time_aligned_to_control_dt": bool(
                conversion.get("time_aligned_to_control_dt", False)
            ),
        }

        return {
            "compatible": all(checks.values()),
            "checks": checks,
            "required_npz_keys": list(required_keys),
            "missing_npz_keys": missing_keys,
            "frame_counts": frame_counts,
            "raw_frames": raw_frames,
            "converted_frames": converted_frames,
            "camera_checks": camera_checks,
        }

    @staticmethod
    def _array_shape(
        episode_arrays: dict[str, dict[str, Any]],
        key: str,
    ) -> list[int]:
        shape = episode_arrays.get(key, {}).get("shape", [])
        return [int(dim) for dim in shape] if isinstance(shape, list) else []

    @staticmethod
    def _array_dtype(
        episode_arrays: dict[str, dict[str, Any]],
        key: str,
    ) -> str:
        return str(episode_arrays.get(key, {}).get("dtype", ""))

    @staticmethod
    def _is_dtype(dtype: str, expected: Any) -> bool:
        try:
            return np.dtype(dtype) == np.dtype(expected)
        except TypeError:
            return False
