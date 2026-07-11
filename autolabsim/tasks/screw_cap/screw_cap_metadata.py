"""Episode metadata construction for the bimanual screw-cap task.

This module serializes task configuration, scene selection, planned waypoints,
screw progress, execution errors, and final object poses. It does not discover
scene objects, construct TaskTarget objects, solve IK, execute robot motion, or
update the screw simulation state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from autolabsim.scene import body_pos, free_joint_pos
from autolabsim.screw import ScrewCapSystem
from autolabsim.task_target import PlannedTaskTarget
from ..common import json_safe
from autolabsim.tasks.screw_cap.screw_cap_scene import ScrewCapSceneState


class ScrewCapMetadataBuilder:
    """Build the stable metadata payload emitted by the screw-cap task.

    A builder instance belongs to one task instance and reuses its MuJoCo
    environment and runtime configuration. ``build`` should be called only
    after all task stages have finished, because final object poses and screw
    progress are read at call time.

    The output intentionally preserves the metadata keys produced by the
    original ``BimanualUnscrewTask._make_metadata`` implementation so that
    existing episode readers remain compatible.
    """

    def __init__(
        self,
        env: Any,
        runtime: Any,
        *,
        task_name: str = "bimanual_unscrew_cap",
    ) -> None:
        self.env = env
        self.runtime = runtime
        self.task_name = task_name

    def build(
        self,
        scene: ScrewCapSceneState,
        tube_plan: Sequence[PlannedTaskTarget],
        cap_plan: Sequence[PlannedTaskTarget],
        unscrew_plan: Sequence[PlannedTaskTarget],
        *,
        screw_system: ScrewCapSystem | None,
        execution_site_errors: Sequence[dict[str, Any]],
        num_steps: int,
    ) -> dict[str, Any]:
        """Return metadata for one completed screw-cap episode.

        Parameters
        ----------
        scene:
            Active tube/cap names and reset information resolved immediately
            after environment reset.
        tube_plan:
            All planned tube-arm waypoints that should appear in metadata.
        cap_plan:
            All planned cap-arm waypoints except the dedicated unscrew plan.
        unscrew_plan:
            Ratchet/unscrew waypoints, including any recorded twist angles.
        screw_system:
            Live screw simulation system. Its final progress is sampled here.
        execution_site_errors:
            Site-position/pose errors collected during execution.
        num_steps:
            Number of recorded control steps in the episode arrays.
        """

        final_obs = self.env.get_observation()
        progress = screw_system.progress if screw_system is not None else None

        return {
            "format": "autolabsim_npz_v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": self.runtime.episode_index,
            "reset_seed": self.runtime.seed,
            "steps": int(num_steps),
            "task": self.task_name,
            "model_path": str(self.runtime.env.model_path),
            "reset_config": str(self.runtime.env.reset_config),
            "tube_arm": self.runtime.tube_arm,
            "cap_arm": self.runtime.cap_arm,
            "active_joint": scene.tube_joint,
            "cap_joint": scene.cap_joint,
            "cap_body": scene.cap_body,
            "slot_index": scene.slot_index,
            "slot_name": scene.slot_name,
            "reset_info": scene.reset_info,
            "tube_waypoints": [item.to_metadata() for item in tube_plan],
            "cap_waypoints": [item.to_metadata() for item in cap_plan],
            "unscrew_waypoints": [
                item.to_metadata() for item in unscrew_plan
            ],
            "execution_site_errors": json_safe(execution_site_errors),
            "screw_progress": {
                "released": bool(progress.released if progress else False),
                "twist_angle": float(
                    progress.twist_angle if progress else 0.0
                ),
                "lift_distance": float(
                    progress.lift_distance if progress else 0.0
                ),
                "release_angle_target": float(self.runtime.release_angle),
            },
            "final_time": float(final_obs["time"]),
            "final_state_summary": json_safe(
                {
                    "tube_pos": free_joint_pos(
                        self.env.model,
                        self.env.data,
                        self.env.mujoco,
                        scene.tube_joint,
                    ),
                    "cap_pos": body_pos(
                        self.env.model,
                        self.env.data,
                        self.env.mujoco,
                        scene.cap_body,
                    ),
                }
            ),
        }