"""Scene discovery helpers for the bimanual screw-cap task.

This module is responsible only for reading task-related object information
from the MuJoCo scene and the latest reset result. It does not build
TaskTarget objects, solve IK, execute trajectories, or update screw motion.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from autolabsim.scene import body_pos, free_joint_pos
from autolabsim.scene_profile import active_joint_fallback
from ..common import (
    cap_body_from_tube_joint,
    cap_joint_from_tube_joint,
    cap_weld_from_tube_joint,
    random_reset_info,
)

@dataclass(frozen=True)
class ScrewCapSceneState:
    """Names and reset information for the active tube-cap pair.
    One episode randomly places a tube in a rack slot. Once ``resolve`` has
    been called, this object provides all names and the original slot pose
    needed by the rest of the workflow.
    Array fields are copied when the state is created so that later MuJoCo
    simulation updates do not silently modify the stored return pose.
    """

    reset_info: dict[str, Any]
    random_info: dict[str, Any] | None

    tube_joint: str
    cap_joint: str
    cap_body: str
    cap_weld: str

    slot_index: int | None
    slot_name: str | None
    slot_pos: np.ndarray
    slot_quat: np.ndarray | None

class ScrewCapSceneQuery:
    """Read active tube, cap, and rack-slot information from the scene.

    Responsibilities:
    - inspect ``env.last_reset_info``;
    - identify the active tube free joint;
    - derive the matching cap joint/body/weld names;
    - preserve the tube's original rack-slot pose;
    - provide current tube and cap positions when later stages need them.

    This class intentionally contains no planning or execution code.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.mujoco = env.mujoco

    def resolve(self) -> ScrewCapSceneState:
        """Resolve the active tube-cap pair for the current episode.

        The preferred source is the random-reset record. When the scene was
        not randomized, the project-level fallback joint is used instead.
        """

        reset_info = dict(self.env.last_reset_info)
        random_info = random_reset_info(reset_info)

        tube_joint = (
            str(random_info["active_joint"])
            if random_info is not None and random_info.get("active_joint")
            else active_joint_fallback()
        )

        cap_joint = cap_joint_from_tube_joint(tube_joint)
        cap_body = cap_body_from_tube_joint(tube_joint)
        cap_weld = cap_weld_from_tube_joint(tube_joint)

        slot_index = self._optional_int(
            random_info.get("slot_index") if random_info else None
        )
        slot_name = (
            str(random_info["slot_name"])
            if random_info is not None and random_info.get("slot_name") is not None
            else None
        )
        slot_pos, slot_quat = self._resolve_original_slot_pose(
            tube_joint,
            random_info,
        )

        return ScrewCapSceneState(
            reset_info=reset_info,
            random_info=random_info,
            tube_joint=tube_joint,
            cap_joint=cap_joint,
            cap_body=cap_body,
            cap_weld=cap_weld,
            slot_index=slot_index,
            slot_name=slot_name,
            slot_pos=slot_pos,
            slot_quat=slot_quat,
        )
    
    def tube_position(self, scene: ScrewCapSceneState) -> np.ndarray:
        """Return the tube free-joint position at the current simulation step."""

        return free_joint_pos(
            self.model,
            self.data,
            self.mujoco,
            scene.tube_joint,
        )

    def cap_position(self, scene: ScrewCapSceneState) -> np.ndarray:
        """Return the cap body's current world position."""

        return body_pos(
            self.model,
            self.data,
            self.mujoco,
            scene.cap_body,
        )

    def _resolve_original_slot_pose(
        self,
        tube_joint: str,
        random_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the tube pose used when it is placed back in the rack.

        For randomized episodes, ``random_single_free_joint.pose`` stores the
        selected rack slot. This is the preferred return target.

        For non-randomized scenes, the legacy behavior is preserved: the
        current tube position is used and no orientation override is applied.
        """

        slot_pose = random_info.get("pose") if random_info else None
        if isinstance(slot_pose, dict) and "pos" in slot_pose:
            slot_pos = np.asarray(slot_pose["pos"], dtype=np.float64).copy()
            slot_quat = np.asarray(
                slot_pose.get("quat", [1.0, 0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).copy()
            return slot_pos, slot_quat

        slot_pos = free_joint_pos(
            self.model,
            self.data,
            self.mujoco,
            tube_joint,
        )
        return np.asarray(slot_pos, dtype=np.float64).copy(), None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)