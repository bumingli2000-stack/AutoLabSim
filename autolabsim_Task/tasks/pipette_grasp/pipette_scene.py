"""Scene queries used by the pipette task.

This module owns object discovery only. It does not construct TaskTarget objects,
solve IK, execute motion, or build episode metadata.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from autolabsim_Task.scene import free_joint_pose, site_pose
from ..common import random_reset_info

'''负责查询：
    当前有效枪头；
    最近枪头；
    当前有效离心管；
    枪头 mount/end site
'''
class PipetteSceneQuery:
    """Read active pipette-tip and centrifuge-tube information from the scene."""

    def __init__(
        self,
        env: Any,
        *,
        pipette_tip_site: str,
        tip_joint_prefix: str,
        tip_site_prefix: str,
        tip_mount_site_suffix: str,
        tip_end_site_suffix: str,
        fallback_tube_joint: str,
    ) -> None:
        self.env = env
        self.model = env.model
        self.data = env.data
        self.mujoco = env.mujoco
        self.pipette_tip_site = pipette_tip_site
        self.tip_joint_prefix = tip_joint_prefix
        self.tip_site_prefix = tip_site_prefix
        self.tip_mount_site_suffix = tip_mount_site_suffix
        self.tip_end_site_suffix = tip_end_site_suffix
        self.fallback_tube_joint = fallback_tube_joint

    def active_tube_joint(self) -> str:
        """Return the randomized active tube joint, or the configured fallback."""

        info = random_reset_info(self.env.last_reset_info)
        if info is not None and info.get("active_joint"):
            return str(info["active_joint"])
        return self.fallback_tube_joint

    def nearest_active_tip(self) -> dict[str, Any]:
        """选择当前距离移液枪最近的有效枪头。

            候选枪头来源：
            1. 优先读取本次随机重置记录中的 active 枪头；
            2. 如果没有随机记录，则扫描场景中所有匹配前缀的枪头 joint。

            过滤规则：
            - z < -1.0 的枪头视为被隐藏或移出工作区；
            - 使用 mount site 与 pipette tip site 的 XY 距离排序；
            - 如果模型没有 mount/end site，则退化为使用 free joint 位姿。
        """
        pipette_tip_pos, _ = site_pose(
            self.model,
            self.data,
            self.mujoco,
            self.pipette_tip_site,
        )
        candidates = self._active_tip_candidates()
        active_tips: list[dict[str, Any]] = []

        for item in candidates:
            joint_name = str(item["joint"])
            tip_pos, _ = free_joint_pose(
                self.model,
                self.data,
                self.mujoco,
                joint_name,
            )
            if tip_pos[2] < -1.0:
                continue

            mount_site = self.tip_site_name(
                joint_name,
                self.tip_mount_site_suffix,
            )
            end_site = self.tip_site_name(
                joint_name,
                self.tip_end_site_suffix,
            )
            mount_pose = self.optional_site_pose(mount_site)
            end_pose = self.optional_site_pose(end_site)

            if mount_pose is None:
                mount_site = None
                mount_pos = tip_pos
                mount_quat = np.asarray(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            else:
                mount_pos, mount_quat = mount_pose

            if end_pose is None:
                end_site = None
                end_pos = tip_pos
                end_quat = np.asarray(
                    [1.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                )
            else:
                end_pos, end_quat = end_pose

            xy_distance = float(
                np.linalg.norm(mount_pos[:2] - pipette_tip_pos[:2])
            )
            active_tips.append(
                {
                    **item,
                    "pos": tip_pos,
                    "mount_site": mount_site,
                    "mount_pos": mount_pos,
                    "mount_quat": mount_quat,
                    "end_site": end_site,
                    "end_pos": end_pos,
                    "end_quat": end_quat,
                    "xy_distance": xy_distance,
                }
            )

        if not active_tips:
            raise RuntimeError("No active pipette tips are visible in the scene")
        return min(active_tips, key=lambda item: item["xy_distance"])

    def tip_site_name(self, joint_name: str, suffix: str) -> str:
        if not joint_name.startswith(self.tip_joint_prefix):
            raise ValueError(
                f"Tip joint does not match prefix "
                f"{self.tip_joint_prefix!r}: {joint_name}"
            )
        index = joint_name[len(self.tip_joint_prefix) :]
        return f"{self.tip_site_prefix}{index}{suffix}"

    def optional_site_pose(
        self,
        site_name: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            return site_pose(
                self.model,
                self.data,
                self.mujoco,
                site_name,
            )
        except ValueError:
            return None

    def _active_tip_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        subset_info = self.env.last_reset_info.get(
            "random_free_joint_subset",
            {},
        )
        active = subset_info.get("active", []) if isinstance(subset_info, dict) else []

        for item in active:
            if not isinstance(item, dict) or "joint" not in item:
                continue
            candidates.append(
                {
                    "joint": str(item["joint"]),
                    "slot_name": item.get("slot_name"),
                }
            )

        if candidates:
            return candidates

        for joint_name in self.env.joint_names:
            if joint_name.startswith(self.tip_joint_prefix):
                candidates.append(
                    {
                        "joint": joint_name,
                        "slot_name": None,
                    }
                )
        return candidates
