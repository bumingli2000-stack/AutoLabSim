"""Typed motion planning and execution context objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


JointState = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ArmMotionConfig:
    name: str
    joint_names: tuple[str, ...]
    actuator_site: str
    gripper_actuator: str


@dataclass(frozen=True)
class IKSettings:
    max_iters: int = 800
    pos_tol: float = 0.0005
    rot_tol: float = 0.02
    damping: float = 0.01


@dataclass(frozen=True)
class GripperSettings:
    open_value: float
    close_value: float


@dataclass(frozen=True)
class VisualServoSettings:
    enabled: bool = True
    max_iters: int = 12
    steps: int = 10
    pos_tol: float = 0.0001
    rot_tol: float = 0.02
    gain: float = 0.8
    integral_gain: float = 0.25
    max_correction: float = 0.02


@dataclass(frozen=True)
class ExecutionSettings:
    steps_per_segment: int = 50
    waypoint_settle_steps: int = 20
    waypoint_settle_pos_tol: float = 0.0005
    visual_servo: VisualServoSettings = field(default_factory=VisualServoSettings)


@dataclass(frozen=True)
class SiteAttachment:
    """Rigid relation from a parent site to a child free joint."""

    joint_name: str
    parent_site: str
    local_pos: np.ndarray
    local_quat: np.ndarray

    @classmethod
    def from_mapping(cls, joint_name: str, parent_site: str, attachment: Mapping[str, np.ndarray]) -> "SiteAttachment":
        return cls(
            joint_name=joint_name,
            parent_site=parent_site,
            local_pos=np.asarray(attachment["local_pos"], dtype=np.float64).copy(),
            local_quat=np.asarray(attachment["local_quat"], dtype=np.float64).copy(),
        )

    def to_mapping(self) -> dict[str, np.ndarray]:
        return {
            "local_pos": np.asarray(self.local_pos, dtype=np.float64).copy(),
            "local_quat": np.asarray(self.local_quat, dtype=np.float64).copy(),
        }


@dataclass(frozen=True)
class KinematicBinding:
    """How an arm actuator site controls a requested site."""

    arm: str
    actuator_site: str
    controlled_site: str
    attachments: tuple[SiteAttachment, ...] = ()


@dataclass(frozen=True)
class PlanningContext:
    bindings: tuple[KinematicBinding, ...] = ()

    def binding_for(self, arm: str, controlled_site: str, actuator_site: str) -> KinematicBinding:
        matches = [
            binding
            for binding in self.bindings
            if binding.arm == arm and binding.controlled_site == controlled_site
        ]
        if matches:
            return matches[-1]
        if controlled_site == actuator_site:
            return KinematicBinding(arm=arm, actuator_site=actuator_site, controlled_site=controlled_site)
        known = sorted({binding.controlled_site for binding in self.bindings if binding.arm == arm})
        raise ValueError(
            f"No kinematic binding for arm={arm!r}, controlled_site={controlled_site!r}. "
            f"Known controlled sites for this arm: {known}"
        )


@dataclass(frozen=True)
class FixedJointState:
    joint_name: str
    state: JointState


@dataclass(frozen=True)
class ExecutionContext:
    fixed_joint_states: tuple[FixedJointState, ...] = ()
    attachments: tuple[SiteAttachment, ...] = ()


def arm_motion_configs(raw: Mapping[str, Mapping[str, object]]) -> dict[str, ArmMotionConfig]:
    configs: dict[str, ArmMotionConfig] = {}
    for name, item in raw.items():
        configs[str(name)] = ArmMotionConfig(
            name=str(name),
            joint_names=tuple(str(joint_name) for joint_name in item["joint_names"]),  # type: ignore[index]
            actuator_site=str(item["gripper_site"]),
            gripper_actuator=str(item["gripper_actuator"]),
        )
    return configs
