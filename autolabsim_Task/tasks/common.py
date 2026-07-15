'''
任务共用的机械臂配置和命名辅助
'''
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import argparse
from typing import Any

import numpy as np

from ..scene_profile import DEFAULT_SCENE_NAMING


ARM_DEFAULTS = {
    "first": {
        "joint_names": ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
        "gripper_site": "2f85:pinch",
        "gripper_actuator": "2f85:fingers_actuator",
        "approach_axis": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
        "closing_axis": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    },
    "second": {
        "joint_names": (
            "ur5e1_shoulder_pan",
            "ur5e1_shoulder_lift",
            "ur5e1_elbow",
            "ur5e1_wrist_1",
            "ur5e1_wrist_2",
            "ur5e1_wrist_3",
        ),
        "gripper_site": "2f85_1pinch",
        "gripper_actuator": "2f85_1fingers_actuator",
        "approach_axis": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "closing_axis": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    },
}


def cap_joint_from_tube_joint(active_joint: str) -> str:
    prefix = DEFAULT_SCENE_NAMING.tube_joint_prefix
    return f"{DEFAULT_SCENE_NAMING.cap_joint_prefix}{active_joint[len(prefix):]}"


def cap_body_from_tube_joint(active_joint: str) -> str:
    prefix = DEFAULT_SCENE_NAMING.tube_joint_prefix
    return f"{DEFAULT_SCENE_NAMING.cap_body_prefix}{active_joint[len(prefix):]}"


def cap_weld_from_tube_joint(active_joint: str) -> str:
    prefix = DEFAULT_SCENE_NAMING.tube_joint_prefix
    return f"{DEFAULT_SCENE_NAMING.cap_weld_prefix}{active_joint[len(prefix):]}"


def random_reset_info(reset_info: dict[str, Any]) -> dict[str, Any] | None:
    info = reset_info.get("random_single_free_joint")
    return info if isinstance(info, dict) else None


def parse_seeds(value: str | None, count: int, seed_start: int) -> list[int]:
    if value is None:
        return [seed_start + i for i in range(count)]

    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds was provided but no valid integer seeds were found")
    return seeds


def parse_vec3(value: str, name: str) -> np.ndarray:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must have exactly 3 numbers")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def parse_optional_vec3(value: str | None, name: str) -> np.ndarray | None:
    if value is None or value.strip().lower() in ("", "none"):
        return None
    return parse_vec3(value, name)


def parse_cameras(value: str) -> tuple[str, ...]:
    cameras = tuple(camera.strip() for camera in value.split(",") if camera.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("--cameras must contain at least one camera name")
    return cameras


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
