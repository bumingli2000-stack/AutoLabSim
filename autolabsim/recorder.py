'''
统一的数据记录格式，便于后续处理
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeRecorder:
    cameras: tuple[str, ...]
    with_images: bool
    qpos_frames: list[np.ndarray] = field(default_factory=list)
    qvel_frames: list[np.ndarray] = field(default_factory=list)
    ctrl_frames: list[np.ndarray] = field(default_factory=list)
    state_frames: list[np.ndarray] = field(default_factory=list)
    action_frames: list[np.ndarray] = field(default_factory=list)
    time_frames: list[float] = field(default_factory=list)
    phase_frames: list[str] = field(default_factory=list)
    image_frames: list[dict[str, np.ndarray]] = field(default_factory=list)

    def record(self, obs: dict[str, Any], action: np.ndarray, phase: str) -> None:
        self.qpos_frames.append(np.asarray(obs["qpos"]).copy())
        self.qvel_frames.append(np.asarray(obs["qvel"]).copy())
        self.ctrl_frames.append(np.asarray(obs["ctrl"]).copy())
        self.state_frames.append(np.asarray(obs["state"]).copy())
        self.action_frames.append(np.asarray(action).copy())
        self.time_frames.append(float(obs["time"]))
        self.phase_frames.append(phase)
        if self.with_images:
            self.image_frames.append({camera: obs["images"][camera].copy() for camera in self.cameras})

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "qpos": np.stack(self.qpos_frames, axis=0),
            "qvel": np.stack(self.qvel_frames, axis=0),
            "ctrl": np.stack(self.ctrl_frames, axis=0),
            "state": np.stack(self.state_frames, axis=0),
            "action": np.stack(self.action_frames, axis=0),
            "time": np.asarray(self.time_frames, dtype=np.float64),
            "phase": np.asarray(self.phase_frames),
        }
        if self.with_images and self.image_frames:
            for camera in self.cameras:
                arrays[f"image_{camera}"] = np.stack([frame[camera] for frame in self.image_frames], axis=0)
        return arrays
