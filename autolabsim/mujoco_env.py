'''
仿真运行时接口层
底层 MuJoCo 环境封装，
把原始 MuJoCo model/data 包成一个统一可调用的环境对象。

主要负责：
加载 XML 场景
创建 MjModel / MjData
reset()
step(action)
读取 observation
渲染相机图像
维护 action 维度、joint 名字、actuator 名字、control dt
'''
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

from .reset_config import ResetConfig, apply_reset_config, load_reset_config


@dataclass(frozen=True)
class EnvConfig:
    model_path: str | Path = "model/scenes/scene_mujoco.xml"
    cameras: tuple[str, ...] = ("wrist_cam", "wrist_cam1")
    image_width: int = 640
    image_height: int = 480
    control_dt: float = 0.05
    frame_skip: int | None = None
    keyframe: str | None = "home"
    render_images: bool = True
    gl_backend: str | None = None
    disable_contact: bool = False
    reset_config: str | Path | ResetConfig | None = None
    reset_seed: int | None = None


class AutoLabMuJoCoEnv:
    """Small MuJoCo wrapper for data collection and policy deployment.

    The class intentionally avoids a hard dependency on gymnasium/lerobot.
    That keeps the scene usable as a low-level source of synchronized state,
    camera images, and actions.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.model_path = Path(self.config.model_path)
        self.reset_config = load_reset_config(self.config.reset_config)
        self.rng = np.random.default_rng(self.config.reset_seed)
        self.last_reset_info: ResetConfig = {}

        if self.config.render_images and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = self.config.gl_backend or "egl"

        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'mujoco'. Install it with: python -m pip install mujoco"
            ) from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        if self.config.disable_contact:
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

        self.frame_skip = (
            max(1, int(self.config.frame_skip))
            if self.config.frame_skip is not None
            else max(1, round(self.config.control_dt / self.model.opt.timestep))
        )
        self.control_dt = float(self.frame_skip * self.model.opt.timestep)
        self.action_dim = int(self.model.nu)
        self.action_names = tuple(self._name_for_id(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu))
        self.joint_names = tuple(self._name_for_id(mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt))
        self.action_low, self.action_high = self._action_bounds()

        self._renderers: dict[str, Any] = {}
        if self.config.render_images:
            for camera in self.config.cameras:
                self._renderers[camera] = mujoco.Renderer(
                    self.model,
                    self.config.image_height,
                    self.config.image_width,
                )

    def reset(self) -> dict[str, Any]:
        self.mujoco.mj_resetData(self.model, self.data)

        if self.config.keyframe:
            key_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_KEY,
                self.config.keyframe,
            )
            if key_id >= 0:
                self.mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self.sync_ctrl_to_qpos()
        self.last_reset_info = apply_reset_config(self.model, self.data, self.mujoco, self.reset_config, self.rng)
        self.mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def step(self, action: np.ndarray | list[float] | tuple[float, ...]):
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action_array.shape}")

        self.data.ctrl[:] = action_array
        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info = {"sim_time": float(self.data.time), "frame_skip": self.frame_skip, "control_dt": self.control_dt}
        return obs, reward, terminated, truncated, info

    def get_observation(self) -> dict[str, Any]:
        state = np.concatenate(
            [
                np.asarray(self.data.qpos).copy(),
                np.asarray(self.data.qvel).copy(),
                np.asarray(self.data.ctrl).copy(),
            ]
        )

        obs: dict[str, Any] = {
            "state": state,
            "qpos": np.asarray(self.data.qpos).copy(),
            "qvel": np.asarray(self.data.qvel).copy(),
            "ctrl": np.asarray(self.data.ctrl).copy(),
            "time": float(self.data.time),
        }

        if self._renderers:
            obs["images"] = self.render_cameras()

        return obs

    def render_cameras(self) -> dict[str, np.ndarray]:
        images = {}
        for camera, renderer in self._renderers.items():
            renderer.update_scene(self.data, camera=camera)
            images[camera] = renderer.render().copy()
        return images

    def sample_home_action(self) -> np.ndarray:
        """Return the current control vector, useful for a first smoke step."""
        return np.asarray(self.data.ctrl).copy()

    def sync_ctrl_to_qpos(self) -> None:
        """Initialize joint-position actuator targets from the current qpos."""
        for action_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[action_id][0])
            if joint_id < 0:
                continue
            if action_id >= len(self.data.ctrl):
                continue
            action_name = self.action_names[action_id]
            if action_name.startswith("2f85"):
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            self.data.ctrl[action_id] = self.data.qpos[qadr]

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def _name_for_id(self, obj_type: Any, obj_id: int) -> str:
        name = self.mujoco.mj_id2name(self.model, obj_type, obj_id)
        return name if name is not None else f"{obj_type.name.lower()}_{obj_id}"

    def _action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        low = np.where(limited, ctrlrange[:, 0], -np.inf)
        high = np.where(limited, ctrlrange[:, 1], np.inf)
        return low, high
