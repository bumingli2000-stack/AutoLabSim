'''
任务基类，基于此类拓展为子任务
'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .episode_io import save_episode
from .mujoco_env import AutoLabMuJoCoEnv, EnvConfig
from .sim import Manager, System


@dataclass(frozen=True)
class TaskConfig:
    env: EnvConfig
    with_images: bool = False
    cameras: tuple[str, ...] = ("overview_camera",)


class AutoLabTask:
    name = "task"

    def __init__(self, config: TaskConfig, systems: list[System] | None = None):
        self.config = config
        # 初始化环境，传入环境配置
        self.env = AutoLabMuJoCoEnv(config.env)
        self.manager = Manager(self.env, systems or [])
        self.task_info: dict[str, Any] = {}

    def reset(self) -> dict[str, Any]:
        return self.manager.reset()

    def finish(self) -> None:
        self.manager.finish()
        self.env.close()

    def save_episode(self, episode_dir: Path, metadata: dict[str, Any], arrays: dict[str, Any]) -> None:
        save_episode(episode_dir, metadata, arrays)
