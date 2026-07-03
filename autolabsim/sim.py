'''
系统管理器和可挂载子系统
'''
from __future__ import annotations

from typing import Iterable

from .mujoco_env import AutoLabMuJoCoEnv


class System:
    def on_reset(self, env: AutoLabMuJoCoEnv) -> None:
        pass

    def before_step(self, env: AutoLabMuJoCoEnv, action) -> None:
        pass

    def after_step(self, env: AutoLabMuJoCoEnv, action, obs) -> None:
        pass

    def on_finish(self, env: AutoLabMuJoCoEnv) -> None:
        pass


class Manager:
    def __init__(self, env: AutoLabMuJoCoEnv, systems: Iterable[System] = ()):
        self.env = env
        self.systems = list(systems)

    def reset(self):
        obs = self.env.reset()
        for system in self.systems:
            system.on_reset(self.env)
        return self.env.get_observation()

    def step(self, action):
        for system in self.systems:
            system.before_step(self.env, action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        for system in self.systems:
            system.after_step(self.env, action, obs)
        if self.systems:
            obs = self.env.get_observation()
        return obs, reward, terminated, truncated, info

    def finish(self) -> None:
        for system in self.systems:
            system.on_finish(self.env)
