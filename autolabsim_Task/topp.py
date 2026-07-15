"""
TOPP-RA joint trajectory helper.

This mirrors AutoBio's trajectory layer: build a smooth joint-space path from
waypoint joint positions, then time-parameterize it under velocity and
acceleration limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/autolabsim_matplotlib")


@dataclass(frozen=True)
class ToppConfig:
    dof: int
    qc_vel: float = 2.0
    qc_acc: float = 2.0


class Topp:
    def __init__(self, config: ToppConfig):
        try:
            import toppra as ta
        except ImportError as exc:
            raise RuntimeError(
                "TOPPRA trajectory execution requires the 'toppra' package. "
                "Install project requirements or run: pip install toppra"
            ) from exc

        self.ta = ta
        self.dof = int(config.dof)
        vel_limits = np.asarray([[-config.qc_vel, config.qc_vel]] * self.dof, dtype=np.float64)
        acc_limits = np.asarray([[-config.qc_acc, config.qc_acc]] * self.dof, dtype=np.float64)
        self.constraints = [
            ta.constraint.JointVelocityConstraint(vel_limits),
            ta.constraint.JointAccelerationConstraint(acc_limits),
        ]

    def jnt_traj(self, q_waypoints: np.ndarray) -> Any:
        q_waypoints = np.asarray(q_waypoints, dtype=np.float64)
        if q_waypoints.ndim != 2 or q_waypoints.shape[1] != self.dof:
            raise ValueError(f"Expected q_waypoints shape (N, {self.dof}), got {q_waypoints.shape}")
        q_waypoints = self._drop_duplicate_waypoints(q_waypoints)
        if q_waypoints.shape[0] < 2:
            raise ValueError("TOPP trajectory needs at least two joint waypoints")

        ss = np.linspace(0.0, 1.0, q_waypoints.shape[0])
        path = self.ta.SplineInterpolator(ss, q_waypoints)
        instance = self.ta.algorithm.TOPPRA(self.constraints, path)
        trajectory = instance.compute_trajectory(0.0, 0.0)
        if trajectory is None:
            raise RuntimeError("TOPPRA failed to compute a trajectory for the requested waypoints")
        return trajectory

    @staticmethod
    def query(trajectory: Any, t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, trajectory.duration))
        return np.asarray(trajectory.eval(t), dtype=np.float64)

    @staticmethod
    def _drop_duplicate_waypoints(q_waypoints: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        keep = [q_waypoints[0]]
        for waypoint in q_waypoints[1:]:
            if np.linalg.norm(waypoint - keep[-1]) > tol:
                keep.append(waypoint)
        return np.asarray(keep, dtype=np.float64)
