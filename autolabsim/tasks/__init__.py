from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..mujoco_env import EnvConfig
from .screw_cap import BimanualUnscrewTask, BimanualUnscrewTaskConfig
from .tube_grasp import TubeGraspTask, TubeGraspTaskConfig


@dataclass(frozen=True)
class TaskRequest:
    task: str
    seed: int
    episode_index: int
    out_dir: Path
    model: str
    reset_config: str | None
    cameras: tuple[str, ...]
    with_images: bool
    control_dt: float
    frame_skip: int | None
    gl_backend: str | None
    params: dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    factory: Callable[[TaskRequest], Any]
    summarize: Callable[[dict[str, Any]], dict[str, Any]]


def _make_env(request: TaskRequest) -> EnvConfig:
    return EnvConfig(
        model_path=request.model,
        reset_config=request.reset_config,
        reset_seed=request.seed,
        cameras=request.cameras,
        render_images=request.with_images,
        control_dt=request.control_dt,
        frame_skip=request.frame_skip,
        gl_backend=request.gl_backend,
    )


def _create_tube_grasp_task(request: TaskRequest) -> TubeGraspTask:
    params = request.params
    return TubeGraspTask(
        TubeGraspTaskConfig(
            env=_make_env(request),
            out_dir=request.out_dir,
            episode_index=request.episode_index,
            seed=request.seed,
            cameras=request.cameras,
            with_images=request.with_images,
            arm=params["arm"],
            open_gripper=params["open_gripper"],
            close_gripper=params["close_gripper"],
            settle_steps=params["settle_steps"],
            steps_per_segment=params["steps_per_segment"],
            grasp_hold_steps=params["grasp_hold_steps"],
            close_steps=params["close_steps"],
            hold_steps=params["hold_steps"],
            grasp_height=params["grasp_height"],
            pregrasp_distance=params["pregrasp_distance"],
            lift_offset=tuple(np.asarray(params["lift_offset"], dtype=np.float64).tolist()),
            pinch_forward_offset=params["pinch_forward_offset"],
            grasp_outward_offset=params["grasp_outward_offset"],
            tool_roll=params["tool_roll"],
            hold_active_tube_until_grasp=params["hold_active_tube_until_grasp"],
            ik_max_iters=params["ik_max_iters"],
            ik_pos_tol=params["ik_pos_tol"],
            ik_rot_tol=params["ik_rot_tol"],
            ik_damping=params["ik_damping"],
        )
    )


def _create_unscrew_task(request: TaskRequest) -> BimanualUnscrewTask:
    params = request.params
    return BimanualUnscrewTask(
        BimanualUnscrewTaskConfig(
            env=_make_env(request),
            out_dir=request.out_dir,
            episode_index=request.episode_index,
            seed=request.seed,
            cameras=request.cameras,
            with_images=request.with_images,
            tube_arm=params["tube_arm"],
            cap_arm=params["cap_arm"],
            open_gripper=params["open_gripper"],
            close_gripper=params["close_gripper"],
            settle_steps=params["settle_steps"],
            steps_per_segment=params["steps_per_segment"],
            grasp_hold_steps=params["grasp_hold_steps"],
            hold_steps=params["hold_steps"],
            close_steps=params["close_steps"],
            cap_hold_steps=params["cap_hold_steps"],
            tube_grasp_height=params["grasp_height"],
            tube_pregrasp_distance=params["pregrasp_distance"],
            tube_lift_offset=tuple(np.asarray(params["lift_offset"], dtype=np.float64).tolist()),
            tube_pinch_forward_offset=params["pinch_forward_offset"],
            tube_grasp_outward_offset=params["grasp_outward_offset"],
            tube_tool_roll=params["tube_tool_roll"] if params["tube_tool_roll"] is not None else params["tool_roll"],
            cap_offset=tuple(np.asarray(params["cap_offset"], dtype=np.float64).tolist()),
            cap_pregrasp_distance=params["cap_pregrasp_distance"],
            cap_post_offset=tuple(np.asarray(params["cap_post_offset"], dtype=np.float64).tolist()),
            cap_clearance_lift=params["cap_clearance_lift"],
            cap_tool_roll=params["cap_tool_roll"],
            ratchet_angle=params["ratchet_angle"],
            thread_pitch=params["thread_pitch"],
            release_lift=params["release_lift"],
            ik_max_iters=params["ik_max_iters"],
            ik_pos_tol=params["ik_pos_tol"],
            ik_rot_tol=params["ik_rot_tol"],
            ik_damping=params["ik_damping"],
        )
    )


def _summarize_tube_grasp(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata_found": True,
        "steps": metadata["steps"],
        "slot_index": metadata["slot_index"],
        "slot_name": metadata["slot_name"],
        "ik_all_waypoints_solved": metadata["ik_all_waypoints_solved"],
        "final_tube_pos": metadata["final_tube_pos"],
    }


def _summarize_unscrew(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata_found": True,
        "steps": metadata["steps"],
        "slot_index": metadata["slot_index"],
        "slot_name": metadata["slot_name"],
        "ik_all_waypoints_solved": all(
            item.get("ik_success", False) for item in metadata["tube_waypoints"] + metadata["cap_waypoints"]
        ),
        "final_tube_pos": metadata["final_state_summary"]["tube_pos"],
        "final_cap_pos": metadata["final_state_summary"]["cap_pos"],
        "screw_released": metadata["screw_progress"]["released"],
        "twist_angle": metadata["screw_progress"]["twist_angle"],
    }


TASK_REGISTRY: dict[str, TaskSpec] = {
    "tube_grasp": TaskSpec(factory=_create_tube_grasp_task, summarize=_summarize_tube_grasp),
    "tube_then_cap_grasp": TaskSpec(factory=_create_unscrew_task, summarize=_summarize_unscrew),
}


def create_task(request: TaskRequest):
    try:
        spec = TASK_REGISTRY[request.task]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {request.task}") from exc
    return spec.factory(request)


def summarize_metadata(task_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = TASK_REGISTRY[task_name]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc
    return spec.summarize(metadata)


def task_names() -> tuple[str, ...]:
    return tuple(TASK_REGISTRY.keys())

__all__ = [
    "BimanualUnscrewTask",
    "BimanualUnscrewTaskConfig",
    "TASK_REGISTRY",
    "TaskRequest",
    "TaskSpec",
    "TubeGraspTask",
    "TubeGraspTaskConfig",
    "create_task",
    "summarize_metadata",
    "task_names",
]
