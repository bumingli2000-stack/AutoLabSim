from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..mujoco_env import EnvConfig
from .screw_cap import BimanualUnscrewTask, BimanualUnscrewTaskConfig
from .pipette_grasp import (
    IKConfig,
    PipetteGraspTask,
    PipetteGraspTaskConfig,
    PipetteHandleGraspConfig,
    PipetteModelConfig,
    PipetteRobotConfig,
    PipetteTimingConfig,
    PipetteTipTargetConfig,
    PipetteTubeTargetConfig,
    VisualServoConfig,
    WaypointSettleConfig,
)


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


def _config_default(config_cls: type, field_name: str) -> Any:
    for field in fields(config_cls):
        if field.name != field_name:
            continue
        if field.default is not MISSING:
            return field.default
        if field.default_factory is not MISSING:  # type: ignore[attr-defined]
            return field.default_factory()  # type: ignore[misc]
        raise KeyError(f"Field has no default: {config_cls.__name__}.{field_name}")
    raise KeyError(f"Unknown config field: {config_cls.__name__}.{field_name}")


def _param(params: dict[str, Any], key: str, config_cls: type, field_name: str | None = None) -> Any:
    value = params.get(key)
    if value is not None:
        return value
    return _config_default(config_cls, field_name or key)


def _vec_param(params: dict[str, Any], key: str, config_cls: type, field_name: str | None = None) -> tuple[float, float, float]:
    value = _param(params, key, config_cls, field_name)
    if isinstance(value, str):
        value = np.fromstring(value.replace(",", " "), sep=" ")
    return tuple(np.asarray(value, dtype=np.float64).tolist())


def _quat_param(params: dict[str, Any], key: str, config_cls: type) -> tuple[float, float, float, float]:
    value = _param(params, key, config_cls)
    if isinstance(value, str):
        value = np.fromstring(value.replace(",", " "), sep=" ")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"{key} must contain exactly 4 numbers")
    return tuple(arr.tolist())


def _create_pipette_grasp_task(request: TaskRequest) -> PipetteGraspTask:
    params = request.params
    robot = PipetteRobotConfig(
        arm=_param(params, "arm", PipetteRobotConfig),
        open_gripper=_param(params, "open_gripper", PipetteRobotConfig),
        close_gripper=_param(params, "close_gripper", PipetteRobotConfig),
    )
    timing = PipetteTimingConfig(
        settle_steps=_param(params, "settle_steps", PipetteTimingConfig),
        free_settle_steps=_param(params, "free_settle_steps", PipetteTimingConfig),
        steps_per_segment=_param(params, "steps_per_segment", PipetteTimingConfig),
        close_steps=_param(params, "close_steps", PipetteTimingConfig),
        hold_steps=_param(params, "hold_steps", PipetteTimingConfig),
        grasp_hold_steps=_param(params, "grasp_hold_steps", PipetteTimingConfig),
    )
    grasp = PipetteHandleGraspConfig(
        pregrasp_distance=_param(params, "pregrasp_distance", PipetteHandleGraspConfig),
        handle_grasp_offset=_vec_param(params, "handle_grasp_offset", PipetteHandleGraspConfig),
        handle_grasp_euler=_vec_param(params, "handle_grasp_euler", PipetteHandleGraspConfig),
        grasp_to_gripper_offset=_vec_param(params, "grasp_to_gripper_offset", PipetteHandleGraspConfig),
        grasp_to_gripper_euler=_vec_param(params, "grasp_to_gripper_euler", PipetteHandleGraspConfig),
        middle_grasp_arm=_param(params, "middle_grasp_arm", PipetteHandleGraspConfig),
        middle_pregrasp_distance=_param(params, "middle_pregrasp_distance", PipetteHandleGraspConfig),
        middle_grasp_offset=_vec_param(params, "middle_grasp_offset", PipetteHandleGraspConfig),
        middle_grasp_euler=_vec_param(params, "middle_grasp_euler", PipetteHandleGraspConfig),
        middle_grasp_to_gripper_offset=_vec_param(
            params,
            "middle_grasp_to_gripper_offset",
            PipetteHandleGraspConfig,
        ),
        middle_grasp_to_gripper_euler=_vec_param(
            params,
            "middle_grasp_to_gripper_euler",
            PipetteHandleGraspConfig,
        ),
        first_retreat_after_handoff_offset=_vec_param(
            params,
            "first_retreat_after_handoff_offset",
            PipetteHandleGraspConfig,
        ),
    )
    pipette = PipetteModelConfig(
        pipette_joint=_param(params, "pipette_joint", PipetteModelConfig),
        pipette_body=_param(params, "pipette_body", PipetteModelConfig),
        pipette_tip_site=_param(params, "pipette_tip_site", PipetteModelConfig),
        parking_weld=_param(params, "parking_weld", PipetteModelConfig),
    )
    tips = PipetteTipTargetConfig(
        tip_joint_prefix=_param(params, "tip_joint_prefix", PipetteTipTargetConfig),
        tip_site_prefix=_param(params, "tip_site_prefix", PipetteTipTargetConfig),
        tip_mount_site_suffix=_param(params, "tip_mount_site_suffix", PipetteTipTargetConfig),
        tip_end_site_suffix=_param(params, "tip_end_site_suffix", PipetteTipTargetConfig),
        tip_pose_servo_enabled=_param(params, "tip_pose_servo_enabled", PipetteTipTargetConfig),
        tip_hover_height=_param(params, "tip_hover_height", PipetteTipTargetConfig),
        tip_mount_offset=_vec_param(params, "tip_mount_offset", PipetteTipTargetConfig),
        tip_mount_target_euler=_vec_param(params, "tip_mount_target_euler", PipetteTipTargetConfig),
        vertical_quat=_quat_param(params, "vertical_quat", PipetteTipTargetConfig),
    )
    tube = PipetteTubeTargetConfig(
        tube_joint=_param(params, "tube_joint", PipetteTubeTargetConfig),
        tube_top_offset=_param(params, "tube_top_offset", PipetteTubeTargetConfig),
        tube_hover_height=_param(params, "tube_hover_height", PipetteTubeTargetConfig),
        tube_near_height=_param(params, "tube_near_height", PipetteTubeTargetConfig),
        tube_target_offset=_vec_param(params, "tube_target_offset", PipetteTubeTargetConfig),
    )
    ik = IKConfig(
        ik_max_iters=_param(params, "ik_max_iters", IKConfig),
        ik_pos_tol=_param(params, "ik_pos_tol", IKConfig),
        ik_rot_tol=_param(params, "ik_rot_tol", IKConfig),
        ik_damping=_param(params, "ik_damping", IKConfig),
    )
    waypoint = WaypointSettleConfig(
        waypoint_settle_steps=_param(params, "waypoint_settle_steps", WaypointSettleConfig),
        waypoint_settle_pos_tol=_param(params, "waypoint_settle_pos_tol", WaypointSettleConfig),
    )
    visual_servo = VisualServoConfig(
        visual_servo_enabled=_param(params, "visual_servo_enabled", VisualServoConfig),
        visual_servo_max_iters=_param(params, "visual_servo_max_iters", VisualServoConfig),
        visual_servo_steps=_param(params, "visual_servo_steps", VisualServoConfig),
        visual_servo_pos_tol=_param(params, "visual_servo_pos_tol", VisualServoConfig),
        visual_servo_rot_tol=_param(params, "visual_servo_rot_tol", VisualServoConfig),
        visual_servo_gain=_param(params, "visual_servo_gain", VisualServoConfig),
        visual_servo_integral_gain=_param(params, "visual_servo_integral_gain", VisualServoConfig),
        visual_servo_max_correction=_param(params, "visual_servo_max_correction", VisualServoConfig),
    )
    return PipetteGraspTask(
        PipetteGraspTaskConfig(
            env=_make_env(request),
            out_dir=request.out_dir,
            episode_index=request.episode_index,
            seed=request.seed,
            cameras=request.cameras,
            with_images=request.with_images,
            robot=robot,
            timing=timing,
            grasp=grasp,
            pipette=pipette,
            tips=tips,
            tube=tube,
            ik=ik,
            waypoint=waypoint,
            visual_servo=visual_servo,
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
            tube_arm=_param(params, "tube_arm", BimanualUnscrewTaskConfig),
            cap_arm=_param(params, "cap_arm", BimanualUnscrewTaskConfig),
            open_gripper=_param(params, "open_gripper", BimanualUnscrewTaskConfig),
            close_gripper=_param(params, "close_gripper", BimanualUnscrewTaskConfig),
            settle_steps=_param(params, "settle_steps", BimanualUnscrewTaskConfig),
            steps_per_segment=_param(params, "steps_per_segment", BimanualUnscrewTaskConfig),
            grasp_hold_steps=_param(params, "grasp_hold_steps", BimanualUnscrewTaskConfig),
            hold_steps=_param(params, "hold_steps", BimanualUnscrewTaskConfig),
            close_steps=_param(params, "close_steps", BimanualUnscrewTaskConfig),
            cap_hold_steps=_param(params, "cap_hold_steps", BimanualUnscrewTaskConfig),
            tube_grasp_height=_param(params, "grasp_height", BimanualUnscrewTaskConfig, "tube_grasp_height"),
            tube_pregrasp_distance=_param(params, "pregrasp_distance", BimanualUnscrewTaskConfig, "tube_pregrasp_distance"),
            tube_lift_offset=_vec_param(params, "lift_offset", BimanualUnscrewTaskConfig, "tube_lift_offset"),
            tube_pinch_forward_offset=_param(params, "pinch_forward_offset", BimanualUnscrewTaskConfig, "tube_pinch_forward_offset"),
            tube_grasp_outward_offset=_param(params, "grasp_outward_offset", BimanualUnscrewTaskConfig, "tube_grasp_outward_offset"),
            tube_tool_roll=_param(params, "tube_tool_roll", BimanualUnscrewTaskConfig),
            cap_approach_axis=_vec_param(params, "cap_approach_axis", BimanualUnscrewTaskConfig),
            cap_offset=_vec_param(params, "cap_offset", BimanualUnscrewTaskConfig),
            cap_pregrasp_distance=_param(params, "cap_pregrasp_distance", BimanualUnscrewTaskConfig),
            cap_post_offset=_vec_param(params, "cap_post_offset", BimanualUnscrewTaskConfig),
            cap_clearance_lift=_param(params, "cap_clearance_lift", BimanualUnscrewTaskConfig),
            cap_tool_roll=_param(params, "cap_tool_roll", BimanualUnscrewTaskConfig),
            ratchet_angle=_param(params, "ratchet_angle", BimanualUnscrewTaskConfig),
            thread_pitch=_param(params, "thread_pitch", BimanualUnscrewTaskConfig),
            release_lift=_param(params, "release_lift", BimanualUnscrewTaskConfig),
            ik_max_iters=_param(params, "ik_max_iters", BimanualUnscrewTaskConfig),
            ik_pos_tol=_param(params, "ik_pos_tol", BimanualUnscrewTaskConfig),
            ik_rot_tol=_param(params, "ik_rot_tol", BimanualUnscrewTaskConfig),
            ik_damping=_param(params, "ik_damping", BimanualUnscrewTaskConfig),
            waypoint_settle_steps=_param(params, "waypoint_settle_steps", BimanualUnscrewTaskConfig),
            waypoint_settle_pos_tol=_param(params, "waypoint_settle_pos_tol", BimanualUnscrewTaskConfig),
            use_topp=_param(params, "use_topp", BimanualUnscrewTaskConfig),
            topp_vel=_param(params, "topp_vel", BimanualUnscrewTaskConfig),
            topp_acc=_param(params, "topp_acc", BimanualUnscrewTaskConfig),
        )
    )


def _summarize_pipette_grasp(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata_found": True,
        "steps": metadata["steps"],
        "slot_index": metadata.get("slot_index"),
        "slot_name": metadata.get("slot_name"),
        "ik_all_waypoints_solved": all(
            item.get("ik_success", False) for item in metadata["grasp_waypoints"] + metadata["lift_waypoints"]
        ),
        "final_pipette_pos": metadata["final_state_summary"]["pipette_pos"],
        "final_pipette_quat": metadata["final_state_summary"]["pipette_quat"],
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
    "pipette_grasp": TaskSpec(factory=_create_pipette_grasp_task, summarize=_summarize_pipette_grasp),
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
    "IKConfig",
    "PipetteGraspTask",
    "PipetteGraspTaskConfig",
    "PipetteHandleGraspConfig",
    "PipetteModelConfig",
    "PipetteRobotConfig",
    "PipetteTimingConfig",
    "PipetteTipTargetConfig",
    "PipetteTubeTargetConfig",
    "TASK_REGISTRY",
    "TaskRequest",
    "TaskSpec",
    "VisualServoConfig",
    "WaypointSettleConfig",
    "create_task",
    "summarize_metadata",
    "task_names",
]
