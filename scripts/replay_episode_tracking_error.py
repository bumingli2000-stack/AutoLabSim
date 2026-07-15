#!/usr/bin/env python3
"""Open-loop replay and joint/TCP tracking diagnostics for AutoLabSim episodes.

The script replays the original per-frame ``action`` array from an AutoLabSim
episode and separates two error sources:

1. command tracking:
       actuator joint-position target -> actual simulated joint/TCP
2. replay fidelity:
       recorded qpos/TCP -> replayed qpos/TCP

Recommended use
---------------
Run from the AutoLabSim repository root:

python scripts/replay_episode_tracking_error.py \
  --episode-dir data/episodes/screw_cap_batch/episode_000_seed_0000 \
  --output-dir outputs/replay_tracking/episode_000 \
  --runtime-adapter-script scripts/deploy_lerobot_act_autolabsim.py \
  --plot

The default ``recorded`` initialization copies frame 0 qpos/qvel/ctrl into the
new environment, then replays actions from frame 1. This matches the recording
alignment used by the dataset, where action[t] produced observation[t].

Notes
-----
- The two gripper actuators are excluded from joint-angle tracking by default,
  because their 0-150/255 control scale is not necessarily equal to a physical
  joint coordinate.
- The runtime adapter is used only to reproduce simulation-side attachment and
  screw-cap lifecycle events. Disable it with ``--no-runtime-adapter`` for a
  pure actuator-only test.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from autolabsim_Task.tasks import TaskRequest, create_task
    from autolabsim_Task.scene import site_pose
except ModuleNotFoundError:
    from autolabsim.tasks import TaskRequest, create_task
    from autolabsim.scene import site_pose


@dataclass(frozen=True)
class JointBinding:
    actuator_id: int
    actuator_name: str
    joint_id: int
    joint_name: str
    qpos_adr: int
    qvel_adr: int
    gear: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one AutoLabSim episode and compute joint/TCP tracking "
            "errors."
        )
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        required=True,
        help="Directory containing episode.npz and metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV, NPZ, JSON and optional plots.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help=(
            "Task registry name. Default maps bimanual_unscrew_cap to "
            "tube_then_cap_grasp."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override model path stored in metadata.json.",
    )
    parser.add_argument(
        "--reset-config",
        type=Path,
        default=None,
        help="Override reset config stored in metadata.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override reset_seed stored in metadata.json.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=None,
        help="Override timestep; default is inferred from episode time.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of replayed frames.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "First action index. Default is 1 for recorded initialization and "
            "0 for reset initialization."
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=("recorded", "reset"),
        default="recorded",
        help=(
            "recorded: initialize qpos/qvel/ctrl from episode frame 0; "
            "reset: use a fresh task reset."
        ),
    )
    parser.add_argument(
        "--action-key",
        choices=("action", "ctrl"),
        default="action",
        help="Episode array replayed as the actuator command.",
    )
    parser.add_argument(
        "--runtime-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reproduce attachment and screw-cap lifecycle events.",
    )
    parser.add_argument(
        "--runtime-adapter-script",
        type=Path,
        default=Path("scripts/deploy_lerobot_act_autolabsim.py"),
        help=(
            "Deployment script exporting ACTScrewRuntimeAdapter. The current "
            "ratchet-v2 deployment script is recommended."
        ),
    )
    parser.add_argument("--cap-grasp-distance", type=float, default=0.055)
    parser.add_argument("--tube-grasp-distance", type=float, default=0.080)
    parser.add_argument("--cap-place-distance", type=float, default=0.120)
    parser.add_argument("--tube-slot-distance", type=float, default=0.120)
    parser.add_argument("--closed-ratio", type=float, default=0.50)
    parser.add_argument("--open-ratio", type=float, default=0.10)
    parser.add_argument("--confirm-steps", type=int, default=3)
    parser.add_argument("--max-twist-step", type=float, default=0.35)
    parser.add_argument(
        "--exclude-actuator",
        action="append",
        default=[],
        help=(
            "Actuator name or numeric id excluded from joint tracking. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--include-grippers",
        action="store_true",
        help=(
            "Include gripper actuator-to-joint tracking. Usually not useful "
            "because gripper control units may not equal joint coordinates."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save error plots with matplotlib.",
    )
    return parser.parse_args()


def resolve_path_from_project(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()


def load_episode(
    episode_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = episode_dir.expanduser().resolve()
    npz_path = root / "episode.npz"
    metadata_path = root / "metadata.json"

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}

    required = ("qpos", "qvel", "ctrl", "action", "time")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"episode.npz is missing: {missing}")

    lengths = {
        key: int(arrays[key].shape[0])
        for key in required
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent episode lengths: {lengths}")

    return metadata, arrays


def infer_control_dt(arrays: dict[str, np.ndarray]) -> float:
    time_values = np.asarray(arrays["time"], dtype=np.float64)
    if len(time_values) < 2:
        return 0.05
    delta = np.diff(time_values)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if len(delta) == 0:
        return 0.05
    return float(np.median(delta))


def registry_task_name(metadata: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    raw = str(metadata.get("task", "tube_then_cap_grasp"))
    if raw == "bimanual_unscrew_cap":
        return "tube_then_cap_grasp"
    return raw


def make_task(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    control_dt: float,
) -> tuple[Any, Any]:
    model_value = (
        args.model
        if args.model is not None
        else metadata.get("model_path", "model/scenes/scene_mujoco_fast_tubes.xml")
    )
    reset_value = (
        args.reset_config
        if args.reset_config is not None
        else metadata.get(
            "reset_config",
            "configs/reset_single_tube_random.json",
        )
    )
    seed = (
        int(args.seed)
        if args.seed is not None
        else int(metadata.get("reset_seed", 0))
    )

    request = TaskRequest(
        task=registry_task_name(metadata, args.task_name),
        seed=seed,
        episode_index=int(metadata.get("episode_index", 0)),
        out_dir=args.output_dir,
        model=str(resolve_path_from_project(model_value)),
        reset_config=str(resolve_path_from_project(reset_value)),
        cameras=(),
        with_images=False,
        control_dt=float(control_dt),
        frame_skip=None,
        gl_backend=None,
        params={},
    )
    task = create_task(request)
    task.reset()
    scene = task.scene_query.resolve()
    task._initialize_screw_system(scene)
    return task, scene


def initialize_from_recorded_frame(
    task: Any,
    arrays: dict[str, np.ndarray],
    frame: int = 0,
) -> None:
    model = task.env.model
    data = task.env.data

    qpos = np.asarray(arrays["qpos"][frame], dtype=np.float64)
    qvel = np.asarray(arrays["qvel"][frame], dtype=np.float64)
    ctrl = np.asarray(arrays["ctrl"][frame], dtype=np.float64)

    if qpos.shape != data.qpos.shape:
        raise ValueError(
            f"Recorded qpos {qpos.shape} != model qpos {data.qpos.shape}"
        )
    if qvel.shape != data.qvel.shape:
        raise ValueError(
            f"Recorded qvel {qvel.shape} != model qvel {data.qvel.shape}"
        )
    if ctrl.shape != data.ctrl.shape:
        raise ValueError(
            f"Recorded ctrl {ctrl.shape} != model ctrl {data.ctrl.shape}"
        )

    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    if "time" in arrays:
        data.time = float(arrays["time"][frame])
    task.env.mujoco.mj_forward(model, data)


def load_adapter_class(path: Path) -> type:
    resolved = resolve_path_from_project(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Runtime adapter script does not exist: {resolved}"
        )
    spec = importlib.util.spec_from_file_location(
        "autolabsim_runtime_adapter_module",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import adapter script: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter_class = getattr(module, "ACTScrewRuntimeAdapter", None)
    if adapter_class is None:
        raise AttributeError(
            f"{resolved} does not export ACTScrewRuntimeAdapter"
        )
    return adapter_class


def make_runtime_adapter(
    args: argparse.Namespace,
    task: Any,
    scene: Any,
) -> Any | None:
    if not args.runtime_adapter:
        return None
    adapter_class = load_adapter_class(args.runtime_adapter_script)
    return adapter_class(
        task,
        scene,
        cap_grasp_distance=args.cap_grasp_distance,
        tube_grasp_distance=args.tube_grasp_distance,
        cap_place_distance=args.cap_place_distance,
        tube_slot_distance=args.tube_slot_distance,
        closed_ratio=args.closed_ratio,
        open_ratio=args.open_ratio,
        confirm_steps=args.confirm_steps,
        max_twist_step=args.max_twist_step,
    )


def id_to_name(mujoco: Any, model: Any, obj_type: Any, index: int) -> str:
    value = mujoco.mj_id2name(model, obj_type, int(index))
    return value if value is not None else f"id_{index}"


def build_joint_bindings(
    task: Any,
    include_grippers: bool,
    excluded: list[str],
) -> list[JointBinding]:
    model = task.env.model
    mujoco = task.env.mujoco

    excluded_ids: set[int] = set()
    excluded_names: set[str] = set()
    for value in excluded:
        try:
            excluded_ids.add(int(value))
        except ValueError:
            excluded_names.add(str(value))

    if not include_grippers:
        excluded_ids.add(
            int(task._gripper_id(task.runtime.cap_arm))
        )
        excluded_ids.add(
            int(task._gripper_id(task.runtime.tube_arm))
        )

    bindings: list[JointBinding] = []
    for actuator_id in range(int(model.nu)):
        actuator_name = id_to_name(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        if (
            actuator_id in excluded_ids
            or actuator_name in excluded_names
        ):
            continue

        trn_type = int(model.actuator_trntype[actuator_id])
        if trn_type not in (
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        ):
            print(
                f"[skip] actuator {actuator_id}:{actuator_name} "
                f"has non-joint transmission type {trn_type}"
            )
            continue

        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            print(
                f"[skip] actuator {actuator_id}:{actuator_name} "
                f"targets non-scalar joint type {joint_type}"
            )
            continue

        gear = float(model.actuator_gear[actuator_id, 0])
        if abs(gear) < 1e-12:
            print(
                f"[skip] actuator {actuator_id}:{actuator_name} has zero gear"
            )
            continue

        joint_name = id_to_name(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        bindings.append(
            JointBinding(
                actuator_id=actuator_id,
                actuator_name=actuator_name,
                joint_id=joint_id,
                joint_name=joint_name,
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                qvel_adr=int(model.jnt_dofadr[joint_id]),
                gear=gear,
            )
        )

    if not bindings:
        raise RuntimeError("No scalar joint-position actuator bindings found")

    print("\nTracked joint actuators:")
    for index, item in enumerate(bindings):
        print(
            f"  [{index:02d}] actuator={item.actuator_id:02d} "
            f"{item.actuator_name} -> joint={item.joint_id:02d} "
            f"{item.joint_name}, qpos={item.qpos_adr}, gear={item.gear:g}"
        )
    return bindings


def command_to_joint_position(
    action: np.ndarray,
    binding: JointBinding,
) -> float:
    # For a scalar joint transmission, actuator length = gear * q.
    # A position actuator's ctrl is the desired actuator length.
    return float(action[binding.actuator_id]) / binding.gear


def quaternion_angle_rad(
    quat_a: np.ndarray,
    quat_b: np.ndarray,
) -> float:
    qa = np.asarray(quat_a, dtype=np.float64)
    qb = np.asarray(quat_b, dtype=np.float64)
    qa /= max(np.linalg.norm(qa), 1e-12)
    qb /= max(np.linalg.norm(qb), 1e-12)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def site_pose_for_qpos(
    task: Any,
    qpos: np.ndarray,
    site_name: str,
    scratch: Any,
) -> tuple[np.ndarray, np.ndarray]:
    scratch.qpos[:] = np.asarray(qpos, dtype=np.float64)
    scratch.qvel[:] = 0.0
    task.env.mujoco.mj_forward(task.env.model, scratch)
    pos, quat = site_pose(
        task.env.model,
        scratch,
        task.env.mujoco,
        site_name,
    )
    return (
        np.asarray(pos, dtype=np.float64).copy(),
        np.asarray(quat, dtype=np.float64).copy(),
    )


def desired_qpos_from_action(
    actual_qpos: np.ndarray,
    action: np.ndarray,
    bindings: list[JointBinding],
) -> np.ndarray:
    desired = np.asarray(actual_qpos, dtype=np.float64).copy()
    for binding in bindings:
        desired[binding.qpos_adr] = command_to_joint_position(
            action,
            binding,
        )
    return desired


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return {
            "mean": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "max_abs": float("nan"),
            "p95_abs": float("nan"),
        }
    absolute = np.abs(finite)
    return {
        "mean": float(np.mean(finite)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "max_abs": float(np.max(absolute)),
        "p95_abs": float(np.percentile(absolute, 95)),
    }


def vector_norm_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        norm = np.abs(array)
    else:
        norm = np.linalg.norm(array, axis=-1)
    return scalar_stats(norm)


def replay(args: argparse.Namespace) -> dict[str, Any]:
    metadata, arrays = load_episode(args.episode_dir)

    control_dt = (
        float(args.control_dt)
        if args.control_dt is not None
        else infer_control_dt(arrays)
    )
    task, scene = make_task(args, metadata, control_dt)

    if args.init_mode == "recorded":
        initialize_from_recorded_frame(task, arrays, frame=0)
        default_start = 1
    else:
        default_start = 0

    start_index = (
        int(args.start_index)
        if args.start_index is not None
        else default_start
    )
    total_frames = int(arrays[args.action_key].shape[0])
    end_index = total_frames
    if args.max_steps is not None:
        end_index = min(end_index, start_index + int(args.max_steps))
    if not (0 <= start_index < end_index <= total_frames):
        raise ValueError(
            f"Invalid replay interval [{start_index}, {end_index}) "
            f"for {total_frames} frames"
        )

    bindings = build_joint_bindings(
        task,
        include_grippers=args.include_grippers,
        excluded=args.exclude_actuator,
    )
    adapter = make_runtime_adapter(args, task, scene)

    mujoco = task.env.mujoco
    model = task.env.model
    scratch_desired = mujoco.MjData(model)
    scratch_recorded = mujoco.MjData(model)

    arm_names: list[str] = []
    for name in (task.runtime.cap_arm, task.runtime.tube_arm):
        if name not in arm_names:
            arm_names.append(name)
    arm_sites = {
        name: str(task._gripper_site(name))
        for name in arm_names
    }

    num_steps = end_index - start_index
    num_joints = len(bindings)

    frame_index = np.empty(num_steps, dtype=np.int32)
    replay_time = np.empty(num_steps, dtype=np.float64)
    phases: list[str] = []

    q_command = np.empty((num_steps, num_joints), dtype=np.float64)
    q_actual = np.empty_like(q_command)
    q_recorded = np.empty_like(q_command)
    q_error_command = np.empty_like(q_command)
    q_error_recorded = np.empty_like(q_command)
    qvel_actual = np.empty_like(q_command)

    tcp: dict[str, dict[str, np.ndarray]] = {}
    for arm_name in arm_names:
        tcp[arm_name] = {
            "desired_pos": np.empty((num_steps, 3), dtype=np.float64),
            "actual_pos": np.empty((num_steps, 3), dtype=np.float64),
            "recorded_pos": np.empty((num_steps, 3), dtype=np.float64),
            "desired_quat": np.empty((num_steps, 4), dtype=np.float64),
            "actual_quat": np.empty((num_steps, 4), dtype=np.float64),
            "recorded_quat": np.empty((num_steps, 4), dtype=np.float64),
            "command_pos_error": np.empty(num_steps, dtype=np.float64),
            "command_rot_error_rad": np.empty(num_steps, dtype=np.float64),
            "recorded_pos_error": np.empty(num_steps, dtype=np.float64),
            "recorded_rot_error_rad": np.empty(num_steps, dtype=np.float64),
        }

    phase_array = arrays.get("phase")
    actions = np.asarray(arrays[args.action_key], dtype=np.float64)
    source_qpos = np.asarray(arrays["qpos"], dtype=np.float64)

    for output_index, source_index in enumerate(
        range(start_index, end_index)
    ):
        action = actions[source_index].copy()

        if adapter is not None:
            adapter.apply_constraints()
            adapter.before_step()

        obs, *_ = task.manager.step(action)

        if adapter is not None:
            constrained = adapter.apply_constraints()
            if constrained is not None:
                obs = constrained
            changed = adapter.after_step(obs, source_index)
            if changed:
                constrained = adapter.apply_constraints()
                obs = constrained or task.env.get_observation()

        actual_full_qpos = np.asarray(
            task.env.data.qpos,
            dtype=np.float64,
        ).copy()
        recorded_full_qpos = source_qpos[source_index].copy()
        desired_full_qpos = desired_qpos_from_action(
            actual_full_qpos,
            action,
            bindings,
        )

        frame_index[output_index] = source_index
        replay_time[output_index] = float(task.env.data.time)
        if phase_array is None:
            phases.append("")
        else:
            phases.append(str(phase_array[source_index]))

        for joint_index, binding in enumerate(bindings):
            command_value = command_to_joint_position(action, binding)
            actual_value = actual_full_qpos[binding.qpos_adr]
            recorded_value = recorded_full_qpos[binding.qpos_adr]

            q_command[output_index, joint_index] = command_value
            q_actual[output_index, joint_index] = actual_value
            q_recorded[output_index, joint_index] = recorded_value
            q_error_command[output_index, joint_index] = (
                actual_value - command_value
            )
            q_error_recorded[output_index, joint_index] = (
                actual_value - recorded_value
            )
            qvel_actual[output_index, joint_index] = float(
                task.env.data.qvel[binding.qvel_adr]
            )

        for arm_name, site_name in arm_sites.items():
            desired_pos, desired_quat = site_pose_for_qpos(
                task,
                desired_full_qpos,
                site_name,
                scratch_desired,
            )
            actual_pos, actual_quat = site_pose(
                model,
                task.env.data,
                mujoco,
                site_name,
            )
            recorded_pos, recorded_quat = site_pose_for_qpos(
                task,
                recorded_full_qpos,
                site_name,
                scratch_recorded,
            )

            actual_pos = np.asarray(actual_pos, dtype=np.float64)
            actual_quat = np.asarray(actual_quat, dtype=np.float64)

            arm_data = tcp[arm_name]
            arm_data["desired_pos"][output_index] = desired_pos
            arm_data["actual_pos"][output_index] = actual_pos
            arm_data["recorded_pos"][output_index] = recorded_pos
            arm_data["desired_quat"][output_index] = desired_quat
            arm_data["actual_quat"][output_index] = actual_quat
            arm_data["recorded_quat"][output_index] = recorded_quat
            arm_data["command_pos_error"][output_index] = float(
                np.linalg.norm(actual_pos - desired_pos)
            )
            arm_data["command_rot_error_rad"][output_index] = (
                quaternion_angle_rad(actual_quat, desired_quat)
            )
            arm_data["recorded_pos_error"][output_index] = float(
                np.linalg.norm(actual_pos - recorded_pos)
            )
            arm_data["recorded_rot_error_rad"][output_index] = (
                quaternion_angle_rad(actual_quat, recorded_quat)
            )

        if (
            output_index == 0
            or (output_index + 1) % max(1, int(args.print_every)) == 0
            or output_index + 1 == num_steps
        ):
            joint_rmse = float(
                np.sqrt(np.mean(q_error_command[output_index] ** 2))
            )
            tcp_text = " ".join(
                f"{arm}:tcp={tcp[arm]['command_pos_error'][output_index]*1000:.2f}mm"
                for arm in arm_names
            )
            print(
                f"frame={source_index:04d} "
                f"joint_rmse={math.degrees(joint_rmse):.3f}deg "
                f"{tcp_text}"
            )

    summary: dict[str, Any] = {
        "episode_dir": str(args.episode_dir.expanduser().resolve()),
        "task": registry_task_name(metadata, args.task_name),
        "reset_seed": (
            int(args.seed)
            if args.seed is not None
            else int(metadata.get("reset_seed", 0))
        ),
        "control_dt": control_dt,
        "init_mode": args.init_mode,
        "action_key": args.action_key,
        "start_index": start_index,
        "end_index_exclusive": end_index,
        "num_steps": num_steps,
        "runtime_adapter": bool(adapter is not None),
        "joint_units": "radian for hinge joints; metre for slide joints",
        "joint_command_error": {
            "all": scalar_stats(q_error_command.reshape(-1)),
            "per_joint": {},
        },
        "joint_replay_error": {
            "all": scalar_stats(q_error_recorded.reshape(-1)),
            "per_joint": {},
        },
        "tcp": {},
    }

    for index, binding in enumerate(bindings):
        summary["joint_command_error"]["per_joint"][
            binding.actuator_name
        ] = {
            "joint_name": binding.joint_name,
            **scalar_stats(q_error_command[:, index]),
        }
        summary["joint_replay_error"]["per_joint"][
            binding.actuator_name
        ] = {
            "joint_name": binding.joint_name,
            **scalar_stats(q_error_recorded[:, index]),
        }

    for arm_name, arm_data in tcp.items():
        summary["tcp"][arm_name] = {
            "site": arm_sites[arm_name],
            "command_position_error_m": scalar_stats(
                arm_data["command_pos_error"]
            ),
            "command_orientation_error_deg": scalar_stats(
                np.degrees(arm_data["command_rot_error_rad"])
            ),
            "recorded_position_error_m": scalar_stats(
                arm_data["recorded_pos_error"]
            ),
            "recorded_orientation_error_deg": scalar_stats(
                np.degrees(arm_data["recorded_rot_error_rad"])
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {
        "frame_index": frame_index,
        "time": replay_time,
        "phase": np.asarray(phases),
        "q_command": q_command,
        "q_actual": q_actual,
        "q_recorded": q_recorded,
        "q_error_command": q_error_command,
        "q_error_recorded": q_error_recorded,
        "qvel_actual": qvel_actual,
        "actuator_ids": np.asarray(
            [item.actuator_id for item in bindings],
            dtype=np.int32,
        ),
        "actuator_names": np.asarray(
            [item.actuator_name for item in bindings]
        ),
        "joint_names": np.asarray(
            [item.joint_name for item in bindings]
        ),
    }
    for arm_name, arm_data in tcp.items():
        for key, value in arm_data.items():
            npz_payload[f"tcp_{arm_name}_{key}"] = value

    np.savez_compressed(
        args.output_dir / "tracking_error.npz",
        **npz_payload,
    )

    csv_path = args.output_dir / "tracking_error.csv"
    fieldnames = ["frame", "time", "phase"]
    for binding in bindings:
        prefix = binding.actuator_name
        fieldnames.extend(
            [
                f"{prefix}.q_cmd",
                f"{prefix}.q_actual",
                f"{prefix}.q_recorded",
                f"{prefix}.error_cmd",
                f"{prefix}.error_recorded",
                f"{prefix}.qvel",
            ]
        )
    for arm_name in arm_names:
        fieldnames.extend(
            [
                f"{arm_name}.tcp_cmd_pos_error_m",
                f"{arm_name}.tcp_cmd_rot_error_deg",
                f"{arm_name}.tcp_recorded_pos_error_m",
                f"{arm_name}.tcp_recorded_rot_error_deg",
            ]
        )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(num_steps):
            row: dict[str, Any] = {
                "frame": int(frame_index[row_index]),
                "time": float(replay_time[row_index]),
                "phase": phases[row_index],
            }
            for joint_index, binding in enumerate(bindings):
                prefix = binding.actuator_name
                row[f"{prefix}.q_cmd"] = q_command[row_index, joint_index]
                row[f"{prefix}.q_actual"] = q_actual[
                    row_index, joint_index
                ]
                row[f"{prefix}.q_recorded"] = q_recorded[
                    row_index, joint_index
                ]
                row[f"{prefix}.error_cmd"] = q_error_command[
                    row_index, joint_index
                ]
                row[f"{prefix}.error_recorded"] = q_error_recorded[
                    row_index, joint_index
                ]
                row[f"{prefix}.qvel"] = qvel_actual[
                    row_index, joint_index
                ]
            for arm_name in arm_names:
                arm_data = tcp[arm_name]
                row[f"{arm_name}.tcp_cmd_pos_error_m"] = (
                    arm_data["command_pos_error"][row_index]
                )
                row[f"{arm_name}.tcp_cmd_rot_error_deg"] = math.degrees(
                    arm_data["command_rot_error_rad"][row_index]
                )
                row[f"{arm_name}.tcp_recorded_pos_error_m"] = (
                    arm_data["recorded_pos_error"][row_index]
                )
                row[f"{arm_name}.tcp_recorded_rot_error_deg"] = (
                    math.degrees(
                        arm_data["recorded_rot_error_rad"][row_index]
                    )
                )
            writer.writerow(row)

    summary_path = args.output_dir / "tracking_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.plot:
        save_plots(
            args.output_dir,
            replay_time,
            bindings,
            q_error_command,
            q_error_recorded,
            tcp,
        )

    print_summary(summary)
    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {args.output_dir / 'tracking_error.npz'}")
    if args.plot:
        print(f"  {args.output_dir / 'joint_tracking_error.png'}")
        print(f"  {args.output_dir / 'tcp_tracking_error.png'}")
    return summary


def save_plots(
    output_dir: Path,
    time_values: np.ndarray,
    bindings: list[JointBinding],
    q_error_command: np.ndarray,
    q_error_recorded: np.ndarray,
    tcp: dict[str, dict[str, np.ndarray]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib not installed; skipping plots")
        return

    relative_time = time_values - time_values[0]

    figure = plt.figure(figsize=(12, 7))
    axes = figure.add_subplot(111)
    for index, binding in enumerate(bindings):
        axes.plot(
            relative_time,
            np.degrees(q_error_command[:, index]),
            linewidth=1.0,
            label=binding.actuator_name,
        )
    axes.set_xlabel("Replay time (s)")
    axes.set_ylabel("Actual - command (deg)")
    axes.set_title("Joint command tracking error")
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(
        output_dir / "joint_tracking_error.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(12, 7))
    axes = figure.add_subplot(111)
    for arm_name, arm_data in tcp.items():
        axes.plot(
            relative_time,
            1000.0 * arm_data["command_pos_error"],
            linewidth=1.5,
            label=f"{arm_name}: command TCP error",
        )
        axes.plot(
            relative_time,
            1000.0 * arm_data["recorded_pos_error"],
            linewidth=1.0,
            linestyle="--",
            label=f"{arm_name}: replay TCP error",
        )
    axes.set_xlabel("Replay time (s)")
    axes.set_ylabel("Position error (mm)")
    axes.set_title("TCP position tracking error")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "tcp_tracking_error.png",
        dpi=180,
    )
    plt.close(figure)


def print_summary(summary: dict[str, Any]) -> None:
    command = summary["joint_command_error"]["all"]
    replay = summary["joint_replay_error"]["all"]

    print("\n" + "=" * 72)
    print("Tracking summary")
    print("=" * 72)
    print(
        "Joint command tracking: "
        f"RMSE={math.degrees(command['rmse']):.4f} deg, "
        f"MAE={math.degrees(command['mae']):.4f} deg, "
        f"max={math.degrees(command['max_abs']):.4f} deg"
    )
    print(
        "Joint replay fidelity:   "
        f"RMSE={math.degrees(replay['rmse']):.4f} deg, "
        f"MAE={math.degrees(replay['mae']):.4f} deg, "
        f"max={math.degrees(replay['max_abs']):.4f} deg"
    )

    for arm_name, data in summary["tcp"].items():
        command_pos = data["command_position_error_m"]
        command_rot = data["command_orientation_error_deg"]
        replay_pos = data["recorded_position_error_m"]
        replay_rot = data["recorded_orientation_error_deg"]
        print(
            f"{arm_name} TCP vs command: "
            f"pos RMSE={1000*command_pos['rmse']:.3f} mm, "
            f"p95={1000*command_pos['p95_abs']:.3f} mm, "
            f"rot RMSE={command_rot['rmse']:.3f} deg"
        )
        print(
            f"{arm_name} TCP vs recorded: "
            f"pos RMSE={1000*replay_pos['rmse']:.3f} mm, "
            f"p95={1000*replay_pos['p95_abs']:.3f} mm, "
            f"rot RMSE={replay_rot['rmse']:.3f} deg"
        )


def main() -> int:
    args = parse_args()
    replay(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
