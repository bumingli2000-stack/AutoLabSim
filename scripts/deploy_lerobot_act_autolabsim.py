#!/usr/bin/env python3
"""Deploy a trained LeRobot ACT policy in the AutoLabSim screw-cap task.

This version contains an ACTScrewRuntimeAdapter that recreates the hidden
simulation-side lifecycle used by the scripted demonstrations:

1. cap gripper -> cap and tube attachments after a valid cap grasp;
2. tube gripper -> tube attachment after a valid side grasp;
3. fixed tube state and ScrewCapSystem.engage() during unscrewing;
4. cumulative ratchet twist while the cap gripper is closed;
5. cap following after screw release;
6. cap release and tube return/release transitions.

The adapter only recreates simulation mechanics. It does not force a grasp.
Transitions require both a sufficiently closed gripper and geometric proximity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import av
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors

try:
    from autolabsim_Task.tasks import TaskRequest, create_task
    from autolabsim_Task.scene import (
        capture_free_joint_state,
        set_free_joint_pose,
        site_pose,
    )
except ModuleNotFoundError:
    # Compatibility with repositories using the original lowercase package.
    from autolabsim.tasks import TaskRequest, create_task
    from autolabsim.scene import (
        capture_free_joint_state,
        set_free_joint_pose,
        site_pose,
    )


CAMERA_SPECS: tuple[tuple[str, str], ...] = (
    ("image_overview_camera", "observation.images.overview_camera"),
    ("image_wrist_cam", "observation.images.wrist_cam"),
    ("image_wrist_cam1", "observation.images.wrist_cam1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a trained LeRobot ACT checkpoint in AutoLabSim."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint pretrained_model directory.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/scenes/scene_mujoco_fast_tubes.xml"),
    )
    parser.add_argument(
        "--reset-config",
        type=Path,
        default=Path("configs/reset_single_tube_random.json"),
    )
    parser.add_argument("--task-name", type=str, default="tube_then_cap_grasp")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--execute-steps", type=int, default=2)
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help=(
            "Enable ACT temporal ensembling. The original ACT default is "
            "0.01. When enabled, the policy replans every control step and "
            "--execute-steps is ignored."
        ),
    )
    parser.add_argument(
        "--arm-action-ema-alpha",
        type=float,
        default=1.0,
        help=(
            "Optional EMA smoothing coefficient for arm actuator commands. "
            "1.0 disables EMA; 0.3-0.6 is a useful range. Gripper dimensions "
            "are not smoothed."
        ),
    )
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
    )
    parser.add_argument("--gl-backend", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/act_rollouts"),
    )
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record the rollout to an H.264 MP4 file.",
    )
    parser.add_argument(
        "--video-layout",
        choices=("overview", "mosaic"),
        default="mosaic",
        help=(
            "overview records only overview_camera; mosaic records the "
            "overview view above two wrist-camera views."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output FPS. Defaults to round(1 / control_dt).",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=23,
        help="H.264 CRF quality. Lower is higher quality; 18-28 is typical.",
    )

    parser.add_argument(
        "--runtime-adapter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the screw-cap attachment and lifecycle adapter.",
    )
    parser.add_argument(
        "--cap-grasp-distance",
        type=float,
        default=0.055,
        help="Maximum cap-gripper-site to cap-body distance for cap grasp.",
    )
    parser.add_argument(
        "--tube-grasp-distance",
        type=float,
        default=0.080,
        help="Maximum tube-gripper-site to tube side-grasp target distance.",
    )
    parser.add_argument(
        "--cap-place-distance",
        type=float,
        default=0.120,
        help="Maximum cap position error from cap_place_pos before release.",
    )
    parser.add_argument(
        "--tube-slot-distance",
        type=float,
        default=0.120,
        help="Maximum tube position error from original slot before release.",
    )
    parser.add_argument(
        "--closed-ratio",
        type=float,
        default=0.50,
        help="Fraction from open to configured close value treated as closed.",
    )
    parser.add_argument(
        "--open-ratio",
        type=float,
        default=0.10,
        help="Fraction from open to configured close value treated as open.",
    )
    parser.add_argument(
        "--confirm-steps",
        type=int,
        default=3,
        help="Consecutive valid steps required for a state transition.",
    )
    parser.add_argument(
        "--max-twist-step",
        type=float,
        default=0.35,
        help="Maximum accepted incremental wrist rotation per control step.",
    )
    return parser.parse_args()


def resolve_existing(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def image_to_chw_float(image: Any, key: str) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{key} must have HWC RGB shape, got {array.shape}")

    if array.dtype == np.uint8:
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    else:
        float_array = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(float_array)):
            raise ValueError(f"{key} contains NaN or Inf")
        tensor = torch.from_numpy(np.ascontiguousarray(float_array))
        tensor = tensor.permute(2, 0, 1)
        if float(tensor.max()) > 1.5:
            tensor = tensor.div(255.0)
    return tensor.clamp_(0.0, 1.0)


def get_camera_image(obs: dict[str, Any], source_key: str) -> Any:
    if source_key in obs:
        return obs[source_key]

    images = obs.get("images")
    if isinstance(images, dict):
        camera_name = source_key.removeprefix("image_")
        for candidate in (camera_name, source_key):
            if candidate in images:
                return images[candidate]
        raise KeyError(
            f"{source_key!r} not found in obs['images']; "
            f"available={sorted(images.keys())}"
        )

    raise KeyError(
        f"{source_key!r} not found; observation keys={sorted(obs.keys())}"
    )



def _rgb_uint8(image: Any, key: str) -> np.ndarray:
    """Convert an HWC RGB observation into contiguous uint8 RGB."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{key} must have HWC RGB shape, got {array.shape}")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    float_array = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(float_array)):
        raise ValueError(f"{key} contains NaN or Inf")
    if float(float_array.max()) <= 1.5:
        float_array = float_array * 255.0
    return np.ascontiguousarray(np.clip(float_array, 0.0, 255.0).astype(np.uint8))


def _resize_nearest_rgb(
    image: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Dependency-free nearest-neighbour resize for RGB frames."""
    source_height, source_width = image.shape[:2]
    y_index = np.linspace(0, source_height - 1, target_height).astype(np.int64)
    x_index = np.linspace(0, source_width - 1, target_width).astype(np.int64)
    return np.ascontiguousarray(image[y_index][:, x_index])


def make_video_frame(
    obs: dict[str, Any],
    layout: str,
) -> np.ndarray:
    overview = _rgb_uint8(
        get_camera_image(obs, "image_overview_camera"),
        "image_overview_camera",
    )
    if layout == "overview":
        return overview

    if layout != "mosaic":
        raise ValueError(f"Unsupported video layout: {layout}")

    wrist_0 = _rgb_uint8(
        get_camera_image(obs, "image_wrist_cam"),
        "image_wrist_cam",
    )
    wrist_1 = _rgb_uint8(
        get_camera_image(obs, "image_wrist_cam1"),
        "image_wrist_cam1",
    )

    height, width = overview.shape[:2]
    lower_height = max(2, height // 2)
    left_width = width // 2
    right_width = width - left_width

    wrist_0 = _resize_nearest_rgb(wrist_0, lower_height, left_width)
    wrist_1 = _resize_nearest_rgb(wrist_1, lower_height, right_width)
    lower = np.concatenate((wrist_0, wrist_1), axis=1)
    return np.ascontiguousarray(np.concatenate((overview, lower), axis=0))


class RolloutVideoRecorder:
    """Stream RGB rollout frames to an H.264 MP4 through PyAV."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        layout: str,
        crf: int,
    ) -> None:
        if fps <= 0.0:
            raise ValueError(f"Video FPS must be positive, got {fps}")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.layout = str(layout)
        self.crf = int(crf)
        self.container: Any | None = None
        self.stream: Any | None = None
        self.frame_count = 0

    def append(self, obs: dict[str, Any]) -> None:
        rgb = make_video_frame(obs, self.layout)
        height, width = rgb.shape[:2]

        if width % 2 or height % 2:
            rgb = rgb[: height - (height % 2), : width - (width % 2)]
            height, width = rgb.shape[:2]

        if self.container is None:
            self.container = av.open(
                str(self.path),
                mode="w",
                options={"movflags": "+faststart"},
            )
            rate = Fraction(str(self.fps)).limit_denominator(1000)
            self.stream = self.container.add_stream("libx264", rate=rate)
            self.stream.width = int(width)
            self.stream.height = int(height)
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {
                "crf": str(self.crf),
                "preset": "medium",
            }
        elif (
            int(self.stream.width) != int(width)
            or int(self.stream.height) != int(height)
        ):
            raise ValueError(
                "Video frame size changed during rollout: "
                f"expected {(self.stream.height, self.stream.width)}, "
                f"got {(height, width)}"
            )

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self) -> None:
        if self.container is None or self.stream is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.container = None
        self.stream = None


def build_policy_observation(
    obs: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if "ctrl" not in obs:
        raise KeyError(
            f"Observation has no 'ctrl'; available={sorted(obs.keys())}"
        )

    state = np.asarray(obs["ctrl"], dtype=np.float32)
    if state.shape != (15,):
        raise ValueError(f"Expected ctrl shape (15,), got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError("Observation ctrl contains NaN or Inf")

    result: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(state.copy()),
    }
    for source_key, target_key in CAMERA_SPECS:
        result[target_key] = image_to_chw_float(
            get_camera_image(obs, source_key),
            source_key,
        )
    return result


def load_policy(
    checkpoint: Path,
    device: str,
    temporal_ensemble_coeff: float | None = None,
) -> tuple[ACTPolicy, Any, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print(f"Loading ACT checkpoint: {checkpoint}")
    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.config.device = device
    if temporal_ensemble_coeff is not None:
        policy.config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        policy.temporal_ensembler = ACTTemporalEnsembler(
            float(temporal_ensemble_coeff),
            int(policy.config.chunk_size),
        )
    policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    print(
        "Loaded policy: "
        f"chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"temporal_ensemble={temporal_ensemble_coeff}, "
        f"device={device}"
    )
    return policy, preprocessor, postprocessor


@torch.inference_mode()
def predict_action_chunk(
    policy: ACTPolicy,
    preprocessor: Any,
    postprocessor: Any,
    obs: dict[str, Any],
) -> np.ndarray:
    policy_input = build_policy_observation(obs)
    processed_input = preprocessor(policy_input)
    normalized_actions = policy.predict_action_chunk(processed_input)
    actions = postprocessor(normalized_actions)

    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)

    array = actions.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 15:
        raise ValueError(f"Expected action chunk [T,15], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Predicted action chunk contains NaN or Inf")
    return np.asarray(array, dtype=np.float64)


@torch.inference_mode()
def predict_ensembled_action(
    policy: ACTPolicy,
    preprocessor: Any,
    postprocessor: Any,
    obs: dict[str, Any],
) -> np.ndarray:
    """Predict one action using ACT's online temporal ensemble."""
    policy_input = build_policy_observation(obs)
    processed_input = preprocessor(policy_input)
    normalized_action = policy.select_action(processed_input)
    action = postprocessor(normalized_action)

    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action)
    array = action.detach().cpu().numpy()
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (15,):
        raise ValueError(f"Expected ensembled action shape (15,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Ensembled action contains NaN or Inf")
    return np.asarray(array, dtype=np.float64)


def clip_to_actuator_range(task: Any, action: np.ndarray) -> np.ndarray:
    clipped = np.asarray(action, dtype=np.float64).copy()
    ctrlrange = np.asarray(task.env.model.actuator_ctrlrange, dtype=np.float64)
    if ctrlrange.shape != (clipped.shape[0], 2):
        raise ValueError(
            f"ctrlrange {ctrlrange.shape} does not match action {clipped.shape}"
        )

    limited = getattr(task.env.model, "actuator_ctrllimited", None)
    limited_mask = (
        np.all(np.isfinite(ctrlrange), axis=1)
        if limited is None
        else np.asarray(limited).astype(bool)
    )
    clipped[limited_mask] = np.clip(
        clipped[limited_mask],
        ctrlrange[limited_mask, 0],
        ctrlrange[limited_mask, 1],
    )
    return clipped


@dataclass
class AdapterMetrics:
    cap_gripper: float
    tube_gripper: float
    cap_distance: float
    tube_distance: float
    cap_place_distance: float
    tube_slot_distance: float


class ACTScrewRuntimeAdapter:
    """Recreate scripted attachment and screw-state transitions online."""

    WAIT_CAP = "wait_cap_grasp"
    CAP_FOLLOW = "cap_holds_cap_and_tube"
    UNSCREW = "tube_fixed_unscrew"
    RELEASED_FOLLOW = "released_cap_follows_gripper"
    CAP_PLACED = "cap_placed_tube_follows_gripper"
    COMPLETE = "complete"

    def __init__(
        self,
        task: Any,
        scene: Any,
        *,
        cap_grasp_distance: float,
        tube_grasp_distance: float,
        cap_place_distance: float,
        tube_slot_distance: float,
        closed_ratio: float,
        open_ratio: float,
        confirm_steps: int,
        max_twist_step: float,
    ) -> None:
        self.task = task
        self.scene = scene
        self.stage = self.WAIT_CAP

        self.cap_grasp_distance = float(cap_grasp_distance)
        self.tube_grasp_distance = float(tube_grasp_distance)
        self.cap_place_distance = float(cap_place_distance)
        self.tube_slot_distance = float(tube_slot_distance)
        self.closed_ratio = float(closed_ratio)
        self.open_ratio = float(open_ratio)
        self.confirm_steps = max(1, int(confirm_steps))
        self.max_twist_step = max(1e-6, float(max_twist_step))

        self.cap_attachments: tuple[Any, ...] = ()
        self.tube_attachment: Any | None = None
        self.held_tube_state: Any | None = None
        self.placed_cap_state: Any | None = None

        self.cap_counter = 0
        self.tube_counter = 0
        self.cap_release_counter = 0
        self.tube_release_counter = 0

        self.accumulated_twist = 0.0
        self.ratchet_angle = max(
            1e-6,
            float(self.task.runtime.ratchet_angle),
        )
        self.ratchet_steps = max(
            1,
            int(self.task.runtime.steps_per_segment),
        )
        self.ratchet_phase = "idle"
        self.ratchet_cycle_index = 0
        self.ratchet_cycle_step = 0
        self.ratchet_cycle_start_twist = 0.0
        self.tube_ready_threshold = self._ratio_threshold(
            self.task.runtime.open_gripper,
            self.task.runtime.tube_close_gripper,
            0.90,
        )

        self.events: list[dict[str, Any]] = []
        self.last_metrics: AdapterMetrics | None = None

        self.cap_gripper_id = int(
            task._gripper_id(task.runtime.cap_arm)
        )
        self.tube_gripper_id = int(
            task._gripper_id(task.runtime.tube_arm)
        )
        self.cap_site = str(
            task._gripper_site(task.runtime.cap_arm)
        )
        self.tube_site = str(
            task._gripper_site(task.runtime.tube_arm)
        )

        self.cap_closed_threshold = self._ratio_threshold(
            task.runtime.open_gripper,
            task.runtime.cap_close_gripper,
            self.closed_ratio,
        )
        self.tube_closed_threshold = self._ratio_threshold(
            task.runtime.open_gripper,
            task.runtime.tube_close_gripper,
            self.closed_ratio,
        )
        self.cap_open_threshold = self._ratio_threshold(
            task.runtime.open_gripper,
            task.runtime.cap_close_gripper,
            self.open_ratio,
        )
        self.tube_open_threshold = self._ratio_threshold(
            task.runtime.open_gripper,
            task.runtime.tube_close_gripper,
            self.open_ratio,
        )

        print(
            "[adapter] thresholds: "
            f"cap_closed>={self.cap_closed_threshold:.2f}, "
            f"tube_closed>={self.tube_closed_threshold:.2f}, "
            f"cap_open<={self.cap_open_threshold:.2f}, "
            f"tube_open<={self.tube_open_threshold:.2f}, "
            f"tube_ready>={self.tube_ready_threshold:.2f}, "
            f"ratchet_angle={self.ratchet_angle:.4f}, "
            f"ratchet_steps={self.ratchet_steps}"
        )

    @staticmethod
    def _ratio_threshold(
        open_value: float,
        close_value: float,
        ratio: float,
    ) -> float:
        return float(open_value) + float(ratio) * (
            float(close_value) - float(open_value)
        )

    def _record_event(
        self,
        step: int,
        name: str,
        **extra: Any,
    ) -> None:
        event = {
            "step": int(step),
            "name": str(name),
            "stage": self.stage,
            **extra,
        }
        self.events.append(event)
        suffix = " ".join(f"{k}={v}" for k, v in extra.items())
        print(f"[adapter] step={step:04d} {name} {suffix}".rstrip())

    def metrics(self, obs: dict[str, Any]) -> AdapterMetrics:
        ctrl = np.asarray(obs["ctrl"], dtype=np.float64)

        cap_site_pos, _ = site_pose(
            self.task.env.model,
            self.task.env.data,
            self.task.env.mujoco,
            self.cap_site,
        )
        tube_site_pos, _ = site_pose(
            self.task.env.model,
            self.task.env.data,
            self.task.env.mujoco,
            self.tube_site,
        )

        cap_pos = np.asarray(
            self.task.scene_query.cap_position(self.scene),
            dtype=np.float64,
        )
        tube_pos = np.asarray(
            self.task.scene_query.tube_position(self.scene),
            dtype=np.float64,
        )

        tube_grasp_target = tube_pos + np.asarray(
            [0.0, 0.0, float(self.task.runtime.tube_grasp_height)],
            dtype=np.float64,
        )
        cap_place_target = np.asarray(
            self.task.runtime.cap_place_pos,
            dtype=np.float64,
        )

        result = AdapterMetrics(
            cap_gripper=float(ctrl[self.cap_gripper_id]),
            tube_gripper=float(ctrl[self.tube_gripper_id]),
            cap_distance=float(np.linalg.norm(cap_site_pos - cap_pos)),
            tube_distance=float(
                np.linalg.norm(tube_site_pos - tube_grasp_target)
            ),
            cap_place_distance=float(
                np.linalg.norm(cap_pos - cap_place_target)
            ),
            tube_slot_distance=float(
                np.linalg.norm(tube_pos - np.asarray(self.scene.slot_pos))
            ),
        )
        self.last_metrics = result
        return result

    def context(self) -> Any:
        if self.stage == self.CAP_FOLLOW:
            return self.task._execution_context(
                attachments=self.cap_attachments,
            )

        if self.stage in (self.UNSCREW, self.RELEASED_FOLLOW):
            if self.held_tube_state is None:
                return self.task._execution_context()
            return self.task._execution_context(
                fixed_joint_states=(
                    (self.scene.tube_joint, self.held_tube_state),
                ),
            )

        if self.stage == self.CAP_PLACED:
            fixed: tuple[Any, ...] = ()
            attachments: tuple[Any, ...] = ()
            if self.placed_cap_state is not None:
                fixed = (
                    (self.scene.cap_joint, self.placed_cap_state),
                )
            if self.tube_attachment is not None:
                attachments = (self.tube_attachment,)
            return self.task._execution_context(
                fixed_joint_states=fixed,
                attachments=attachments,
            )

        if self.stage == self.COMPLETE:
            if self.placed_cap_state is None:
                return self.task._execution_context()
            return self.task._execution_context(
                fixed_joint_states=(
                    (self.scene.cap_joint, self.placed_cap_state),
                ),
            )

        return self.task._execution_context()

    def apply_constraints(self) -> dict[str, Any] | None:
        return self.task.executor.apply_constraints(self.context())

    def before_step(self) -> None:
        screw = self.task.screw_system
        if (
            screw is not None
            and self.stage == self.UNSCREW
            and not screw.progress.released
        ):
            screw.set_commanded_twist(self.accumulated_twist)

    def _update_ratchet_twist(
        self,
        obs: dict[str, Any],
        step: int,
    ) -> None:
        """Reconstruct scripted cumulative screw progress from ratchet phases.

        The demonstrations do not derive cumulative screw progress from the
        measured gripper quaternion. Each planned twist waypoint carries an
        explicit cumulative ``twist_angle`` and the execution controller
        interpolates that value during ``steps_per_segment`` control steps.

        ACT predicts only actuator commands, so deployment reconstructs the
        omitted latent progress from the learned gripper sequence:

        closed + tube secured -> interpolate one ratchet segment;
        open                  -> wait for wrist rewind;
        closed again          -> interpolate the next segment.
        """
        screw = self.task.screw_system
        if (
            screw is None
            or self.stage != self.UNSCREW
            or screw.progress.released
        ):
            return

        metrics = self.last_metrics or self.metrics(obs)
        cap_closed = metrics.cap_gripper >= self.cap_closed_threshold
        cap_open = metrics.cap_gripper <= self.cap_open_threshold
        tube_ready = metrics.tube_gripper >= self.tube_ready_threshold

        if self.ratchet_phase == "wait_first_twist":
            if cap_closed and tube_ready:
                self.ratchet_phase = "twisting"
                self.ratchet_cycle_step = 0
                self.ratchet_cycle_start_twist = self.accumulated_twist
                self._record_event(
                    step,
                    "ratchet_cycle_started",
                    cycle=self.ratchet_cycle_index + 1,
                    start_twist=f"{self.accumulated_twist:.4f}",
                )

        elif self.ratchet_phase == "wait_open":
            if cap_open:
                self.ratchet_phase = "wait_regrip"
                self._record_event(
                    step,
                    "ratchet_gripper_opened",
                    cycle=self.ratchet_cycle_index,
                    twist=f"{self.accumulated_twist:.4f}",
                )

        elif self.ratchet_phase == "wait_regrip":
            if cap_closed and tube_ready:
                self.ratchet_phase = "twisting"
                self.ratchet_cycle_step = 0
                self.ratchet_cycle_start_twist = self.accumulated_twist
                self._record_event(
                    step,
                    "ratchet_cycle_started",
                    cycle=self.ratchet_cycle_index + 1,
                    start_twist=f"{self.accumulated_twist:.4f}",
                )

        if self.ratchet_phase != "twisting":
            screw.set_commanded_twist(self.accumulated_twist)
            return

        # One scripted twist target is executed over steps_per_segment frames.
        self.ratchet_cycle_step += 1
        remaining = max(
            0.0,
            float(self.task.runtime.release_angle)
            - self.ratchet_cycle_start_twist,
        )
        segment = min(self.ratchet_angle, remaining)
        fraction = min(
            1.0,
            self.ratchet_cycle_step / float(self.ratchet_steps),
        )
        self.accumulated_twist = float(
            np.clip(
                self.ratchet_cycle_start_twist + fraction * segment,
                0.0,
                float(self.task.runtime.release_angle),
            )
        )
        screw.set_commanded_twist(self.accumulated_twist)

        if int(step) % 5 == 0 or fraction >= 1.0:
            print(
                "[adapter] "
                f"ratchet_cycle={self.ratchet_cycle_index + 1} "
                f"progress={fraction:.2f} "
                f"twist={self.accumulated_twist:.4f}"
            )

        if fraction >= 1.0:
            self.ratchet_cycle_index += 1
            self._record_event(
                step,
                "ratchet_cycle_completed",
                cycle=self.ratchet_cycle_index,
                twist=f"{self.accumulated_twist:.4f}",
            )
            # The final cycle releases directly without another open/rewind.
            if (
                self.accumulated_twist
                >= float(self.task.runtime.release_angle) - 1e-6
            ):
                self.ratchet_phase = "complete"
            else:
                self.ratchet_phase = "wait_open"

    def after_step(
        self,
        obs: dict[str, Any],
        step: int,
    ) -> bool:
        """Update state after one environment step.

        Returns True when a lifecycle transition changes the constraint context.
        """
        metrics = self.metrics(obs)
        changed = False

        if self.stage == self.WAIT_CAP:
            valid = (
                metrics.cap_gripper >= self.cap_closed_threshold
                and metrics.cap_distance <= self.cap_grasp_distance
            )
            self.cap_counter = self.cap_counter + 1 if valid else 0

            if self.cap_counter >= self.confirm_steps:
                self.cap_attachments = self.task._capture_attachments(
                    self.cap_site,
                    (self.scene.cap_joint, self.scene.tube_joint),
                )
                self.stage = self.CAP_FOLLOW
                self._record_event(
                    step,
                    "cap_and_tube_attached_to_cap_gripper",
                    cap_distance=f"{metrics.cap_distance:.4f}",
                    cap_gripper=f"{metrics.cap_gripper:.2f}",
                )
                changed = True

        elif self.stage == self.CAP_FOLLOW:
            valid = (
                metrics.tube_gripper >= self.tube_closed_threshold
                and metrics.tube_distance <= self.tube_grasp_distance
            )
            self.tube_counter = self.tube_counter + 1 if valid else 0

            if self.tube_counter >= self.confirm_steps:
                self.tube_attachment = self.task._capture_attachments(
                    self.tube_site,
                    (self.scene.tube_joint,),
                )[0]
                self.held_tube_state = capture_free_joint_state(
                    self.task.env.model,
                    self.task.env.data,
                    self.task.env.mujoco,
                    self.scene.tube_joint,
                )

                screw = self.task.screw_system
                if screw is None:
                    raise RuntimeError("ScrewCapSystem is not initialized")
                screw.engage(self.task.env)
                screw.set_commanded_twist(0.0)

                self.accumulated_twist = 0.0
                self.ratchet_phase = "wait_first_twist"
                self.ratchet_cycle_index = 0
                self.ratchet_cycle_step = 0
                self.ratchet_cycle_start_twist = 0.0
                self.stage = self.UNSCREW
                self._record_event(
                    step,
                    "tube_attached_and_screw_engaged",
                    tube_distance=f"{metrics.tube_distance:.4f}",
                    tube_gripper=f"{metrics.tube_gripper:.2f}",
                )
                changed = True

        elif self.stage == self.UNSCREW:
            self._update_ratchet_twist(obs, step)
            screw = self.task.screw_system
            if screw is not None and screw.progress.released:
                screw.start_follow_after_release(self.task.env)
                self.stage = self.RELEASED_FOLLOW
                self._record_event(
                    step,
                    "screw_released_cap_follow_enabled",
                    twist=f"{screw.progress.twist_angle:.4f}",
                )
                changed = True

        elif self.stage == self.RELEASED_FOLLOW:
            valid = (
                metrics.cap_gripper <= self.cap_open_threshold
                and metrics.cap_place_distance <= self.cap_place_distance
            )
            self.cap_release_counter = (
                self.cap_release_counter + 1 if valid else 0
            )

            if self.cap_release_counter >= self.confirm_steps:
                screw = self.task.screw_system
                if screw is not None:
                    screw.release_follow()

                self.placed_cap_state = capture_free_joint_state(
                    self.task.env.model,
                    self.task.env.data,
                    self.task.env.mujoco,
                    self.scene.cap_joint,
                )
                self.stage = self.CAP_PLACED
                self._record_event(
                    step,
                    "cap_released_on_table",
                    cap_place_distance=f"{metrics.cap_place_distance:.4f}",
                )
                changed = True

        elif self.stage == self.CAP_PLACED:
            valid = (
                metrics.tube_gripper <= self.tube_open_threshold
                and metrics.tube_slot_distance <= self.tube_slot_distance
            )
            self.tube_release_counter = (
                self.tube_release_counter + 1 if valid else 0
            )

            if self.tube_release_counter >= self.confirm_steps:
                if self.scene.slot_quat is not None:
                    set_free_joint_pose(
                        self.task.env.model,
                        self.task.env.data,
                        self.task.env.mujoco,
                        self.scene.tube_joint,
                        self.scene.slot_pos,
                        self.scene.slot_quat,
                    )
                self.stage = self.COMPLETE
                self._record_event(
                    step,
                    "tube_released_in_rack",
                    tube_slot_distance=f"{metrics.tube_slot_distance:.4f}",
                )
                changed = True

        return changed

    def summary(self) -> dict[str, Any]:
        metrics = self.last_metrics
        return {
            "stage": self.stage,
            "events": self.events,
            "accumulated_twist": float(self.accumulated_twist),
            "ratchet_phase": self.ratchet_phase,
            "ratchet_cycle_index": int(self.ratchet_cycle_index),
            "ratchet_angle": float(self.ratchet_angle),
            "ratchet_steps": int(self.ratchet_steps),
            "thresholds": {
                "cap_closed": self.cap_closed_threshold,
                "tube_closed": self.tube_closed_threshold,
                "cap_open": self.cap_open_threshold,
                "tube_open": self.tube_open_threshold,
                "cap_grasp_distance": self.cap_grasp_distance,
                "tube_grasp_distance": self.tube_grasp_distance,
                "cap_place_distance": self.cap_place_distance,
                "tube_slot_distance": self.tube_slot_distance,
            },
            "last_metrics": None
            if metrics is None
            else {
                "cap_gripper": metrics.cap_gripper,
                "tube_gripper": metrics.tube_gripper,
                "cap_distance": metrics.cap_distance,
                "tube_distance": metrics.tube_distance,
                "cap_place_distance": metrics.cap_place_distance,
                "tube_slot_distance": metrics.tube_slot_distance,
            },
        }


def make_task(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    request = TaskRequest(
        task=args.task_name,
        seed=int(args.seed),
        episode_index=0,
        out_dir=args.output_dir,
        model=str(resolve_existing(args.model, "MuJoCo model")),
        reset_config=str(
            resolve_existing(args.reset_config, "reset config")
        ),
        cameras=("overview_camera", "wrist_cam", "wrist_cam1"),
        with_images=True,
        control_dt=float(args.control_dt),
        frame_skip=None,
        gl_backend=args.gl_backend,
        params={},
    )
    task = create_task(request)

    task.reset()
    scene = task.scene_query.resolve()
    task._initialize_screw_system(scene)

    initial_action = np.asarray(
        task.env.data.ctrl,
        dtype=np.float64,
    ).copy()
    initial_action[task._gripper_id(task.runtime.tube_arm)] = (
        task.runtime.open_gripper
    )
    initial_action[task._gripper_id(task.runtime.cap_arm)] = (
        task.runtime.open_gripper
    )

    obs = task.env.get_observation()
    for _ in range(max(0, int(task.runtime.settle_steps))):
        obs, *_ = task.manager.step(initial_action)

    return task, scene, obs


def rollout(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = resolve_existing(args.checkpoint, "checkpoint")
    if not checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be pretrained_model directory: {checkpoint}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, postprocessor = load_policy(
        checkpoint,
        args.device,
        args.temporal_ensemble_coeff,
    )
    task, scene, obs = make_task(args)
    policy.reset()

    first_policy_obs = build_policy_observation(obs)
    print("Runtime observation:")
    for key, value in first_policy_obs.items():
        print(
            f"  {key}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}"
        )

    first_chunk = predict_action_chunk(
        policy,
        preprocessor,
        postprocessor,
        obs,
    )
    print(
        "First predicted chunk: "
        f"shape={first_chunk.shape}, "
        f"min={first_chunk.min():.4f}, "
        f"max={first_chunk.max():.4f}"
    )

    if args.inference_only:
        return {
            "mode": "inference_only",
            "checkpoint": str(checkpoint),
            "seed": int(args.seed),
            "action_chunk_shape": list(first_chunk.shape),
            "action_min": float(first_chunk.min()),
            "action_max": float(first_chunk.max()),
        }

    use_temporal_ensemble = args.temporal_ensemble_coeff is not None
    execute_steps = 1 if use_temporal_ensemble else int(args.execute_steps)
    if execute_steps <= 0:
        raise ValueError("--execute-steps must be positive")
    if execute_steps > first_chunk.shape[0]:
        raise ValueError(
            f"--execute-steps={execute_steps} exceeds "
            f"chunk length {first_chunk.shape[0]}"
        )
    ema_alpha = float(args.arm_action_ema_alpha)
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("--arm-action-ema-alpha must be in (0, 1]")

    adapter = (
        ACTScrewRuntimeAdapter(
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
        if args.runtime_adapter
        else None
    )

    rollout_dir = (
        args.output_dir
        / f"seed_{int(args.seed):04d}_steps_{int(args.max_steps):04d}"
    )
    rollout_dir.mkdir(parents=True, exist_ok=True)

    video_fps = (
        float(args.video_fps)
        if args.video_fps is not None
        else float(round(1.0 / float(args.control_dt)))
    )
    video_path = rollout_dir / f"rollout_{args.video_layout}.mp4"
    video_recorder = (
        RolloutVideoRecorder(
            video_path,
            fps=video_fps,
            layout=args.video_layout,
            crf=args.video_crf,
        )
        if args.record_video
        else None
    )
    if video_recorder is not None:
        video_recorder.append(obs)
        print(
            f"Recording video: {video_path} "
            f"({args.video_layout}, {video_fps:g} FPS)"
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    times: list[float] = []
    stages: list[str] = []

    step = 0
    chunk_count = 0
    start_wall = time.perf_counter()
    current_chunk = first_chunk
    previous_executed_action: np.ndarray | None = None
    if adapter is not None:
        gripper_ids = {adapter.cap_gripper_id, adapter.tube_gripper_id}
    else:
        gripper_ids = {13, 14}
    arm_dims = np.asarray(
        [i for i in range(15) if i not in gripper_ids],
        dtype=np.int64,
    )

    while step < int(args.max_steps):
        if use_temporal_ensemble:
            num_to_execute = 1
        else:
            if chunk_count > 0:
                current_chunk = predict_action_chunk(
                    policy,
                    preprocessor,
                    postprocessor,
                    obs,
                )
            num_to_execute = min(
                execute_steps,
                current_chunk.shape[0],
                int(args.max_steps) - step,
            )

        for local_index in range(num_to_execute):
            if use_temporal_ensemble:
                action = predict_ensembled_action(
                    policy,
                    preprocessor,
                    postprocessor,
                    obs,
                )
            else:
                action = current_chunk[local_index].copy()

            if previous_executed_action is not None and ema_alpha < 1.0:
                action[arm_dims] = (
                    ema_alpha * action[arm_dims]
                    + (1.0 - ema_alpha)
                    * previous_executed_action[arm_dims]
                )

            if not args.no_clip:
                action = clip_to_actuator_range(task, action)

            if adapter is not None:
                adapter.apply_constraints()
                adapter.before_step()

            obs, *_ = task.manager.step(action)

            if adapter is not None:
                constrained_obs = adapter.apply_constraints()
                if constrained_obs is not None:
                    obs = constrained_obs

                changed = adapter.after_step(obs, step + 1)
                if changed:
                    constrained_obs = adapter.apply_constraints()
                    obs = constrained_obs or task.env.get_observation()

            previous_executed_action = np.asarray(
                action,
                dtype=np.float64,
            ).copy()

            states.append(
                np.asarray(obs["ctrl"], dtype=np.float32).copy()
            )
            actions.append(
                np.asarray(action, dtype=np.float32).copy()
            )
            times.append(
                float(obs.get("time", step * args.control_dt))
            )
            stages.append(
                adapter.stage if adapter is not None else "disabled"
            )
            if video_recorder is not None:
                video_recorder.append(obs)

            step += 1

            if step % max(1, int(args.print_every)) == 0:
                progress = (
                    task.screw_system.progress
                    if task.screw_system is not None
                    else None
                )
                stage = adapter.stage if adapter is not None else "disabled"
                metric_text = ""
                if adapter is not None and adapter.last_metrics is not None:
                    m = adapter.last_metrics
                    metric_text = (
                        f" cap_g={m.cap_gripper:.1f}"
                        f" tube_g={m.tube_gripper:.1f}"
                        f" cap_d={m.cap_distance:.3f}"
                        f" tube_d={m.tube_distance:.3f}"
                    )
                print(
                    f"step={step:04d} "
                    f"chunks={chunk_count + 1:04d} "
                    f"stage={stage} "
                    f"engaged={bool(progress.engaged) if progress else False} "
                    f"released={bool(progress.released) if progress else False} "
                    f"twist={float(progress.twist_angle) if progress else 0.0:.4f}"
                    f"{metric_text}"
                )

        chunk_count += 1

    if video_recorder is not None:
        video_recorder.close()
        print(
            f"Saved video: {video_path} "
            f"({video_recorder.frame_count} frames)"
        )

    elapsed = time.perf_counter() - start_wall
    progress = (
        task.screw_system.progress
        if task.screw_system is not None
        else None
    )

    np.savez_compressed(
        rollout_dir / "rollout.npz",
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        time=np.asarray(times, dtype=np.float64),
        stage=np.asarray(stages),
    )

    summary = {
        "checkpoint": str(checkpoint),
        "seed": int(args.seed),
        "control_dt": float(args.control_dt),
        "max_steps": int(args.max_steps),
        "executed_steps": int(step),
        "execute_steps_per_chunk": int(execute_steps),
        "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
        "arm_action_ema_alpha": float(ema_alpha),
        "predicted_chunks": int(chunk_count),
        "wall_time_s": float(elapsed),
        "mean_control_rate_hz": (
            float(step / elapsed) if elapsed > 0 else None
        ),
        "video": (
            {
                "path": str(video_path),
                "layout": args.video_layout,
                "fps": video_fps,
                "frames": int(video_recorder.frame_count),
            }
            if video_recorder is not None
            else None
        ),
        "scene": {
            "tube_joint": getattr(scene, "tube_joint", None),
            "cap_joint": getattr(scene, "cap_joint", None),
            "slot_index": getattr(scene, "slot_index", None),
            "slot_name": getattr(scene, "slot_name", None),
        },
        "screw_progress": {
            "engaged": bool(progress.engaged) if progress else False,
            "released": bool(progress.released) if progress else False,
            "twist_angle": (
                float(progress.twist_angle) if progress else 0.0
            ),
            "lift_distance": (
                float(progress.lift_distance) if progress else 0.0
            ),
        },
        "runtime_adapter": (
            adapter.summary()
            if adapter is not None
            else {"enabled": False}
        ),
    }

    with (rollout_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nRollout complete")
    print(f"  output: {rollout_dir}")
    print(f"  steps: {step}")
    if video_recorder is not None:
        print(f"  video: {video_path}")
    print(
        "  adapter stage: "
        f"{adapter.stage if adapter is not None else 'disabled'}"
    )
    print(
        "  screw released: "
        f"{summary['screw_progress']['released']}"
    )
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = rollout(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())