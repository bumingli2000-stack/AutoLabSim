from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from autolabsim_Task.episode_io import save_episode
from autolabsim_ACT.autolabsim.learning.act_dataset import NormalizationStats
from autolabsim_ACT.autolabsim.learning.act_model import ACTConfig, ACTPolicy
from autolabsim_Task.mujoco_env import AutoLabMuJoCoEnv, EnvConfig
from autolabsim_Task.recorder import EpisodeRecorder
from autolabsim_Task.scene_profile import resolve_scene_spec, scene_rooted_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a trained ACT checkpoint in AutoLabSim.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", default="fast_tubes")
    parser.add_argument("--model", default=None)
    parser.add_argument("--reset-config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--execute-steps", type=int, default=5, help="Actions executed before replanning.")
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--gl-backend", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def observation_to_tensors(
    obs: dict,
    *,
    state_key: str,
    camera_names: tuple[str, ...],
    image_size: tuple[int, int],
    stats: NormalizationStats,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = np.asarray(obs[state_key], dtype=np.float32)
    state = (state - stats.state_mean) / stats.state_std
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    images = []
    for camera in camera_names:
        image = np.asarray(obs["images"][camera])[..., :3]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        if tuple(tensor.shape[-2:]) != image_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=image_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        images.append(tensor)
    image_tensor = torch.stack(images, dim=0).unsqueeze(0).to(device)
    return state_tensor, image_tensor


def clamp_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    result = action.copy()
    finite_low = np.isfinite(low)
    finite_high = np.isfinite(high)
    result[finite_low] = np.maximum(result[finite_low], low[finite_low])
    result[finite_high] = np.minimum(result[finite_high], high[finite_high])
    return result


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "autolabsim_act_checkpoint_v1":
        raise ValueError("Unsupported checkpoint format")

    model_config = ACTConfig(**checkpoint["model_config"])
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    camera_names = tuple(checkpoint["camera_names"])
    state_key = str(checkpoint["state_key"])
    image_size = tuple(int(x) for x in checkpoint["image_size"])
    device = torch.device(args.device)
    model = ACTPolicy(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scene_spec = resolve_scene_spec(
        args.scene,
        model=args.model,
        reset_config=args.reset_config,
        cameras=camera_names,
    )
    model_path, reset_path = scene_rooted_path(scene_spec, PROJECT_ROOT)
    env = AutoLabMuJoCoEnv(
        EnvConfig(
            model_path=str(model_path),
            reset_config=str(reset_path) if reset_path is not None else None,
            reset_seed=args.seed,
            cameras=camera_names,
            render_images=True,
            control_dt=args.control_dt,
            frame_skip=args.frame_skip,
            gl_backend=args.gl_backend,
        )
    )
    recorder = EpisodeRecorder(cameras=camera_names, with_images=True)
    viewer = None
    try:
        obs = env.reset()
        if np.asarray(obs[state_key]).shape[-1] != model_config.state_dim:
            raise ValueError(
                f"Deployment state dimension {np.asarray(obs[state_key]).shape[-1]} "
                f"does not match checkpoint {model_config.state_dim}"
            )
        if env.action_dim != model_config.action_dim:
            raise ValueError(
                f"Environment action_dim={env.action_dim}, checkpoint action_dim={model_config.action_dim}"
            )
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        step = 0
        previous_action = np.asarray(env.data.ctrl, dtype=np.float64).copy()
        while step < args.max_steps:
            state_tensor, image_tensor = observation_to_tensors(
                obs,
                state_key=state_key,
                camera_names=camera_names,
                image_size=image_size,
                stats=stats,
                device=device,
            )
            with torch.inference_mode():
                output = model(state_tensor, image_tensor)
            chunk = output["actions"][0].float().cpu().numpy()
            chunk = chunk * stats.action_std[None, :] + stats.action_mean[None, :]

            for local_index in range(min(args.execute_steps, len(chunk))):
                action = clamp_action(chunk[local_index], env.action_low, env.action_high)
                smoothing = float(np.clip(args.action_smoothing, 0.0, 0.999))
                action = smoothing * previous_action + (1.0 - smoothing) * action
                obs, _, terminated, truncated, _ = env.step(action)
                recorder.record(obs, action, phase="act_policy")
                previous_action = action
                step += 1
                if viewer is not None:
                    viewer.sync()
                if terminated or truncated or step >= args.max_steps:
                    break
        print(f"rollout_steps: {step}")

        if args.out_dir is not None:
            metadata = {
                "format": "autolabsim_act_rollout_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(args.checkpoint),
                "scene": args.scene,
                "seed": args.seed,
                "steps": step,
                "camera_names": list(camera_names),
                "state_key": state_key,
                "warning": (
                    "This generic runner applies only actuator actions. Script-only attachment, "
                    "fixed-joint, or screw-twist state transitions require a task-specific adapter."
                ),
            }
            save_episode(args.out_dir, metadata, recorder.to_arrays())
            print(f"saved_rollout: {args.out_dir}")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
