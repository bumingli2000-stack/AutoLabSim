from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim.episode_io import load_episode


def episode_dirs(path: Path) -> list[Path]:
    if (path / "metadata.json").exists() and (path / "episode.npz").exists():
        return [path]
    episodes = [candidate.parent for candidate in path.glob("**/metadata.json") if (candidate.parent / "episode.npz").exists()]
    return sorted(episodes)


def image_keys(arrays: dict[str, np.ndarray]) -> list[str]:
    return sorted(key for key in arrays if key.startswith("image_"))


def camera_name_from_key(key: str) -> str:
    return key.removeprefix("image_")


def lerobot_image_feature_name(camera: str) -> str:
    return f"observation.images.{camera}"


def state_array(arrays: dict[str, np.ndarray], source: str) -> np.ndarray:
    if source == "state":
        return arrays["state"]
    if source == "qpos":
        return arrays["qpos"]
    if source == "qpos_qvel":
        return np.concatenate([arrays["qpos"], arrays["qvel"]], axis=1)
    if source == "qpos_ctrl":
        return np.concatenate([arrays["qpos"], arrays["ctrl"]], axis=1)
    if source == "qpos_qvel_ctrl":
        return np.concatenate([arrays["qpos"], arrays["qvel"], arrays["ctrl"]], axis=1)
    raise ValueError(f"Unsupported state source: {source}")


def task_text(metadata: dict[str, Any], args: argparse.Namespace) -> str:
    if args.task:
        return args.task
    if metadata.get("task") == "tube_then_cap_grasp":
        return "grasp the tube with ur5e1, lift it, then grasp the cap with ur5e"
    if metadata.get("task") == "tube_grasp":
        return "grasp the tube with the robot arm"
    return "autolabsim manipulation task"


def infer_schema(episodes: list[Path], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    first_metadata, first_arrays = load_episode(episodes[0])
    states = state_array(first_arrays, args.state_source)
    actions = np.asarray(first_arrays["action"])
    if states.ndim != 2:
        raise ValueError(f"Expected state array shape TxD, got {states.shape}")
    if actions.ndim != 2:
        raise ValueError(f"Expected action array shape TxA, got {actions.shape}")
    if len(states) != len(actions):
        raise ValueError(f"State/action length mismatch in {episodes[0]}: {len(states)} vs {len(actions)}")

    selected_image_keys = image_keys(first_arrays)
    if args.cameras:
        requested = {camera.strip() for camera in args.cameras.split(",") if camera.strip()}
        selected_image_keys = [f"image_{camera}" for camera in sorted(requested)]

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (int(states.shape[1]),),
            "names": [args.state_source],
        },
        "action": {
            "dtype": "float32",
            "shape": (int(actions.shape[1]),),
            "names": ["mujoco_ctrl"],
        },
    }

    cameras: dict[str, Any] = {}
    for key in selected_image_keys:
        if key not in first_arrays:
            raise ValueError(f"Requested camera '{camera_name_from_key(key)}' is not saved in {episodes[0]}")
        images = first_arrays[key]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected {key} shape TxHxWx3, got {images.shape}")
        camera = camera_name_from_key(key)
        feature_name = lerobot_image_feature_name(camera)
        features[feature_name] = {
            "dtype": "video" if args.use_videos else "image",
            "shape": (3, int(images.shape[1]), int(images.shape[2])),
            "names": ["channels", "height", "width"],
        }
        cameras[key] = {
            "camera": camera,
            "feature_name": feature_name,
            "shape": tuple(int(x) for x in images.shape[1:]),
        }

    summary = {
        "episodes": len(episodes),
        "state_source": args.state_source,
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "fps": int(args.fps),
        "cameras": [item["camera"] for item in cameras.values()],
        "first_episode": str(episodes[0]),
        "first_task": task_text(first_metadata, args),
    }
    return features, {"summary": summary, "cameras": cameras}


def validate_episode_arrays(
    episode_dir: Path,
    arrays: dict[str, np.ndarray],
    selected_image_keys: list[str],
    state_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(state_array(arrays, state_source), dtype=np.float32)
    actions = np.asarray(arrays["action"], dtype=np.float32)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected 2D state/action arrays in {episode_dir}")
    if len(states) != len(actions):
        raise ValueError(f"State/action length mismatch in {episode_dir}: {len(states)} vs {len(actions)}")
    for key in selected_image_keys:
        if key not in arrays:
            raise ValueError(f"Missing {key} in {episode_dir}")
        if len(arrays[key]) != len(states):
            raise ValueError(f"Image/state length mismatch for {key} in {episode_dir}: {len(arrays[key])} vs {len(states)}")
    return states, actions


def convert(args: argparse.Namespace) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'lerobot'. Install it first, for example: python -m pip install lerobot"
        ) from exc

    source = Path(args.source)
    episodes = episode_dirs(source)
    if not episodes:
        raise RuntimeError(f"No AutoLabSim episodes found under {source}")

    features, info = infer_schema(episodes, args)
    selected_image_keys = list(info["cameras"])
    print(json.dumps(info["summary"], indent=2))

    if args.dry_run:
        print("dry_run: no files written")
        return

    root = Path(args.out)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {root}. Use --overwrite to replace it.")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=int(args.fps),
        robot_type=args.robot_type,
        features=features,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
        vcodec=args.vcodec,
    )

    for episode_index, episode_dir in enumerate(episodes):
        metadata, arrays = load_episode(episode_dir)
        states, actions = validate_episode_arrays(episode_dir, arrays, selected_image_keys, args.state_source)
        task = task_text(metadata, args)
        print(f"[{episode_index + 1}/{len(episodes)}] {episode_dir} frames={len(states)} task={task!r}")

        for frame_index in range(len(states)):
            frame: dict[str, Any] = {
                "observation.state": states[frame_index],
                "action": actions[frame_index],
                "task": task,
            }
            for key in selected_image_keys:
                camera = camera_name_from_key(key)
                frame[lerobot_image_feature_name(camera)] = arrays[key][frame_index]
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=not args.no_parallel_encoding)

    dataset.finalize()
    write_conversion_metadata(root, source, episodes, features, info, args)
    print(f"lerobot_dataset: {root}")


def write_conversion_metadata(
    root: Path,
    source: Path,
    episodes: list[Path],
    features: dict[str, Any],
    info: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    metadata = {
        "source": str(source),
        "episode_dirs": [str(path) for path in episodes],
        "features": features,
        "summary": info["summary"],
        "args": vars(args),
    }
    with (root / "autolabsim_conversion.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AutoLabSim NPZ episodes to a LeRobotDataset.")
    parser.add_argument("source", help="AutoLabSim episode directory, or a batch directory containing episodes.")
    parser.add_argument("--out", required=True, help="Output LeRobot dataset root directory.")
    parser.add_argument("--repo-id", default="autolabsim/tube-task", help="LeRobot repo_id stored in metadata.")
    parser.add_argument("--robot-type", default="autolabsim_dual_ur5e", help="Robot type stored in metadata.")
    parser.add_argument("--task", default=None, help="Natural language task text. Defaults from AutoLabSim metadata.")
    parser.add_argument("--fps", type=int, default=20, help="Dataset FPS. Default matches control_dt=0.05.")
    parser.add_argument(
        "--state-source",
        choices=("state", "qpos", "qpos_qvel", "qpos_ctrl", "qpos_qvel_ctrl"),
        default="state",
        help="Which AutoLabSim arrays to export as observation.state.",
    )
    parser.add_argument("--cameras", default=None, help="Comma-separated camera names to export. Defaults to all saved cameras.")
    parser.add_argument("--no-videos", dest="use_videos", action="store_false", help="Store images instead of encoded videos.")
    parser.add_argument("--vcodec", default="h264", choices=("h264", "hevc", "libsvtav1"), help="Video codec for LeRobot video features.")
    parser.add_argument("--video-backend", default=None, help="Override LeRobot video backend.")
    parser.add_argument("--image-writer-processes", type=int, default=0, help="Parallel image writer processes.")
    parser.add_argument("--image-writer-threads", type=int, default=0, help="Parallel image writer threads.")
    parser.add_argument("--no-parallel-encoding", action="store_true", help="Disable parallel video encoding in save_episode.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output directory before conversion.")
    parser.add_argument("--dry-run", action="store_true", help="Print inferred schema without writing files.")
    parser.set_defaults(use_videos=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        convert(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
