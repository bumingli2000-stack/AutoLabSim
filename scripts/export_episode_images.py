from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim_Task.episode_io import load_episode


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def write_png_sequence(images: np.ndarray, out_dir: Path, stride: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for frame_id in range(0, len(images), stride):
        out_path = out_dir / f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(out_path), rgb_to_bgr(images[frame_id]))
        written += 1
    return written


def write_mp4(images: np.ndarray, out_path: Path, stride: int, fps: float) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = images.shape[1], images.shape[2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    written = 0
    try:
        for frame_id in range(0, len(images), stride):
            writer.write(rgb_to_bgr(images[frame_id]))
            written += 1
    finally:
        writer.release()
    return written


def episode_dirs(path: Path) -> list[Path]:
    if (path / "metadata.json").exists() and (path / "episode.npz").exists():
        return [path]

    episodes = [candidate.parent for candidate in path.glob("**/metadata.json") if (candidate.parent / "episode.npz").exists()]
    return sorted(episodes)


def available_cameras(arrays: dict[str, np.ndarray]) -> list[str]:
    return sorted(key.removeprefix("image_") for key in arrays if key.startswith("image_"))


def parse_camera_selection(value: str) -> list[str] | None:
    cameras = [camera.strip() for camera in value.split(",") if camera.strip()]
    if not cameras:
        raise argparse.ArgumentTypeError("--camera must be 'all' or a comma-separated camera list")
    if len(cameras) == 1 and cameras[0].lower() == "all":
        return None
    return cameras


def export_one_episode(
    episode_dir: Path,
    camera: str,
    out: Path,
    export_format: str,
    stride: int,
    fps: float,
    index: int | None = None,
) -> dict[str, object]:
    metadata, arrays = load_episode(episode_dir)
    image_key = f"image_{camera}"
    if image_key not in arrays:
        available = available_cameras(arrays)
        raise KeyError(f"{image_key} not found in {episode_dir}. Available cameras: {available}")

    images = arrays[image_key]
    if images.ndim != 4:
        raise ValueError(f"Expected image stack with shape TxHxWx3, got {images.shape}")

    stem = f"{index:03d}" if index is not None else camera
    result: dict[str, object] = {
        "episode": str(episode_dir),
        "camera": camera,
        "frames": int(len(images)),
        "shape": list(images.shape),
        "reset_seed": metadata.get("reset_seed"),
        "slot_name": metadata.get("slot_name"),
        "slot_index": metadata.get("slot_index"),
    }

    if export_format in ("png", "both"):
        if index is None:
            png_dir = out if export_format == "png" else out / f"{camera}_png"
        else:
            png_dir = out / stem
        result["png_dir"] = str(png_dir)
        result["png_frames_written"] = write_png_sequence(images, png_dir, stride)

    if export_format in ("mp4", "both"):
        if index is None:
            mp4_path = out if export_format == "mp4" and out.suffix else out / f"{camera}.mp4"
        else:
            mp4_path = out / f"{stem}.mp4"
        result["mp4_path"] = str(mp4_path)
        result["mp4_frames_written"] = write_mp4(images, mp4_path, stride, fps)

    return result


def export_episode_cameras(
    episode_dir: Path,
    cameras: list[str] | None,
    out: Path,
    export_format: str,
    stride: int,
    fps: float,
    index: int | None,
    multi_episode: bool,
    multi_camera: bool,
) -> list[dict[str, object]]:
    metadata, arrays = load_episode(episode_dir)
    selected_cameras = available_cameras(arrays) if cameras is None else cameras
    if not selected_cameras:
        requested = "all" if cameras is None else ",".join(cameras)
        saved = metadata.get("cameras", [])
        with_images = metadata.get("script_args", {}).get("with_images")
        raise RuntimeError(
            f"No saved camera images found in {episode_dir}. "
            f"Requested camera={requested}; metadata.cameras={saved}; with_images={with_images}. "
            "Regenerate the episode with --with-images and --cameras overview_camera,wrist_cam,wrist_cam1."
        )

    results = []
    for camera in selected_cameras:
        camera_out = out / camera if multi_camera and (multi_episode or not out.suffix) else out
        results.append(
            export_one_episode(
                episode_dir,
                camera,
                camera_out,
                export_format,
                stride,
                fps,
                index if multi_episode else None,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export saved episode camera images to PNG frames or MP4.")
    parser.add_argument("path", help="Episode directory, or a batch directory containing multiple episodes.")
    parser.add_argument("--camera", default="all", type=parse_camera_selection, help="Camera name, comma-separated camera names, or 'all' to export every saved camera.")
    parser.add_argument("--out", default=None, help="Output directory/file. Defaults under episode/exports or batch/exports.")
    parser.add_argument("--format", choices=("png", "mp4", "both"), default="mp4", help="Export format.")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth frame.")
    parser.add_argument("--fps", type=float, default=20.0, help="MP4 frames per second.")
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    path = Path(args.path)
    episodes = episode_dirs(path)
    if not episodes:
        raise RuntimeError(f"No episodes found under {path}")

    is_batch = len(episodes) > 1 or not ((path / "metadata.json").exists())
    default_root = path / "exports" if is_batch else path / "exports"
    out = Path(args.out) if args.out else default_root
    if is_batch:
        out.mkdir(parents=True, exist_ok=True)
    multi_camera = args.camera is None or len(args.camera) > 1
    if multi_camera and out.suffix:
        raise ValueError("--out must be a directory when exporting multiple cameras")

    print(f"input: {path}")
    print(f"episodes_found: {len(episodes)}")
    print(f"camera: {'all' if args.camera is None else ','.join(args.camera)}")

    try:
        for index, episode_dir in enumerate(episodes):
            results = export_episode_cameras(
                episode_dir,
                args.camera,
                out,
                args.format,
                args.stride,
                args.fps,
                index if is_batch else None,
                is_batch,
                multi_camera,
            )
            for result in results:
                print(result)
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
