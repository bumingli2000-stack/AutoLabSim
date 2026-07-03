from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from ..scene_profile import resolve_scene_spec, scene_names, scene_rooted_path
from . import TaskRequest, create_task, summarize_metadata, task_names
from .common import parse_cameras, parse_seeds


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a batch of scripted AutoLabSim episodes.")
    parser.add_argument(
        "--scene",
        choices=scene_names(),
        default="fast_tubes",
        help="Named scene profile that supplies default model, reset config, cameras, and naming.",
    )
    parser.add_argument(
        "--task",
        choices=task_names(),
        default="tube_grasp",
        help="Task sequence to generate.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of episodes to generate when --seeds is not set.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed when generating a contiguous seed range.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, for example: 0,1,2,3,4.")
    parser.add_argument("--out-root", default="data/episodes/tube_grasp_batch", help="Directory containing generated episodes and manifest.json.")
    parser.add_argument("--model", default=None, help="MuJoCo XML model path. Overrides the selected scene profile.")
    parser.add_argument("--reset-config", default=None, help="Reset config JSON. Overrides the selected scene profile.")
    parser.add_argument("--active-joint", default="centrifuge_50ml_screw_joint_1", help="Fallback tube free joint if reset info is absent.")
    parser.add_argument("--with-images", action="store_true", help="Save camera images. Default skips images for speed.")
    parser.add_argument("--cameras", default=None, help="Comma-separated cameras saved when --with-images is set. Overrides the selected scene profile.")
    parser.add_argument("--control-dt", type=float, default=0.05, help="Control timestep in seconds.")
    parser.add_argument("--gl-backend", default=None, help="MuJoCo GL backend for image rendering, for example egl or glfw.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override frame_skip.")
    parser.add_argument("--dry-run", action="store_true", help="Print episode plan without generating episodes.")
    return parser


def run_episode(index: int, seed: int, episode_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene_spec = resolve_scene_spec(
        args.scene,
        model=args.model,
        reset_config=args.reset_config,
        cameras=parse_cameras(args.cameras) if args.cameras else None,
    )
    model_path, reset_path = scene_rooted_path(scene_spec, PROJECT_ROOT)
    request = TaskRequest(
        task=args.task,
        seed=seed,
        episode_index=index,
        out_dir=episode_dir,
        model=str(model_path),
        reset_config=str(reset_path) if reset_path is not None else None,
        cameras=scene_spec.cameras,
        with_images=bool(args.with_images),
        control_dt=args.control_dt,
        frame_skip=args.frame_skip,
        gl_backend=args.gl_backend,
        params=vars(args).copy(),
    )
    task = create_task(request)
    try:
        metadata = task.run()
    finally:
        task.finish()
    return summarize_metadata(args.task, metadata)


def _dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    scene_spec = resolve_scene_spec(
        args.scene,
        model=args.model,
        reset_config=args.reset_config,
        cameras=parse_cameras(args.cameras) if args.cameras else None,
    )
    return {
        "scene": scene_spec.name,
        "model": scene_spec.model_path,
        "reset_config": scene_spec.reset_config,
        "task": args.task,
        "with_images": args.with_images,
        "cameras": list(scene_spec.cameras) if args.with_images else [],
    }


def run_batch(args: argparse.Namespace) -> int:
    seeds = parse_seeds(args.seeds, args.count, args.seed_start)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "autolabsim_batch_manifest_v1",
        "out_root": str(out_root),
        "scene": args.scene,
        "task": args.task,
        "seeds": seeds,
        "episodes": [],
    }

    print(f"batch_out_root: {out_root}")
    print(f"episode_count: {len(seeds)}")

    for index, seed in enumerate(seeds):
        episode_dir = out_root / f"episode_{index:03d}_seed_{seed:04d}"
        print(f"[{index + 1}/{len(seeds)}] seed={seed} out={episode_dir}")

        record: dict[str, Any] = {
            "index": index,
            "seed": seed,
            "episode_dir": str(episode_dir),
            "status": "pending",
        }

        if args.dry_run:
            record["status"] = "dry_run"
            record["summary"] = _dry_run_summary(args)
            manifest["episodes"].append(record)
            print("  dry_run: would create task, run scripted trajectory, and save episode")
            continue

        try:
            summary = run_episode(index, seed, episode_dir, args)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            manifest["episodes"].append(record)
            print(f"episode_failed: seed={seed} error={exc}", file=sys.stderr)
            break

        record["status"] = "ok"
        record["summary"] = summary
        manifest["episodes"].append(record)

        extra = ""
        if "screw_released" in summary:
            extra = f" released={summary['screw_released']} twist={summary['twist_angle']:.3f}"
        print(f"  ok: steps={summary['steps']} slot={summary['slot_name']} ik={summary['ik_all_waypoints_solved']}{extra}")

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    ok_count = sum(1 for item in manifest["episodes"] if item.get("status") == "ok")
    print(f"manifest: {manifest_path}")
    print(f"completed_episodes: {ok_count}/{len(seeds)}")
    return 0 if ok_count == len([seed for seed in seeds if not args.dry_run]) or args.dry_run else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_batch_parser()
    args = parser.parse_args(argv)
    return run_batch(args)
