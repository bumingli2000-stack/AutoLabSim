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
        default="tube_then_cap_grasp",
        help="Task sequence to generate.",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of episodes to generate when --seeds is not set.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed when generating a contiguous seed range.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, for example: 0,1,2,3,4.")
    parser.add_argument(
        "--out-root",
        default="data/episodes/tube_then_cap_grasp_batch",
        help="Directory containing generated episodes and manifest.json.",
    )
    parser.add_argument("--model", default=None, help="MuJoCo XML model path. Overrides the selected scene profile.")
    parser.add_argument("--reset-config", default=None, help="Reset config JSON. Overrides the selected scene profile.")
    parser.add_argument("--active-joint", default="centrifuge_50ml_screw_joint_1", help="Fallback tube free joint if reset info is absent.")
    parser.add_argument("--with-images", action="store_true", help="Save camera images. Default skips images for speed.")
    parser.add_argument("--cameras", default=None, help="Comma-separated cameras saved when --with-images is set. Overrides the selected scene profile.")
    parser.add_argument("--control-dt", type=float, default=0.05, help="Control timestep in seconds.")
    parser.add_argument("--gl-backend", default=None, help="MuJoCo GL backend for image rendering, for example egl or glfw.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override frame_skip.")
    parser.add_argument("--dry-run", action="store_true", help="Print episode plan without generating episodes.")
    parser.add_argument("--arm", default=None, help="Task arm selector for supported tasks, for example first or second.")
    parser.add_argument("--open-gripper", type=float, default=None, help="Open gripper control value for supported tasks.")
    parser.add_argument("--close-gripper", type=float, default=None, help="Closed gripper control value for supported tasks.")
    parser.add_argument("--initial-static-steps", type=int, default=None, help="ADP initial static frames recorded before any simulation step.")
    parser.add_argument("--settle-steps", type=int, default=None, help="Initial settle steps for supported tasks.")
    parser.add_argument("--tool-stabilize-steps", type=int, default=None, help="ADP hold steps after capturing the held pipette before moving to the tip.")
    parser.add_argument("--free-settle-steps", type=int, default=None, help="Unrecorded free-object settle steps before supported tasks capture object state.")
    parser.add_argument("--steps-per-segment", type=int, default=None, help="Motion interpolation steps for supported tasks.")
    parser.add_argument("--close-steps", type=int, default=None, help="Gripper close/open interpolation steps for supported tasks.")
    parser.add_argument("--hold-steps", type=int, default=None, help="Final hold steps for supported tasks.")
    parser.add_argument("--release-wait-steps", type=int, default=None, help="Tip drop wait steps for ADP tip-release tasks.")
    parser.add_argument("--tip-mount-settle-steps", type=int, default=None, help="Hold steps after ADP tip insertion before attaching the tip.")
    parser.add_argument("--grasp-hold-steps", type=int, default=None, help="Hold steps before closing gripper for supported tasks.")
    parser.add_argument("--pregrasp-distance", type=float, default=None, help="Pregrasp offset distance for supported tasks.")
    parser.add_argument("--handle-grasp-offset", default=None, help="Pipette-body-local handle grasp offset as 'x y z'.")
    parser.add_argument("--handle-grasp-euler", default=None, help="Pipette-body-local grasp frame XYZ Euler angles in radians.")
    parser.add_argument("--grasp-to-gripper-offset", default=None, help="Grasp-frame-local gripper target offset as 'x y z'.")
    parser.add_argument("--grasp-to-gripper-euler", default=None, help="Grasp-frame-local gripper XYZ Euler angles in radians.")
    parser.add_argument("--lift-offset", default=None, help="Object lift offset as 'x y z' for supported tasks.")
    parser.add_argument("--middle-grasp-arm", default=None, help="Arm used to grasp the pipette middle during handoff.")
    parser.add_argument("--middle-pregrasp-distance", type=float, default=None, help="Pregrasp offset distance for the pipette middle grasp.")
    parser.add_argument("--middle-grasp-offset", default=None, help="Pipette-body-local middle grasp offset as 'x y z'.")
    parser.add_argument("--middle-grasp-euler", default=None, help="Pipette-body-local middle grasp frame XYZ Euler angles in radians.")
    parser.add_argument("--middle-grasp-to-gripper-offset", default=None, help="Middle-grasp-frame-local gripper target offset as 'x y z'.")
    parser.add_argument("--middle-grasp-to-gripper-euler", default=None, help="Middle-grasp-frame-local gripper XYZ Euler angles in radians.")
    parser.add_argument("--first-retreat-after-handoff-offset", default=None, help="World offset for the first gripper after second-arm handoff as 'x y z'.")
    parser.add_argument("--pipette-joint", default=None, help="Pipette free joint name for pipette_grasp.")
    parser.add_argument("--pipette-body", default=None, help="Pipette body name used as grasp reference for pipette_grasp.")
    parser.add_argument("--pipette-tip-site", default=None, help="Pipette tip site name used for tip-hover targeting.")
    parser.add_argument("--tip-joint-prefix", default=None, help="Free-joint name prefix used to find visible pipette tips.")
    parser.add_argument("--tip-site-prefix", default=None, help="Attached tip site prefix, for example tip for tip01mount_site.")
    parser.add_argument("--tip-mount-site-suffix", default=None, help="Attached tip mount site suffix.")
    parser.add_argument("--tip-end-site-suffix", default=None, help="Attached tip end site suffix.")
    parser.add_argument("--tip-pose-servo", dest="tip_pose_servo_enabled", action="store_true", default=None, help="Enable pipette tip pose servo against the selected tip mount site orientation.")
    parser.add_argument("--no-tip-pose-servo", dest="tip_pose_servo_enabled", action="store_false", help="Disable pipette tip pose servo.")
    parser.add_argument("--tip-hover-height", type=float, default=None, help="Target height above the nearest visible tip, in meters.")
    parser.add_argument("--tip-retract-height", type=float, default=None, help="Height to lift vertically after mounting an ADP tip before moving to the tube.")
    parser.add_argument("--tip-hover-steps", type=int, default=None, help="Motion steps for moving the ADP to the selected tip hover pose.")
    parser.add_argument("--tip-length", type=float, default=None, help="Tip body length used by supported pipette/ADP tasks.")
    parser.add_argument("--tip-mount-offset", default=None, help="Mounted-tip target offset in the selected tip mount-site frame as 'x y z'.")
    parser.add_argument("--tip-mount-target-euler", default=None, help="Mounted-tip target XYZ Euler angles in the selected tip mount-site frame.")
    parser.add_argument("--parking-weld", default=None, help="Optional parking weld equality name released after grasp.")
    parser.add_argument("--vertical-quat", default=None, help="Target lifted pipette quaternion as 'w x y z'.")
    parser.add_argument("--tube-joint", default=None, help="Fallback centrifuge tube free joint for pipette_grasp.")
    parser.add_argument("--tube-top-offset", type=float, default=None, help="Tube top z offset from the tube free-joint origin.")
    parser.add_argument("--tube-hover-height", type=float, default=None, help="Height above tube top for the pipette tip hover target.")
    parser.add_argument("--tube-near-height", type=float, default=None, help="Height above tube top for the final near-tube pipette tip target.")
    parser.add_argument("--tube-target-offset", default=None, help="Tube-local/world target offset added to the active tube position as 'x y z'.")
    parser.add_argument("--ik-max-iters", type=int, default=None, help="IK iteration limit for supported tasks.")
    parser.add_argument("--ik-pos-tol", type=float, default=None, help="IK position tolerance for supported tasks.")
    parser.add_argument("--ik-rot-tol", type=float, default=None, help="IK rotation tolerance for supported tasks.")
    parser.add_argument("--ik-damping", type=float, default=None, help="IK damping for supported tasks.")
    parser.add_argument("--waypoint-settle-steps", type=int, default=None, help="Waypoint settle steps for supported tasks.")
    parser.add_argument("--waypoint-settle-pos-tol", type=float, default=None, help="Waypoint settle position tolerance.")
    parser.add_argument("--visual-servo", dest="visual_servo_enabled", action="store_true", default=None, help="Enable closed-loop visual/site servo refinement after each debugged waypoint.")
    parser.add_argument("--no-visual-servo", dest="visual_servo_enabled", action="store_false", help="Disable closed-loop visual/site servo refinement.")
    parser.add_argument("--visual-servo-max-iters", type=int, default=None, help="Maximum correction iterations for visual/site servo refinement.")
    parser.add_argument("--visual-servo-steps", type=int, default=None, help="Control steps used for each visual/site servo correction.")
    parser.add_argument("--visual-servo-pos-tol", type=float, default=None, help="Position tolerance for visual/site servo refinement.")
    parser.add_argument("--visual-servo-rot-tol", type=float, default=None, help="Rotation tolerance in radians for visual/site servo refinement.")
    parser.add_argument("--visual-servo-gain", type=float, default=None, help="World-space correction gain for visual/site servo refinement.")
    parser.add_argument("--visual-servo-integral-gain", type=float, default=None, help="Integral correction gain for visual/site servo refinement.")
    parser.add_argument("--visual-servo-max-correction", type=float, default=None, help="Maximum world-space correction per visual/site servo iteration.")
    return parser


def run_episode(index: int, seed: int, episode_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene_spec = resolve_scene_spec(
        args.scene,
        model=args.model,
        reset_config=args.reset_config,
        cameras=parse_cameras(args.cameras) if args.cameras else None,
    )
    model_path, reset_path = scene_rooted_path(scene_spec, PROJECT_ROOT)
    # 创建通用的 TaskRequest 对象，包含创建任务所需的所有参数
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
        # 在任务完成后，关闭 MuJoCo renderer 和环境资源。
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
    # 创建输出目录
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
        slot = summary.get("slot_name")
        slot_text = f" slot={slot}" if slot is not None else ""
        print(f"  ok: steps={summary['steps']}{slot_text} ik={summary['ik_all_waypoints_solved']}{extra}")

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    ok_count = sum(1 for item in manifest["episodes"] if item.get("status") == "ok")
    print(f"manifest: {manifest_path}")
    print(f"completed_episodes: {ok_count}/{len(seeds)}")
    return 0 if ok_count == len([seed for seed in seeds if not args.dry_run]) or args.dry_run else 1


def main(argv: list[str] | None = None) -> int:
    # 解析命令行参数
    parser = build_batch_parser()
    args = parser.parse_args(argv)
    # 批量 episode 调度器
    return run_batch(args)