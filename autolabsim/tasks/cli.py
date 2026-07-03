from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ..scene_profile import resolve_scene_spec, scene_names, scene_rooted_path
from . import TaskRequest, create_task, summarize_metadata, task_names
from .common import ARM_DEFAULTS, parse_cameras, parse_optional_vec3, parse_seeds


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
    parser.add_argument("--arm", choices=sorted(ARM_DEFAULTS), default="second", help="[tube_grasp] Robot arm to execute with.")
    parser.add_argument("--tube-arm", choices=sorted(ARM_DEFAULTS), default="second", help="[tube_then_cap_grasp] Arm used to grasp/lift the tube.")
    parser.add_argument("--cap-arm", choices=sorted(ARM_DEFAULTS), default="first", help="[tube_then_cap_grasp] Arm used to grasp the cap after tube lift.")
    parser.add_argument("--approach-axis", default="1 0 0", help="Reserved tube grasp approach axis override.")
    parser.add_argument("--closing-axis", default="0 1 0", help="Reserved tube grasp closing axis override.")
    parser.add_argument("--tube-approach-axis", default=None, help="[tube_then_cap_grasp] Reserved tube-stage approach axis override.")
    parser.add_argument("--tube-closing-axis", default=None, help="[tube_then_cap_grasp] Reserved tube-stage finger closing axis override.")
    parser.add_argument("--tube-tool-roll", type=float, default=0, help="[tube_then_cap_grasp] Override tube-stage gripper roll around local approach axis.")
    parser.add_argument("--with-images", action="store_true", help="Save camera images. Default skips images for speed.")
    parser.add_argument("--cameras", default=None, help="Comma-separated cameras saved when --with-images is set. Overrides the selected scene profile.")
    parser.add_argument("--position-only", action="store_true", help="Reserved IK option for future use.")
    parser.add_argument("--allow-partial-plan", action="store_true", help="Reserved planning option for future use.")
    parser.add_argument("--grasp-height", type=float, default=0.09, help="Height above tube free-joint origin for the pinch center.")
    parser.add_argument("--pregrasp-distance", type=float, default=0.10, help="Distance before the grasp along the approach axis.")
    parser.add_argument("--lift-distance", type=float, default=0.10, help="Reserved lift distance fallback.")
    parser.add_argument("--lift-offset", type=lambda value: parse_optional_vec3(value, "lift_offset"), default=[0.25, 0, 0.12], help="World XYZ offset from grasp to lift waypoint.")
    parser.add_argument("--pinch-forward-offset", type=float, default=0.02, help="Move the pinch target forward along the approach axis.")
    parser.add_argument("--grasp-outward-offset", type=float, default=0.02, help="Move grasp waypoints outward, opposite the approach axis.")
    parser.add_argument("--tool-roll", type=float, default=float(np.pi), help="Roll the gripper around its local approach axis in radians.")
    parser.add_argument("--cap-body", default=None, help="[tube_then_cap_grasp] Reserved cap body override.")
    parser.add_argument("--cap-approach-axis", default="0 0 -1", help="[tube_then_cap_grasp] Reserved cap approach axis override.")
    parser.add_argument("--cap-closing-axis", default="1 0 0", help="[tube_then_cap_grasp] Reserved cap closing axis override.")
    parser.add_argument("--cap-tool-roll", type=float, default=0.0, help="[tube_then_cap_grasp] Roll for the cap gripper around local approach axis.")
    parser.add_argument("--cap-offset", type=lambda value: parse_optional_vec3(value, "cap_offset"), default=[0.0, 0.0, 0.02], help="[tube_then_cap_grasp] World XYZ offset from cap body center to grasp point.")
    parser.add_argument("--cap-grasp-outward-offset", type=float, default=0.0, help="[tube_then_cap_grasp] Reserved outward grasp offset.")
    parser.add_argument("--cap-pregrasp-distance", type=float, default=0.10, help="[tube_then_cap_grasp] Distance before cap grasp along the approach axis.")
    parser.add_argument("--cap-post-distance", type=float, default=0.08, help="[tube_then_cap_grasp] Reserved cap post distance fallback.")
    parser.add_argument("--cap-post-offset", type=lambda value: parse_optional_vec3(value, "cap_post_offset"), default=[0.0, 0.0, 0.08], help="[tube_then_cap_grasp] World XYZ offset from cap grasp to cap post point.")
    parser.add_argument("--cap-clearance-lift", type=float, default=0.06, help="[tube_then_cap_grasp] Lift cap upward after unscrewing before moving to the place position.")
    parser.add_argument("--cap-hold-steps", type=int, default=20, help="[tube_then_cap_grasp] Final hold steps after cap post point.")
    parser.add_argument("--cap-target-mode", choices=("planned", "current"), default="current", help="[tube_then_cap_grasp] Reserved cap target mode.")
    parser.add_argument("--cap-attach-distance-threshold", type=float, default=0.025, help="[tube_then_cap_grasp] Reserved cap attach threshold.")
    parser.add_argument("--cap-scripted-follow-steps", type=int, default=16, help="[tube_then_cap_grasp] Reserved cap follow duration.")
    parser.add_argument("--ratchet-angle", type=float, default=float(np.pi / 2.0), help="[tube_then_cap_grasp] Twist angle per grip before opening and rewinding the wrist.")
    parser.add_argument("--thread-pitch", type=float, default=0.008, help="[tube_then_cap_grasp] Cap lift per full turn while unscrewing.")
    parser.add_argument("--release-lift", type=float, default=0.008, help="[tube_then_cap_grasp] Maximum scripted cap lift while unscrewing.")
    parser.add_argument("--open-gripper", type=float, default=0.0, help="Open gripper command.")
    parser.add_argument("--close-gripper", type=float, default=255.0, help="Close gripper command.")
    parser.add_argument("--control-dt", type=float, default=0.05, help="Control timestep in seconds.")
    parser.add_argument("--gl-backend", default=None, help="MuJoCo GL backend for image rendering, for example egl or glfw.")
    parser.add_argument("--frame-skip", type=int, default=None, help="Override frame_skip.")
    parser.add_argument("--settle-steps", type=int, default=20, help="Unrecorded settle steps before planning.")
    parser.add_argument(
        "--no-hold-active-tube-until-grasp",
        dest="hold_active_tube_until_grasp",
        action="store_false",
        help="Disable scripted holding of the active tube before the gripper closes.",
    )
    parser.add_argument(
        "--no-scripted-hold-tube-after-lift",
        dest="scripted_hold_tube_after_lift",
        action="store_false",
        help="[tube_then_cap_grasp] Reserved tube hold toggle.",
    )
    parser.add_argument(
        "--scripted-reposition-cap-for-planning",
        action="store_true",
        help="[tube_then_cap_grasp] Reserved cap planning reposition toggle.",
    )
    parser.add_argument(
        "--no-scripted-hold-cap-after-grasp",
        dest="scripted_hold_cap_after_grasp",
        action="store_false",
        help="[tube_then_cap_grasp] Reserved cap follow toggle.",
    )
    parser.add_argument("--steps-per-segment", type=int, default=30, help="Motion segment steps.")
    parser.add_argument("--grasp-hold-steps", type=int, default=8, help="Open hold steps at grasp pose.")
    parser.add_argument("--close-steps", type=int, default=12, help="Close gripper steps.")
    parser.add_argument("--hold-steps", type=int, default=20, help="Final hold steps.")
    parser.add_argument("--ik-max-iters", type=int, default=500, help="IK maximum iterations per waypoint.")
    parser.add_argument("--ik-pos-tol", type=float, default=0.003, help="IK position tolerance.")
    parser.add_argument("--ik-rot-tol", type=float, default=0.05, help="IK rotation tolerance.")
    parser.add_argument("--ik-damping", type=float, default=0.08, help="IK damping.")
    parser.add_argument("--dry-run", action="store_true", help="Print episode plan without generating episodes.")
    parser.set_defaults(hold_active_tube_until_grasp=True)
    parser.set_defaults(scripted_hold_tube_after_lift=True)
    parser.set_defaults(scripted_reposition_cap_for_planning=False)
    parser.set_defaults(scripted_hold_cap_after_grasp=False)
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
        with_images=args.with_images,
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
        "arm": args.arm,
        "tube_arm": args.tube_arm,
        "cap_arm": args.cap_arm if args.task == "tube_then_cap_grasp" else None,
        "tool_roll": args.tool_roll,
        "tube_tool_roll": args.tube_tool_roll,
        "cap_tool_roll": args.cap_tool_roll if args.task == "tube_then_cap_grasp" else None,
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
