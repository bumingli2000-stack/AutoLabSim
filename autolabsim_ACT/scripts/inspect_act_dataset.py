from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from autolabsim_ACT.autolabsim.learning.act_dataset import discover_episodes


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AutoLabSim episodes before ACT training.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    episodes = discover_episodes(args.data_root)
    print(f"episodes: {len(episodes)}")
    total_frames = 0
    common_keys: set[str] | None = None
    ctrl_action_same: list[float] = []
    ctrl_next_action: list[float] = []

    for i, episode in enumerate(episodes[: max(1, args.limit)]):
        with np.load(episode.npz_path, allow_pickle=False) as data:
            keys = set(data.files)
            common_keys = keys if common_keys is None else common_keys.intersection(keys)
            total_frames += int(data["action"].shape[0])
            print(f"\n[{i}] {episode.episode_dir}")
            for key in sorted(data.files):
                value = data[key]
                print(f"  {key:24s} shape={str(value.shape):18s} dtype={value.dtype}")
            if "ctrl" in data.files and data["ctrl"].shape == data["action"].shape:
                ctrl = np.asarray(data["ctrl"], dtype=np.float64)
                action = np.asarray(data["action"], dtype=np.float64)
                ctrl_action_same.append(float(np.mean(np.abs(ctrl - action))))
                if len(action) > 1:
                    ctrl_next_action.append(float(np.mean(np.abs(ctrl[:-1] - action[1:]))))

    all_frames = sum(ep.length for ep in episodes)
    print("\nsummary")
    print(f"  all_frames: {all_frames}")
    print(f"  inspected_frames: {total_frames}")
    print(f"  common_keys: {sorted(common_keys or [])}")
    if ctrl_action_same:
        print(f"  mean |ctrl[t]-action[t]|: {np.mean(ctrl_action_same):.8f}")
    if ctrl_next_action:
        print(f"  mean |ctrl[t]-action[t+1]|: {np.mean(ctrl_next_action):.8f}")
    image_keys = sorted(key for key in (common_keys or set()) if key.startswith("image_"))
    print(f"  cameras: {[key.removeprefix('image_') for key in image_keys]}")
    print("\nRecommended ACT label alignment: observation[t] -> action[t+1:t+1+chunk].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
