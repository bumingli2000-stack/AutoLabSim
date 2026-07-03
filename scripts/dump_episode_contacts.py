from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabsim.episode_io import load_episode


DEFAULT_FILTERS = ("2f85", "pad", "cap", "tube", "centrifuge")


def geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
    return name or f"geom_{int(geom_id)}"


def body_name(mujoco: Any, model: Any, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[int(geom_id)])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name or f"body_{body_id}"


def pair_key(
    mujoco: Any,
    model: Any,
    geom1: int,
    geom2: int,
) -> tuple[str, str]:
    lhs = f"{body_name(mujoco, model, geom1)}/{geom_name(mujoco, model, geom1)}"
    rhs = f"{body_name(mujoco, model, geom2)}/{geom_name(mujoco, model, geom2)}"
    return tuple(sorted((lhs, rhs)))


def matches_filter(pair: tuple[str, str], filters: tuple[str, ...]) -> bool:
    if not filters:
        return True
    text = " ".join(pair).lower()
    return any(item.lower() in text for item in filters)


def parse_filters(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() in ("", "none", "all"):
        return ()
    return tuple(item for item in raw.replace(",", " ").split() if item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an episode and summarize MuJoCo contact pairs by phase.")
    parser.add_argument("episode_dir", help="Episode directory containing metadata.json and episode.npz.")
    parser.add_argument("--filters", default=",".join(DEFAULT_FILTERS), help="Comma/space separated substrings. Use 'all' to disable filtering.")
    parser.add_argument("--top", type=int, default=40, help="Maximum number of contact rows to print.")
    parser.add_argument("--phase-contains", default=None, help="Only include phases containing this substring.")
    args = parser.parse_args()

    import mujoco

    metadata, arrays = load_episode(args.episode_dir)
    model_path = metadata.get("model_path")
    if not model_path:
        raise ValueError("metadata.json does not contain model_path")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    filters = parse_filters(args.filters)

    qpos = arrays["qpos"]
    qvel = arrays["qvel"]
    phases = arrays["phase"].astype(str)
    contacts: dict[tuple[str, tuple[str, str]], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "min_dist": 0.0, "first_frame": None, "last_frame": None}
    )

    for frame, phase in enumerate(phases):
        if args.phase_contains and args.phase_contains not in phase:
            continue
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = pair_key(mujoco, model, int(contact.geom1), int(contact.geom2))
            if not matches_filter(pair, filters):
                continue
            key = (phase, pair)
            item = contacts[key]
            item["count"] += 1
            item["min_dist"] = min(float(item["min_dist"]), float(contact.dist))
            item["first_frame"] = frame if item["first_frame"] is None else item["first_frame"]
            item["last_frame"] = frame

    rows = sorted(
        contacts.items(),
        key=lambda kv: (kv[1]["min_dist"], -kv[1]["count"]),
    )
    print(f"episode: {args.episode_dir}")
    print(f"filters: {filters or 'all'}")
    print(f"contact_rows: {len(rows)}")
    for (phase, pair), item in rows[: max(0, args.top)]:
        penetration_mm = -1000.0 * float(item["min_dist"])
        print(
            f"phase={phase} count={item['count']} "
            f"penetration_mm={penetration_mm:.3f} "
            f"frames={item['first_frame']}..{item['last_frame']} "
            f"pair={pair[0]} <-> {pair[1]}"
        )


if __name__ == "__main__":
    main()
