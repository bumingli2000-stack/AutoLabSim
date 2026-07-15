'''
样本读写
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_episode(
    episode_dir: str | Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    path = Path(episode_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.savez_compressed(path / "episode.npz", **arrays)


def load_episode(episode_dir: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(episode_dir)

    with (path / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    with np.load(path / "episode.npz") as data:
        arrays = {key: data[key] for key in data.files}

    return metadata, arrays
