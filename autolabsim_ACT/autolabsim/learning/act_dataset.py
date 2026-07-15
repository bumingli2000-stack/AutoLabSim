from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeInfo:
    episode_dir: Path
    npz_path: Path
    length: int


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Sequence[float]]) -> "NormalizationStats":
        return cls(
            state_mean=np.asarray(data["state_mean"], dtype=np.float32),
            state_std=np.asarray(data["state_std"], dtype=np.float32),
            action_mean=np.asarray(data["action_mean"], dtype=np.float32),
            action_std=np.asarray(data["action_std"], dtype=np.float32),
        )


def discover_episodes(data_root: str | Path) -> list[EpisodeInfo]:
    root = Path(data_root).expanduser().resolve()
    paths = sorted(root.rglob("episode.npz"))
    episodes: list[EpisodeInfo] = []
    for npz_path in paths:
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "action" not in data.files:
                    continue
                length = int(data["action"].shape[0])
        except (OSError, ValueError):
            continue
        if length > 1:
            episodes.append(EpisodeInfo(npz_path.parent, npz_path, length))
    if not episodes:
        raise FileNotFoundError(f"No valid episode.npz files found below: {root}")
    return episodes


def split_episodes(
    episodes: Sequence[EpisodeInfo], val_ratio: float, seed: int
) -> tuple[list[EpisodeInfo], list[EpisodeInfo]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    indices = np.arange(len(episodes))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_ratio))
    if len(indices) > 1 and val_ratio > 0.0:
        val_count = max(1, min(len(indices) - 1, val_count))
    val_ids = set(int(i) for i in indices[:val_count])
    train = [ep for i, ep in enumerate(episodes) if i not in val_ids]
    val = [ep for i, ep in enumerate(episodes) if i in val_ids]
    return train, val


def _stream_moments(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    for array in arrays:
        x = np.asarray(array, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {x.shape}")
        count += int(x.shape[0])
        part_sum = x.sum(axis=0)
        part_sq = np.square(x).sum(axis=0)
        total = part_sum if total is None else total + part_sum
        total_sq = part_sq if total_sq is None else total_sq + part_sq
    if count == 0 or total is None or total_sq is None:
        raise ValueError("Cannot compute statistics from an empty dataset")
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def compute_normalization_stats(
    episodes: Sequence[EpisodeInfo], state_key: str
) -> NormalizationStats:
    def state_arrays() -> Iterable[np.ndarray]:
        for episode in episodes:
            with np.load(episode.npz_path, allow_pickle=False) as data:
                if state_key not in data.files:
                    raise KeyError(f"{episode.npz_path} has no key {state_key!r}")
                yield np.asarray(data[state_key])

    def action_arrays() -> Iterable[np.ndarray]:
        for episode in episodes:
            with np.load(episode.npz_path, allow_pickle=False) as data:
                yield np.asarray(data["action"])

    state_mean, state_std = _stream_moments(state_arrays())
    action_mean, action_std = _stream_moments(action_arrays())
    # Near-constant channels should not be amplified by normalization.
    state_std = np.maximum(state_std, 1e-3)
    action_std = np.maximum(action_std, 1e-3)
    return NormalizationStats(state_mean, state_std, action_mean, action_std)


class AutoLabACTDataset(Dataset[dict[str, torch.Tensor]]):
    """Frame-indexed ACT dataset over AutoLabSim episode.npz files.

    Recorder semantics in AutoLabSim are post-step: observation t is recorded after
    action t has already been applied. Therefore ``action_offset=1`` is the safe
    default; sample t predicts future controls beginning at action t+1.
    """

    def __init__(
        self,
        episodes: Sequence[EpisodeInfo],
        *,
        camera_names: Sequence[str],
        state_key: str,
        chunk_size: int,
        action_offset: int,
        sample_stride: int,
        stats: NormalizationStats,
        image_size: tuple[int, int] | None = (224, 224),
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if action_offset < 0:
            raise ValueError("action_offset must be non-negative")
        if sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        self.episodes = list(episodes)
        self.camera_names = tuple(camera_names)
        self.state_key = state_key
        self.chunk_size = int(chunk_size)
        self.action_offset = int(action_offset)
        self.sample_stride = int(sample_stride)
        self.stats = stats
        self.image_size = image_size
        self.samples: list[tuple[int, int]] = []

        for episode_index, episode in enumerate(self.episodes):
            with np.load(episode.npz_path, allow_pickle=False) as data:
                required = {state_key, "action", *(f"image_{name}" for name in self.camera_names)}
                missing = sorted(required.difference(data.files))
                if missing:
                    raise KeyError(f"{episode.npz_path} is missing keys: {missing}")
                lengths = [int(data[state_key].shape[0]), int(data["action"].shape[0])]
                lengths.extend(int(data[f"image_{name}"].shape[0]) for name in self.camera_names)
                length = min(lengths)
            valid_observations = max(0, length - self.action_offset)
            self.samples.extend(
                (episode_index, t)
                for t in range(0, valid_observations, self.sample_stride)
            )
        if not self.samples:
            raise ValueError("No trainable frames remain after action_offset/sample_stride")

        if self.stats.state_mean.shape[0] != self.state_dim:
            raise ValueError("State statistics dimension does not match the dataset")
        if self.stats.action_mean.shape[0] != self.action_dim:
            raise ValueError("Action statistics dimension does not match the dataset")

    @property
    def state_dim(self) -> int:
        with np.load(self.episodes[0].npz_path, allow_pickle=False) as data:
            return int(data[self.state_key].shape[-1])

    @property
    def action_dim(self) -> int:
        with np.load(self.episodes[0].npz_path, allow_pickle=False) as data:
            return int(data["action"].shape[-1])

    def __len__(self) -> int:
        return len(self.samples)

    @lru_cache(maxsize=6)
    def _load_episode(self, path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            keys = {self.state_key, "action", *(f"image_{name}" for name in self.camera_names)}
            return {key: np.asarray(data[key]) for key in keys}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, t = self.samples[index]
        episode = self.episodes[episode_index]
        arrays = self._load_episode(str(episode.npz_path))

        state = np.asarray(arrays[self.state_key][t], dtype=np.float32)
        state = (state - self.stats.state_mean) / self.stats.state_std

        images: list[torch.Tensor] = []
        for camera in self.camera_names:
            image = np.asarray(arrays[f"image_{camera}"][t])
            if image.ndim != 3 or image.shape[-1] not in (3, 4):
                raise ValueError(f"Expected HWC RGB/RGBA image, got {image.shape}")
            image = image[..., :3]
            tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
            if self.image_size is not None and tuple(tensor.shape[-2:]) != self.image_size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            images.append(tensor)

        action_array = np.asarray(arrays["action"], dtype=np.float32)
        start = t + self.action_offset
        end = min(start + self.chunk_size, action_array.shape[0])
        valid = max(0, end - start)
        chunk = np.empty((self.chunk_size, action_array.shape[-1]), dtype=np.float32)
        pad_mask = np.ones(self.chunk_size, dtype=np.bool_)
        if valid > 0:
            chunk[:valid] = action_array[start:end]
            chunk[valid:] = action_array[end - 1]
            pad_mask[:valid] = False
        else:
            chunk[:] = action_array[-1]
        chunk = (chunk - self.stats.action_mean) / self.stats.action_std

        return {
            "state": torch.from_numpy(state),
            "images": torch.stack(images, dim=0),
            "actions": torch.from_numpy(chunk),
            "pad_mask": torch.from_numpy(pad_mask),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }
