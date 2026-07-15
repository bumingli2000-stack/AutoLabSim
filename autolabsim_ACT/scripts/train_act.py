from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from autolabsim_ACT.autolabsim.learning.act_dataset import (
    AutoLabACTDataset,
    compute_normalization_stats,
    discover_episodes,
    split_episodes,
)
from autolabsim_ACT.autolabsim.learning.act_model import ACTConfig, ACTPolicy, act_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT directly on AutoLabSim episode.npz data.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cameras", default="overview_camera,wrist_cam,wrist_cam1")
    parser.add_argument("--state-key", choices=("ctrl", "qpos", "qvel", "state"), default="ctrl")
    parser.add_argument("--action-offset", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    *,
    model: ACTPolicy,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val: float,
    metadata: dict[str, Any],
) -> None:
    payload = {
        **metadata,
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def run_epoch(
    model: ACTPolicy,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
    kl_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
    count = 0

    for batch in loader:
        state = batch["state"].to(device, non_blocking=True)
        images = batch["images"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
            if amp_enabled
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast_context:
            output = model(state, images, actions, pad_mask)
            losses = act_loss(output, actions, pad_mask, kl_weight)

        if training:
            if scaler is not None:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = int(state.shape[0])
        count += batch_size
        for key in sums:
            sums[key] += float(losses[key].detach()) * batch_size

    return {key: value / max(1, count) for key, value in sums.items()}


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    camera_names = tuple(name.strip() for name in args.cameras.split(",") if name.strip())
    if not camera_names:
        raise ValueError("At least one camera is required")

    episodes = discover_episodes(args.data_root)
    train_episodes, val_episodes = split_episodes(episodes, args.val_ratio, args.seed)
    stats = compute_normalization_stats(train_episodes, args.state_key)
    image_size = (args.image_height, args.image_width)

    train_dataset = AutoLabACTDataset(
        train_episodes,
        camera_names=camera_names,
        state_key=args.state_key,
        chunk_size=args.chunk_size,
        action_offset=args.action_offset,
        sample_stride=args.sample_stride,
        stats=stats,
        image_size=image_size,
    )
    val_dataset = None
    if val_episodes:
        val_dataset = AutoLabACTDataset(
            val_episodes,
            camera_names=camera_names,
            state_key=args.state_key,
            chunk_size=args.chunk_size,
            action_offset=args.action_offset,
            sample_stride=args.sample_stride,
            stats=stats,
            image_size=image_size,
        )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    val_loader = (
        DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)
        if val_dataset is not None
        else None
    )

    model_config = ACTConfig(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        num_cameras=len(camera_names),
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        nheads=args.nheads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = ACTPolicy(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    start_epoch = 0
    best_val = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", math.inf))

    metadata = {
        "format": "autolabsim_act_checkpoint_v1",
        "model_config": model_config.to_dict(),
        "normalization": stats.to_dict(),
        "camera_names": list(camera_names),
        "state_key": args.state_key,
        "action_offset": args.action_offset,
        "image_size": list(image_size),
        "data_root": str(args.data_root),
        "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_episode_dirs": [str(ep.episode_dir) for ep in train_episodes],
        "val_episode_dirs": [str(ep.episode_dir) for ep in val_episodes],
    }
    with (args.out_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    metrics_path = args.out_dir / "metrics.jsonl"
    print(f"episodes train/val: {len(train_episodes)}/{len(val_episodes)}")
    print(f"samples train/val: {len(train_dataset)}/{len(val_dataset) if val_dataset else 0}")
    print(f"state_dim={model_config.state_dim} action_dim={model_config.action_dim}")
    print(f"device={device} amp={amp_enabled}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            kl_weight=args.kl_weight,
            grad_clip=args.grad_clip,
        )
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                kl_weight=args.kl_weight,
                grad_clip=args.grad_clip,
            )
            selection_value = val_metrics["loss"]
        else:
            val_metrics = {}
            selection_value = train_metrics["loss"]
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch:03d} "
            f"train={train_metrics['loss']:.5f} "
            f"val={val_metrics.get('loss', float('nan')):.5f} "
            f"l1={train_metrics['l1']:.5f} kl={train_metrics['kl']:.5f}"
        )

        save_checkpoint(
            args.out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val=min(best_val, selection_value),
            metadata=metadata,
        )
        if selection_value < best_val:
            best_val = selection_value
            save_checkpoint(
                args.out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val=best_val,
                metadata=metadata,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
