from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torchvision.models import resnet18


@dataclass(frozen=True)
class ACTConfig:
    state_dim: int
    action_dim: int
    num_cameras: int
    chunk_size: int = 50
    hidden_dim: int = 256
    latent_dim: int = 32
    nheads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SinePositionEmbedding2D(nn.Module):
    def __init__(self, hidden_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for 2-D sine positions")
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        quarter = self.hidden_dim // 4
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        dim = torch.arange(quarter, device=device, dtype=dtype)
        scale = self.temperature ** (dim / max(1, quarter - 1))
        y = y[:, None] / scale[None, :]
        x = x[:, None] / scale[None, :]
        y_embed = torch.cat((y.sin(), y.cos()), dim=-1)[:, None, :].expand(height, width, -1)
        x_embed = torch.cat((x.sin(), x.cos()), dim=-1)[None, :, :].expand(height, width, -1)
        pos = torch.cat((y_embed, x_embed), dim=-1)
        return pos.reshape(height * width, self.hidden_dim)


class VisualBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        network = resnet18(weights=None)
        self.body = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(images))


class ACTPolicy(nn.Module):
    """Compact CVAE Action Chunking Transformer for AutoLabSim."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        if h % config.nheads != 0:
            raise ValueError("hidden_dim must be divisible by nheads")

        self.backbone = VisualBackbone(h)
        self.position_2d = SinePositionEmbedding2D(h)
        self.camera_embed = nn.Embedding(config.num_cameras, h)

        self.state_proj = nn.Linear(config.state_dim, h)
        self.action_proj = nn.Linear(config.action_dim, h)
        self.action_encoder_pos = nn.Embedding(config.chunk_size + 2, h)
        self.encoder_cls = nn.Parameter(torch.zeros(1, 1, h))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        self.to_mu = nn.Linear(h, config.latent_dim)
        self.to_logvar = nn.Linear(h, config.latent_dim)
        self.latent_proj = nn.Linear(config.latent_dim, h)

        self.state_memory_pos = nn.Parameter(torch.zeros(1, 1, h))
        self.latent_memory_pos = nn.Parameter(torch.zeros(1, 1, h))
        self.action_queries = nn.Embedding(config.chunk_size, h)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=h,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.action_head = nn.Linear(h, config.action_dim)

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.encoder_cls, std=0.02)
        nn.init.normal_(self.state_memory_pos, std=0.02)
        nn.init.normal_(self.latent_memory_pos, std=0.02)

    def _encode_latent(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        pad_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        cls = self.encoder_cls.expand(batch, -1, -1)
        state_token = self.state_proj(state).unsqueeze(1)
        action_tokens = self.action_proj(actions)
        sequence = torch.cat((cls, state_token, action_tokens), dim=1)
        positions = self.action_encoder_pos.weight[: sequence.shape[1]].unsqueeze(0)
        sequence = sequence + positions

        key_padding_mask = None
        if pad_mask is not None:
            prefix = torch.zeros((batch, 2), dtype=torch.bool, device=pad_mask.device)
            key_padding_mask = torch.cat((prefix, pad_mask.bool()), dim=1)
        encoded = self.latent_encoder(sequence, src_key_padding_mask=key_padding_mask)
        cls_output = encoded[:, 0]
        mu = self.to_mu(cls_output)
        logvar = self.to_logvar(cls_output).clamp(-10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def _visual_memory(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, K, 3, H, W]
        batch, cameras, channels, height, width = images.shape
        if cameras != self.config.num_cameras or channels != 3:
            raise ValueError(
                f"Expected images [B,{self.config.num_cameras},3,H,W], got {tuple(images.shape)}"
            )
        normalized = (images - self.image_mean) / self.image_std
        flat = normalized.reshape(batch * cameras, channels, height, width)
        features = self.backbone(flat)
        _, hidden, feat_h, feat_w = features.shape
        features = features.flatten(2).transpose(1, 2)
        spatial_pos = self.position_2d(feat_h, feat_w, features.device, features.dtype)
        spatial_pos = spatial_pos.unsqueeze(0).expand(batch * cameras, -1, -1)
        camera_ids = torch.arange(cameras, device=features.device).repeat_interleave(batch)
        # reshape order is B,K; generate matching camera ids explicitly.
        camera_ids = torch.arange(cameras, device=features.device).unsqueeze(0).expand(batch, -1).reshape(-1)
        camera_pos = self.camera_embed(camera_ids).unsqueeze(1)
        features = features + spatial_pos + camera_pos
        return features.reshape(batch, cameras * feat_h * feat_w, hidden)

    def forward(
        self,
        state: torch.Tensor,
        images: torch.Tensor,
        actions: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = state.shape[0]
        if actions is None:
            z = torch.zeros((batch, self.config.latent_dim), device=state.device, dtype=state.dtype)
            mu = z
            logvar = z
        else:
            z, mu, logvar = self._encode_latent(state, actions, pad_mask)

        state_token = self.state_proj(state).unsqueeze(1) + self.state_memory_pos
        latent_token = self.latent_proj(z).unsqueeze(1) + self.latent_memory_pos
        visual_tokens = self._visual_memory(images)
        memory = torch.cat((state_token, latent_token, visual_tokens), dim=1)

        queries = self.action_queries.weight.unsqueeze(0).expand(batch, -1, -1)
        target = torch.zeros_like(queries) + queries
        decoded = self.decoder(target, memory)
        predicted_actions = self.action_head(decoded)
        return {"actions": predicted_actions, "mu": mu, "logvar": logvar}


def act_loss(
    output: dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    pad_mask: torch.Tensor,
    kl_weight: float,
) -> dict[str, torch.Tensor]:
    valid = (~pad_mask.bool()).unsqueeze(-1)
    absolute_error = (output["actions"] - target_actions).abs()
    denominator = valid.sum().clamp_min(1) * target_actions.shape[-1]
    l1 = (absolute_error * valid).sum() / denominator
    mu = output["mu"]
    logvar = output["logvar"]
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
    total = l1 + float(kl_weight) * kl
    return {"loss": total, "l1": l1, "kl": kl}
