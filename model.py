from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 32
    patch_size: int = 8
    channels: int = 3
    encoder_dim: int = 64
    encoder_depth: int = 2
    encoder_heads: int = 4
    mlp_ratio: float = 4.0
    num_classes: int = 10
    # Metasmoothness "menu" levers (the ViT analogue of metasmooth.Routine).
    # Defaults reproduce the original behavior exactly (mean pool, pre-norm,
    # GELU, unscaled logits), so existing callers/tests are unchanged.
    pre_norm: bool = True
    pool: Literal["mean", "cls"] = "mean"
    smooth_activation: bool = True
    final_logit_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.encoder_depth < 0:
            raise ValueError("encoder_depth cannot be negative")
        if self.encoder_dim % self.encoder_heads != 0:
            raise ValueError("encoder_dim must be divisible by encoder_heads")
        if self.encoder_dim % 4 != 0:
            raise ValueError("encoder_dim must be divisible by 4")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if self.pool not in ("mean", "cls"):
            raise ValueError("pool must be 'mean' or 'cls'")
        if self.final_logit_scale <= 0:
            raise ValueError("final_logit_scale must be positive")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size**2

    @property
    def patch_dim(self) -> int:
        return self.channels * self.patch_size**2


def patchify(images: Tensor, patch_size: int) -> Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    batch, channels, height, width = images.shape
    if height != width:
        raise ValueError("only square images are supported")
    if height % patch_size != 0:
        raise ValueError("image dimensions must be divisible by patch_size")

    grid = height // patch_size
    return (
        images.reshape(batch, channels, grid, patch_size, grid, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, grid * grid, patch_size * patch_size * channels)
    )


def _sincos_1d(positions: Tensor, dim: int, dtype: torch.dtype) -> Tensor:
    frequencies = torch.arange(0, dim, 2, device=positions.device, dtype=dtype)
    frequencies = torch.exp(-math.log(10_000.0) * frequencies / dim)
    angles = positions.to(dtype=dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


def _sincos_2d(
    grid_size: int, dim: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    coordinates = torch.arange(grid_size, device=device)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.cat(
        (
            _sincos_1d(rows.flatten(), dim // 2, dtype),
            _sincos_1d(columns.flatten(), dim // 2, dtype),
        ),
        dim=1,
    )


class MathSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, dim = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, sequence, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
        attended = attended.transpose(1, 2).reshape(batch, sequence, dim)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        pre_norm: bool = True,
        smooth_activation: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        activation: nn.Module = nn.GELU() if smooth_activation else nn.ReLU()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MathSelfAttention(dim, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, dim),
        )
        self.pre_norm = pre_norm

    def forward(self, inputs: Tensor) -> Tensor:
        if self.pre_norm:
            inputs = inputs + self.attention(self.attention_norm(inputs))
            return inputs + self.mlp(self.mlp_norm(inputs))

        # Post-norm: normalize the residual sum.  The worktree's ViT
        # metasmoothness probes favored this for cleaner metagradients.
        inputs = self.attention_norm(inputs + self.attention(inputs))
        return self.mlp_norm(inputs + self.mlp(inputs))


class VisionTransformerClassifier(nn.Module):
    def __init__(self, config: ViTConfig | None = None) -> None:
        super().__init__()
        self.config = config or ViTConfig()

        self.patch_embedding = nn.Linear(self.config.patch_dim, self.config.encoder_dim)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, self.config.encoder_dim))
            if self.config.pool == "cls"
            else None
        )
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(
                self.config.encoder_dim,
                self.config.encoder_heads,
                self.config.mlp_ratio,
                pre_norm=self.config.pre_norm,
                smooth_activation=self.config.smooth_activation,
            )
            for _ in range(self.config.encoder_depth)
        )
        self.encoder_norm = nn.LayerNorm(self.config.encoder_dim)
        self.head = nn.Linear(self.config.encoder_dim, self.config.num_classes)

    def forward(self, images: Tensor) -> Tensor:
        self._validate_inputs(images)
        patches = patchify(images, self.config.patch_size)

        positions = _sincos_2d(
            self.config.grid_size,
            self.config.encoder_dim,
            device=images.device,
            dtype=images.dtype,
        )
        encoded = self.patch_embedding(patches) + positions.unsqueeze(0)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(images.shape[0], -1, -1).to(encoded.dtype)
            encoded = torch.cat((cls_token, encoded), dim=1)
        for block in self.encoder_blocks:
            encoded = block(encoded)
        encoded = self.encoder_norm(encoded)
        # Pool over tokens: mean (the paper's smoothness-friendly average pooling)
        # or the CLS token (the standard ViT default), then project to logits and
        # optionally scale them down (the paper's dominant smoothness lever).
        if self.config.pool == "mean":
            pooled = encoded.mean(dim=1)
        else:
            pooled = encoded[:, 0]
        return self.head(pooled) / self.config.final_logit_scale

    def _validate_inputs(self, images: Tensor) -> None:
        expected_image_shape = (
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        )
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_image_shape:
            raise ValueError(
                f"images must have shape [batch, {expected_image_shape[0]}, "
                f"{expected_image_shape[1]}, {expected_image_shape[2]}]"
            )


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return per_example_cross_entropy_loss(logits, labels).mean()


def per_example_cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [batch] matching the logits batch")
    if labels.dtype != torch.long:
        raise ValueError("labels must have torch.long dtype")
    if logits.device != labels.device:
        raise ValueError("logits and labels must share a device")
    if labels.numel() and (
        torch.any(labels < 0) or torch.any(labels >= logits.shape[1])
    ):
        raise ValueError("labels contains an invalid class index")

    return F.cross_entropy(logits, labels, reduction="none")
