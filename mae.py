from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class MaeConfig:
    image_size: int = 32
    patch_size: int = 8
    channels: int = 3
    encoder_dim: int = 64
    encoder_depth: int = 2
    encoder_heads: int = 4
    decoder_dim: int = 32
    decoder_depth: int = 1
    decoder_heads: int = 4
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.encoder_depth < 0 or self.decoder_depth < 0:
            raise ValueError("encoder_depth and decoder_depth cannot be negative")
        if self.encoder_dim % self.encoder_heads != 0:
            raise ValueError("encoder_dim must be divisible by encoder_heads")
        if self.decoder_dim % self.decoder_heads != 0:
            raise ValueError("decoder_dim must be divisible by decoder_heads")
        if self.encoder_dim % 4 != 0 or self.decoder_dim % 4 != 0:
            raise ValueError("encoder_dim and decoder_dim must be divisible by 4")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")

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


def unpatchify(
    patches: Tensor, patch_size: int, image_size: int, channels: int = 3
) -> Tensor:
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, patches, patch_pixels]")
    if patch_size <= 0 or image_size <= 0 or channels <= 0:
        raise ValueError("patch_size, image_size, and channels must be positive")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    batch, num_patches, patch_dim = patches.shape
    grid = image_size // patch_size
    if num_patches != grid**2:
        raise ValueError(f"expected {grid**2} patches, got {num_patches}")
    if patch_dim != channels * patch_size**2:
        raise ValueError(
            f"expected patch dimension {channels * patch_size**2}, got {patch_dim}"
        )

    return (
        patches.reshape(batch, grid, grid, patch_size, patch_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, image_size, image_size)
    )


def _mask_seed(image_id: int, step: int, seed: int) -> int:
    payload = struct.pack("<qqq", image_id, step, seed)
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def make_patch_mask(
    image_ids: Tensor | Sequence[int],
    step: int,
    seed: int,
    num_patches: int,
    mask_ratio: float,
) -> Tensor:
    """Return a deterministic boolean mask where True denotes a hidden patch."""
    if isinstance(image_ids, Tensor):
        if image_ids.ndim != 1:
            raise ValueError("image_ids must be one-dimensional")
        output_device = image_ids.device
        ids = [int(value) for value in image_ids.detach().cpu().tolist()]
    else:
        output_device = torch.device("cpu")
        ids = [int(value) for value in image_ids]

    if num_patches <= 0:
        raise ValueError("num_patches must be positive")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")

    num_masked = int(num_patches * mask_ratio)
    rows: list[Tensor] = []
    for image_id in ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_mask_seed(image_id, int(step), int(seed)))
        hidden = torch.randperm(num_patches, generator=generator)[:num_masked]
        row = torch.zeros(num_patches, dtype=torch.bool)
        row[hidden] = True
        rows.append(row)

    mask = torch.stack(rows) if rows else torch.empty((0, num_patches), dtype=torch.bool)
    return mask.to(output_device)


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
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = MathSelfAttention(dim, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, config: MaeConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaeConfig()

        self.patch_embedding = nn.Linear(self.config.patch_dim, self.config.encoder_dim)
        self.encoder_blocks = nn.ModuleList(
            TransformerBlock(
                self.config.encoder_dim,
                self.config.encoder_heads,
                self.config.mlp_ratio,
            )
            for _ in range(self.config.encoder_depth)
        )
        self.encoder_norm = nn.LayerNorm(self.config.encoder_dim)

        self.decoder_embedding = nn.Linear(
            self.config.encoder_dim, self.config.decoder_dim
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            TransformerBlock(
                self.config.decoder_dim,
                self.config.decoder_heads,
                self.config.mlp_ratio,
            )
            for _ in range(self.config.decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(self.config.decoder_dim)
        self.reconstruction_head = nn.Linear(
            self.config.decoder_dim, self.config.patch_dim
        )

    def forward(self, images: Tensor, patch_mask: Tensor) -> Tensor:
        self._validate_inputs(images, patch_mask)
        patches = patchify(images, self.config.patch_size)
        batch = images.shape[0]

        encoder_positions = _sincos_2d(
            self.config.grid_size,
            self.config.encoder_dim,
            device=images.device,
            dtype=images.dtype,
        )
        embedded = self.patch_embedding(patches) + encoder_positions.unsqueeze(0)

        visible_counts = (~patch_mask).sum(dim=1)
        if not torch.all(visible_counts == visible_counts[0]):
            raise ValueError("each image must have the same number of visible patches")
        num_visible = int(visible_counts[0])
        visible_indices = torch.where(~patch_mask)[1].reshape(batch, num_visible)
        gather_indices = visible_indices.unsqueeze(-1).expand(
            -1, -1, self.config.encoder_dim
        )
        encoded = embedded.gather(dim=1, index=gather_indices)
        for block in self.encoder_blocks:
            encoded = block(encoded)
        encoded = self.encoder_norm(encoded)

        decoded_visible = self.decoder_embedding(encoded)
        decoder_tokens = self.mask_token.expand(
            batch, self.config.num_patches, self.config.decoder_dim
        )
        decoder_indices = visible_indices.unsqueeze(-1).expand(
            -1, -1, self.config.decoder_dim
        )
        decoder_tokens = decoder_tokens.scatter(
            dim=1, index=decoder_indices, src=decoded_visible
        )
        decoder_positions = _sincos_2d(
            self.config.grid_size,
            self.config.decoder_dim,
            device=images.device,
            dtype=images.dtype,
        )
        decoded = decoder_tokens + decoder_positions.unsqueeze(0)
        for block in self.decoder_blocks:
            decoded = block(decoded)
        return self.reconstruction_head(self.decoder_norm(decoded))

    def _validate_inputs(self, images: Tensor, patch_mask: Tensor) -> None:
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
        if patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must have boolean dtype")
        if patch_mask.device != images.device:
            raise ValueError("patch_mask and images must be on the same device")
        if patch_mask.shape != (images.shape[0], self.config.num_patches):
            raise ValueError(
                f"patch_mask must have shape [{images.shape[0]}, "
                f"{self.config.num_patches}]"
            )


def masked_reconstruction_loss(
    images: Tensor, predictions: Tensor, patch_mask: Tensor
) -> Tensor:
    return per_example_masked_reconstruction_loss(
        images, predictions, patch_mask
    ).mean()


def per_example_masked_reconstruction_loss(
    images: Tensor, predictions: Tensor, patch_mask: Tensor
) -> Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [batch, patches, patch_pixels]")
    if patch_mask.dtype != torch.bool or patch_mask.shape != predictions.shape[:2]:
        raise ValueError("patch_mask must be boolean and match the first prediction axes")
    if patch_mask.device != images.device or predictions.device != images.device:
        raise ValueError("images, predictions, and patch_mask must share a device")

    channels = images.shape[1]
    patch_area, remainder = divmod(predictions.shape[-1], channels)
    patch_size = math.isqrt(patch_area)
    if remainder or patch_size**2 != patch_area:
        raise ValueError("prediction dimension does not describe square image patches")

    targets = patchify(images, patch_size)
    if targets.shape != predictions.shape:
        raise ValueError("prediction shape does not match patchified images")
    masked_counts = patch_mask.sum(dim=1)
    if torch.any(masked_counts == 0):
        raise ValueError("each example must contain at least one masked patch")

    per_patch_loss = (predictions - targets).square().mean(dim=-1)
    return (per_patch_loss * patch_mask).sum(dim=1) / masked_counts
