"""The ViT metasmoothness "menu" -- the transformer analogue of
`metasmooth.Routine`.  Shared by the Phase-A diagnostic runner
(`_run_vit_metasmooth_local.py`) and the Phase-B accuracy runner
(`_run_vit_smooth_train_local.py`) so the routine definitions stay in one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from model import ViTConfig


@dataclass(frozen=True)
class ViTGeometry:
    """Fixed ViT-tiny geometry (everything a routine does *not* vary)."""

    image_size: int = 32
    patch: int = 8
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 2.0


@dataclass(frozen=True)
class ViTRoutine:
    """A point in the ViT metasmoothness design menu.

    pool          "mean" (the paper's smoothness-friendly average pool) or "cls".
    pre_norm      True = standard pre-norm ViT and the corrected smooth choice;
                  False = post-norm, which the completed sweep found less smooth.
    smooth_act    GELU (True) vs ReLU (False).
    final_scale   divide logits by this (the ResNet-9 study's dominant lever).
    width_mult    encoder_dim multiplier (wider == more metasmooth, per the paper).
    depth         encoder_depth override (None -> the geometry default).
    """

    name: str
    pool: str = "mean"
    pre_norm: bool = True
    smooth_act: bool = True
    final_scale: float = 1.0
    width_mult: float = 1.0
    depth: int | None = None

    def build_config(self, num_classes: int, geom: ViTGeometry) -> ViTConfig:
        # encoder_dim must be divisible by both encoder_heads and 4.
        step = math.lcm(geom.heads, 4)
        raw = max(step, round(geom.dim * self.width_mult))
        dim = max(step, (raw // step) * step)
        return ViTConfig(
            image_size=geom.image_size,
            patch_size=geom.patch,
            encoder_dim=dim,
            encoder_depth=self.depth if self.depth is not None else geom.depth,
            encoder_heads=geom.heads,
            mlp_ratio=geom.mlp_ratio,
            num_classes=num_classes,
            pre_norm=self.pre_norm,
            pool=self.pool,
            smooth_activation=self.smooth_act,
            final_logit_scale=self.final_scale,
        )


# baseline = a standard ViT: mean pool, pre-norm, GELU, unscaled logits.
BASELINE_VIT = ViTRoutine(name="baseline")
# Corrected smooth routine from the completed Phase-A sweep: mean pooling,
# pre-norm, GELU, and logits/10. Post-norm increased curvature and is omitted.
SMOOTH_VIT = ViTRoutine(name="smooth", pre_norm=True, final_scale=10.0)
# smooth + the width lever.
SMOOTH_WIDE_VIT = ViTRoutine(
    name="smooth_wide", pre_norm=True, final_scale=10.0, width_mult=2.0
)
