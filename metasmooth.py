"""Finite-difference *metasmoothness* diagnostics for a weighted-data
ResNet-9 / CIFAR-10 learning algorithm.

This implements the two metasmoothness metrics of Engstrom et al. (2025),
"Optimizing ML Training with Metagradient Descent" (arXiv:2503.13751), so we can
measure how amenable a training routine is to metagradient-based optimization,
and then modify the routine (per the paper's "menu" of design choices) to make
it as metasmooth as possible.

Definitions (Section 3.1).  Let ``f = phi . A`` be the *training function*: the
learning algorithm ``A`` maps a metaparameter ``z`` to trained parameters
``theta``, and the output function ``phi`` maps ``theta`` to a scalar (here the
held-out classification loss).  Fix a step ``h > 0`` and a direction ``v`` and
let ``Delta_f(z; v) = (f(z + h v) - f(z)) / h``.

  * Definition 1 / Eq. (5) -- curvature metasmoothness of ``f``::

        S_{h,v}(f; z) = | (Delta_f(z + h v) - Delta_f(z)) / h |
                      = | f(z + 2 h v) - 2 f(z + h v) + f(z) | / h**2 ,

    a second-order finite difference (directional curvature).  *Smaller is
    smoother* -- a beta-smooth ``f`` has ``S <= beta``.

  * Definition 2 / Eq. (6) -- empirical metasmoothness of ``A`` in *parameter*
    space.  With ``theta_0 = A(z)``, ``theta_h = A(z + h v)``,
    ``theta_2h = A(z + 2 h v)``, ``Delta_A(z;v) = (theta_h - theta_0)/h``,
    ``Delta_A(z+hv;v) = (theta_2h - theta_h)/h`` and ``d = |theta_2h - theta_0|``::

        S_hat_{h,v}(A; z) = sign(Delta_A(z;v))^T diag(d / ||d||_1) sign(Delta_A(z+hv;v)) ,

    a per-coordinate, range-weighted average sign agreement between consecutive
    finite-difference metagradients.  ``S_hat in [-1, 1]``; *larger (-> 1) is
    smoother*.

Both metrics need only three runs of the (deterministic) learning algorithm at
``z``, ``z + h v`` and ``z + 2 h v`` -- no metagradients are required.

The metaparameter ``z`` is a vector of continuous *data weights* evaluated at
``z = 0`` (the paper's count relaxation, Sec. 4.1.2): training example ``i`` is
weighted by ``exp(z_i)`` (per-example) or ``exp(z_{c(i)})`` (per-cluster, with
``c(i)`` the cluster / class of ``i``).  At ``z = 0`` every weight is 1, which
recovers ordinary unweighted training.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def configure_determinism(seed: int, *, tf32: bool = False, strict: bool = False) -> None:
    """Make training reproducible so that the *only* thing varying between the
    three algorithm runs of a metasmoothness probe is the metaparameter ``z``.

    With ``strict=False`` we rely on fixed seeds, deterministic cuDNN
    convolutions and (in this module) atomics-free pooling, which is enough for
    bit-reproducible runs on a fixed machine without the
    ``CUBLAS_WORKSPACE_CONFIG`` constraint that ``strict=True`` imposes.

    ``tf32`` defaults to False: the finite-difference curvature in Definition 1
    is a tiny second difference, so we keep full fp32 matmul/conv precision.
    """
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if strict:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Training routine "menu" (the paper's smoothness design choices)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Routine:
    """A point in the paper's menu of metasmoothness design choices (Sec. 3.2,
    Remark 4, Fig. 4).

    pool
        ``"max"`` or ``"avg"``.  The paper swaps max pooling for average pooling
        to improve smoothness.
    bn_before_act
        ``True`` puts BatchNorm *before* the activation (the paper's smooth
        choice); ``False`` is the non-smooth "BN after activation" placement.
    final_scale
        Divide the network's output logits by this factor.  The paper "scales
        down the last layer's output by a factor of 10" to improve smoothness.
    width
        Channel-width multiplier.  Wider networks are more metasmooth.
    smooth_act
        Replace the ``ReLU`` (a non-smooth max) with a smooth ``GELU``.  Off by
        default to stay close to the paper's listed menu; a clearly relevant and
        cheap extra lever.
    norm
        Normalization layer: ``"bn"`` (BatchNorm, the default / paper choice) or
        ``"gn"`` (GroupNorm).  GroupNorm has no running-stat buffers, which makes
        exact metagradients through training trivial (the functional engine holds
        buffers fixed) -- useful for the curation model in Phase C.  ``bn_before_act``
        is reinterpreted as "norm before activation".
    gn_groups
        Number of GroupNorm groups (used only when ``norm == "gn"``).
    name
        Human-readable label used in reports.
    """

    pool: Literal["max", "avg"] = "max"
    bn_before_act: bool = True
    final_scale: float = 1.0
    width: float = 1.0
    smooth_act: bool = False
    norm: Literal["bn", "gn"] = "bn"
    gn_groups: int = 8
    name: str = "baseline"

    def __post_init__(self) -> None:
        if self.pool not in ("max", "avg"):
            raise ValueError("pool must be 'max' or 'avg'")
        if self.final_scale <= 0:
            raise ValueError("final_scale must be positive")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.norm not in ("bn", "gn"):
            raise ValueError("norm must be 'bn' or 'gn'")
        if self.gn_groups <= 0:
            raise ValueError("gn_groups must be positive")


# Mirrors the notebook's ResNet-9 exactly: max pooling, BN-before-ReLU,
# unscaled logits, base width.
BASELINE_ROUTINE = Routine(name="baseline")

# Paper menu at *equal capacity* (width unchanged): average pooling, BN before
# activation, and a 10x-smaller final-layer output.  Width is a separate lever
# (see ``SMOOTH_WIDE_ROUTINE``) so the headline comparison is capacity-matched.
SMOOTH_ROUTINE = Routine(
    pool="avg",
    bn_before_act=True,
    final_scale=10.0,
    width=1.0,
    smooth_act=False,
    name="smooth",
)

# The full menu including the paper's width lever (wider == more metasmooth).
SMOOTH_WIDE_ROUTINE = Routine(
    pool="avg",
    bn_before_act=True,
    final_scale=10.0,
    width=2.0,
    smooth_act=False,
    name="smooth_wide",
)

# The smooth routine with GroupNorm instead of BatchNorm.  GroupNorm has no
# running-stat buffers, so exact metagradients through training are clean (the
# functional engine holds buffers fixed) -- this is the curation model for Phase
# C.  Re-check its metasmoothness with the Phase-A benchmark before trusting it.
SMOOTH_GN_ROUTINE = Routine(
    pool="avg",
    bn_before_act=True,
    final_scale=10.0,
    width=1.0,
    smooth_act=False,
    norm="gn",
    name="smooth_gn",
)


def _pool2x2(x: Tensor, mode: str) -> Tensor:
    """Atomics-free (hence deterministic) 2x2 stride-2 spatial pooling."""
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("spatial dims must be even to 2x2 pool")
    blocks = x.reshape(b, c, h // 2, 2, w // 2, 2)
    return blocks.amax(dim=(3, 5)) if mode == "max" else blocks.mean(dim=(3, 5))


def _global_pool(x: Tensor, mode: str) -> Tensor:
    return x.amax(dim=(2, 3)) if mode == "max" else x.mean(dim=(2, 3))


def _make_norm(out_ch: int, norm: str, gn_groups: int) -> nn.Module:
    if norm == "gn":
        groups = gn_groups if out_ch % gn_groups == 0 else 1
        return nn.GroupNorm(groups, out_ch)
    return nn.BatchNorm2d(out_ch)


class ConvBlock(nn.Module):
    """Conv -> (norm/act in the configured order), with optional 2x2 pool."""

    def __init__(
        self, in_ch: int, out_ch: int, *, pool: str, bn_before_act: bool,
        smooth_act: bool, norm: str = "bn", gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm = _make_norm(out_ch, norm, gn_groups)
        self.act = nn.GELU() if smooth_act else nn.ReLU(inplace=False)
        self.bn_before_act = bn_before_act
        self.pool = pool
        self.do_pool = False

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn_before_act:
            x = self.act(self.norm(x))
        else:
            x = self.norm(self.act(x))
        if self.do_pool:
            x = _pool2x2(x, self.pool)
        return x


class ResNet9(nn.Module):
    """The notebook's ResNet-9, parameterised by a :class:`Routine`."""

    def __init__(self, routine: Routine, num_classes: int = 10, in_ch: int = 3) -> None:
        super().__init__()
        self.routine = routine
        w = routine.width

        def width(channels: int) -> int:
            return max(1, round(channels * w))

        def block(cin: int, cout: int, pool: bool) -> ConvBlock:
            b = ConvBlock(
                cin, cout, pool=routine.pool,
                bn_before_act=routine.bn_before_act, smooth_act=routine.smooth_act,
                norm=routine.norm, gn_groups=routine.gn_groups,
            )
            b.do_pool = pool
            return b

        c64, c128, c256, c512 = width(64), width(128), width(256), width(512)
        self.conv1 = block(in_ch, c64, pool=False)
        self.conv2 = block(c64, c128, pool=True)
        self.res1 = nn.Sequential(
            block(c128, c128, pool=False), block(c128, c128, pool=False)
        )
        self.conv3 = block(c128, c256, pool=True)
        self.conv4 = block(c256, c512, pool=True)
        self.res2 = nn.Sequential(
            block(c512, c512, pool=False), block(c512, c512, pool=False)
        )
        self.head = nn.Linear(c512, num_classes)
        self.final_scale = routine.final_scale
        self.pool_mode = routine.pool

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.res2(x)
        x = _global_pool(x, self.pool_mode)
        return self.head(x) / self.final_scale


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class CifarSubset:
    """A fixed, in-memory CIFAR-10 subset (no random augmentation)."""

    train_images: Tensor  # [n_train, 3, 32, 32]
    train_labels: Tensor  # [n_train]
    val_images: Tensor  # [n_val, 3, 32, 32]
    val_labels: Tensor  # [n_val]

    @property
    def n_train(self) -> int:
        return self.train_images.shape[0]

    @property
    def n_val(self) -> int:
        return self.val_images.shape[0]


_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar_subset(
    data_dir: str,
    *,
    n_train: int,
    n_val: int,
    seed: int,
    device: torch.device,
) -> CifarSubset:
    """Load a deterministic CIFAR-10 subset, normalized, onto ``device``.

    Train and val examples are disjoint draws from the official training split
    (so labels are available for both); no random crop/flip is applied, since
    metasmoothness requires the algorithm's only varying input to be ``z``.

    If ``{data_dir}/cifar10_train.npz`` exists (keys ``images`` uint8
    ``[N, 32, 32, 3]`` and ``labels`` ``[N]``) it is used directly; this lets
    environments where the torchvision mirror is throttled prebuild the cache
    (see ``_build_cifar_cache.py``).  Otherwise torchvision downloads CIFAR-10.
    """
    cache = os.path.join(data_dir, "cifar10_train.npz")
    if os.path.exists(cache):
        blob = np.load(cache)
        data_np, targets = blob["images"], blob["labels"].tolist()
    else:
        from torchvision import datasets

        raw = datasets.CIFAR10(root=data_dir, train=True, download=True)
        data_np, targets = raw.data, raw.targets

    images = torch.from_numpy(data_np).float().div_(255.0)  # [N, 32, 32, 3]
    images = images.permute(0, 3, 1, 2).contiguous()  # [N, 3, 32, 32]
    mean = torch.tensor(_CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    labels = torch.tensor(targets, dtype=torch.long)

    if n_train + n_val > images.shape[0]:
        raise ValueError("n_train + n_val exceeds the CIFAR-10 training split")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(images.shape[0], generator=generator)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    return CifarSubset(
        train_images=images[train_idx].to(device),
        train_labels=labels[train_idx].to(device),
        val_images=images[val_idx].to(device),
        val_labels=labels[val_idx].to(device),
    )


# --------------------------------------------------------------------------- #
# Metaparameter (continuous data weights at z = 0)
# --------------------------------------------------------------------------- #
@dataclass
class Metaparam:
    """A continuous data-weight metaparameterization evaluated at ``z = 0``.

    kind
        ``"per_example"`` -- one coordinate per training example.
        ``"per_cluster"`` -- one coordinate per cluster (here, per CIFAR class).
    dim
        Dimensionality of ``z``.
    coord_of_example
        ``[n_train]`` long tensor mapping each example to its ``z`` coordinate
        (identity for per-example; the class label for per-cluster).
    """

    kind: Literal["per_example", "per_cluster"]
    dim: int
    coord_of_example: Tensor

    def example_log_weights(self, z: Tensor) -> Tensor:
        """``z`` -> per-example log-weights (gathered to one entry per example)."""
        return z[self.coord_of_example]


def make_metaparam(
    kind: Literal["per_example", "per_cluster"],
    subset: CifarSubset,
    *,
    num_clusters: int | None = None,
) -> Metaparam:
    n = subset.n_train
    if kind == "per_example":
        coord = torch.arange(n, device=subset.train_labels.device)
        return Metaparam("per_example", n, coord)
    if kind == "per_cluster":
        # Clusters default to CIFAR-10 classes (a natural, label-aligned grouping).
        coord = subset.train_labels.clone()
        dim = num_clusters or int(subset.train_labels.max().item()) + 1
        return Metaparam("per_cluster", dim, coord)
    raise ValueError(f"unknown metaparam kind: {kind}")


# --------------------------------------------------------------------------- #
# The learning algorithm A(z)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 500
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    warmup_frac: float = 0.3
    recalibrate_bn: bool = True
    # Precision.  Default fp32 ("off") to resolve the tiny Definition-1 second
    # difference; "fp16"/"bf16" autocast trade precision for ~2x speed.
    amp: Literal["off", "fp16", "bf16"] = "off"
    tf32: bool = False
    init_seed: int = 0
    order_seed: int = 1

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[self.amp]


def _lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to zero."""
    warmup = int(total_steps * cfg.warmup_frac)
    if step < warmup:
        return cfg.lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * progress))


def _recalibrate_bn(
    model: nn.Module, images: Tensor, batch_size: int,
    autocast_dtype: torch.dtype | None = None,
) -> None:
    """Recompute BatchNorm running statistics over the (final-model) training
    set so ``model.eval()`` is meaningful after short training.  Without this,
    too-few BN updates leave running stats far from the activation statistics and
    eval-mode logits explode -- an artifact that would masquerade as
    non-smoothness."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        return
    saved = {m: m.momentum for m in bns}
    model.train()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average -> exact population stats
    use_amp = autocast_dtype is not None
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
        for i in range(0, images.shape[0], batch_size):
            model(images[i : i + batch_size])
    for m in bns:
        m.momentum = saved[m]


@dataclass
class RunResult:
    theta: Tensor  # flattened trained parameters (CPU float32)
    val_loss: float  # held-out cross-entropy (the output function phi)
    train_loss: float
    val_acc: float = 0.0  # held-out top-1 accuracy (for smoothness-vs-accuracy)


def _flat_params(model: nn.Module) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _example_multipliers(z: Tensor, metaparam: Metaparam) -> Tensor:
    """Per-example loss weights ``exp(z_{c(i)})`` (==1 at z=0)."""
    return torch.exp(metaparam.example_log_weights(z))


def run_algorithm(
    subset: CifarSubset,
    metaparam: Metaparam,
    z: Tensor,
    routine: Routine,
    train_cfg: TrainConfig,
    *,
    device: torch.device,
) -> RunResult:
    """The learning algorithm ``A(z)``: deterministically train ``ResNet9`` with
    per-example loss weights ``exp(z)`` and return trained params + held-out loss.

    Determinism: parameters are initialized from ``init_seed`` and the per-epoch
    data order is fixed by ``order_seed`` -- both independent of ``z`` -- so two
    calls with equal ``z`` give identical results, while different ``z`` differ
    only through the loss weights.
    """
    configure_determinism(train_cfg.init_seed, tf32=train_cfg.tf32)
    model = ResNet9(routine, num_classes=int(subset.train_labels.max().item()) + 1)
    model.to(device)
    model.train()
    autocast_dtype = train_cfg.autocast_dtype
    use_amp = autocast_dtype is not None

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg.lr,
        momentum=train_cfg.momentum,
        weight_decay=train_cfg.weight_decay,
        nesterov=train_cfg.nesterov,
    )

    multipliers = _example_multipliers(z, metaparam).to(device)
    n = subset.n_train
    steps_per_epoch = max(1, n // train_cfg.batch_size)
    total_steps = train_cfg.epochs * steps_per_epoch

    order_gen = torch.Generator().manual_seed(train_cfg.order_seed)
    step = 0
    last_train_loss = float("nan")
    for _ in range(train_cfg.epochs):
        perm = torch.randperm(n, generator=order_gen).to(device)
        for s in range(steps_per_epoch):
            idx = perm[s * train_cfg.batch_size : (s + 1) * train_cfg.batch_size]
            images = subset.train_images[idx]
            labels = subset.train_labels[idx]
            weights = multipliers[idx]

            lr_t = _lr_at(step, total_steps, train_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr_t

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                logits = model(images)
                per_example = F.cross_entropy(logits, labels, reduction="none")
                loss = (weights * per_example).mean()
            loss.backward()
            optimizer.step()
            last_train_loss = float(loss.detach())
            step += 1

    if train_cfg.recalibrate_bn:
        _recalibrate_bn(
            model, subset.train_images, train_cfg.batch_size, autocast_dtype
        )
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
        val_logits = model(subset.val_images).float()
        val_loss = F.cross_entropy(val_logits, subset.val_labels).item()
        val_acc = (val_logits.argmax(1) == subset.val_labels).float().mean().item()

    return RunResult(
        theta=_flat_params(model).float().cpu(),
        val_loss=val_loss,
        train_loss=last_train_loss,
        val_acc=val_acc,
    )


# --------------------------------------------------------------------------- #
# The metasmoothness metrics
# --------------------------------------------------------------------------- #
@dataclass
class DirectionResult:
    """Metasmoothness of one probe direction ``v`` at ``z = 0``."""

    s_curvature: float  # Definition 1 / Eq. (5); lower = smoother
    s_hat: float  # Definition 2 / Eq. (6); higher (-> 1) = smoother
    f0: float
    fh: float
    f2h: float
    first_diff: float  # Delta_f(z;v) = (f_h - f_0)/h
    second_diff: float  # f_2h - 2 f_h + f_0
    acc0: float = 0.0  # held-out top-1 accuracy at z0


def measure_direction(
    run_fn: Callable[[Tensor], RunResult],
    z0: Tensor,
    v: Tensor,
    h: float,
) -> DirectionResult:
    """Compute both metasmoothness metrics from the three runs at ``z0``,
    ``z0 + h v`` and ``z0 + 2 h v``."""
    r0 = run_fn(z0)
    rh = run_fn(z0 + h * v)
    r2h = run_fn(z0 + 2.0 * h * v)

    f0, fh, f2h = r0.val_loss, rh.val_loss, r2h.val_loss
    second_diff = f2h - 2.0 * fh + f0
    s_curvature = abs(second_diff) / (h * h)

    # Definition 2 in parameter space.
    delta0 = (rh.theta - r0.theta) / h
    deltah = (r2h.theta - rh.theta) / h
    d = (r2h.theta - r0.theta).abs()
    total = d.sum()
    if total > 0:
        weights = d / total
        s_hat = float(
            (torch.sign(delta0) * torch.sign(deltah) * weights).sum()
        )
    else:
        s_hat = float("nan")

    return DirectionResult(
        s_curvature=s_curvature,
        s_hat=s_hat,
        f0=f0,
        fh=fh,
        f2h=f2h,
        first_diff=(fh - f0) / h,
        second_diff=second_diff,
        acc0=r0.val_acc,
    )


def sample_direction(dim: int, seed: int, *, normalize: bool, device: torch.device) -> Tensor:
    """Random probe direction ``v ~ N(0, I)`` (optionally unit-normalized)."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=gen)
    if normalize:
        v = v / v.norm().clamp_min(1e-12)
    return v.to(device)


@dataclass
class BenchmarkResult:
    routine: str
    metaparam_kind: str
    h: float
    num_directions: int
    s_curvature: list[float]
    s_hat: list[float]
    f0: float  # unperturbed held-out loss (same across directions)
    directions: list[DirectionResult]

    @property
    def s_curvature_mean(self) -> float:
        return float(np.mean(self.s_curvature))

    @property
    def s_curvature_std(self) -> float:
        return float(np.std(self.s_curvature))

    @property
    def val_acc(self) -> float:
        """Held-out top-1 accuracy at ``z = 0`` (same across directions)."""
        return self.directions[0].acc0

    @property
    def s_hat_mean(self) -> float:
        return float(np.nanmean(self.s_hat))

    @property
    def s_hat_std(self) -> float:
        return float(np.nanstd(self.s_hat))

    def summary(self) -> str:
        return (
            f"{self.routine:<10s} | z={self.metaparam_kind:<11s} | "
            f"f0={self.f0:.4f} acc={self.val_acc:.3f} | "
            f"S(Def1) {self.s_curvature_mean:9.3f} +/- {self.s_curvature_std:7.3f} "
            f"(lower=smoother) | "
            f"S_hat(Def2) {self.s_hat_mean:+.4f} +/- {self.s_hat_std:.4f} "
            f"(higher=smoother)"
        )


def benchmark_metasmoothness(
    subset: CifarSubset,
    metaparam: Metaparam,
    routine: Routine,
    train_cfg: TrainConfig,
    *,
    h: float = 0.05,
    num_directions: int = 4,
    direction_seed: int = 1000,
    normalize_directions: bool = False,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """Benchmark a routine's metasmoothness at ``z = 0`` over several random
    probe directions.  The same direction seeds are used across routines so the
    comparison is apples-to-apples."""

    def run_fn(z: Tensor) -> RunResult:
        return run_algorithm(subset, metaparam, z, routine, train_cfg, device=device)

    z0 = torch.zeros(metaparam.dim, device=device)
    directions: list[DirectionResult] = []
    for k in range(num_directions):
        v = sample_direction(
            metaparam.dim, direction_seed + k,
            normalize=normalize_directions, device=device,
        )
        result = measure_direction(run_fn, z0, v, h)
        directions.append(result)
        if progress is not None:
            progress(
                f"  [{routine.name}/{metaparam.kind}] dir {k + 1}/{num_directions}: "
                f"S={result.s_curvature:.3f}  S_hat={result.s_hat:+.4f}"
            )

    return BenchmarkResult(
        routine=routine.name,
        metaparam_kind=metaparam.kind,
        h=h,
        num_directions=num_directions,
        s_curvature=[d.s_curvature for d in directions],
        s_hat=[d.s_hat for d in directions],
        f0=directions[0].f0,
        directions=directions,
    )
