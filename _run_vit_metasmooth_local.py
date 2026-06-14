"""Phase A (ViT): measure & maximize the metasmoothness of a Vision Transformer
backbone on CIFAR-10, locally (RTX 4050) and overnight.

This is the transformer analogue of `_run_metasmooth_vm.py` (which did the same
for ResNet-9).  We reuse the paper's two finite-difference diagnostics
(arXiv:2503.13751, Defs 1 & 2) straight out of `metasmooth.py` -- they are
model-agnostic -- and only swap in a ViT learning algorithm `A(z)`:

  * The backbone is `model.VisionTransformerClassifier`, parameterized by a
    metasmoothness "menu" (`ViTRoutine` below): mean-vs-CLS pooling, pre-vs-post
    norm, GELU-vs-ReLU, the final-logit-scale lever (the ResNet-9 study's
    dominant knob), and width / depth.
  * Training is the repo's *functional smooth AdamW* (eps inside the sqrt) -- the
    paper's "smooth optimizer" menu item, and the realistic choice for ViTs
    (which underfit badly under plain SGD).  `weighted_inner_step(..., create_graph
    =False)` is just an ordinary deterministic step; no metagradients are needed
    for the diagnostic (only 3 trainings per probe direction).

Stages (mirroring the ResNet-9 Phase A):
  1. head-to-head : baseline_vit vs smooth_vit vs smooth_wide_vit, per_cluster & per_example
  2. menu search  : single-knob ablation off the baseline (per_cluster)
  3. per-example  : S_hat vs h for the smooth routine (push it off the noise floor)

fp32 throughout: fp16/bf16 flip the sign of the tiny Definition-1 second
difference.  JSON is written after *every* bench, and a soft MAX_MINUTES budget
stops launching new benches -- so an overnight run (or a thermal-throttled /
interrupted one) always leaves usable partial results.

Config via env vars (defaults sized for an overnight laptop run):
  N_TRAIN N_VAL EPOCHS BATCH LR EPS BETA1 BETA2 WD WARMUP_FRAC TEMPERATURE
  DIM DEPTH HEADS PATCH MLP_RATIO
  H HEAD_DIRS PE_DIRS ABL_DIRS WIDE_DIRS DIR_SEED MAX_MINUTES CIFAR_DIR OUT
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call

import metasmooth as ms
from _vit_menu import (
    BASELINE_VIT,
    SMOOTH_VIT,
    SMOOTH_WIDE_VIT,
    ViTGeometry,
    ViTRoutine,
)
from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from model import ViTConfig, VisionTransformerClassifier, cross_entropy_loss

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = os.environ.get("CIFAR_DIR", "./data")
OUT = os.environ.get("OUT", "artifacts/vit_metasmooth_results.json")

N_TRAIN = int(os.environ.get("N_TRAIN", "4000"))
N_VAL = int(os.environ.get("N_VAL", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "500"))

# Smooth-AdamW hyperparameters (fp32; eps kept inside the sqrt by the engine).
LR = float(os.environ.get("LR", "2e-3"))
EPS = float(os.environ.get("EPS", "1e-4"))
BETA1 = float(os.environ.get("BETA1", "0.9"))
BETA2 = float(os.environ.get("BETA2", "0.99"))
WD = float(os.environ.get("WD", "0.0"))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.3"))
TEMP = float(os.environ.get("TEMPERATURE", "1.0"))

# ViT-tiny geometry (CIFAR 32x32, patch 8 -> 16 tokens).
DIM = int(os.environ.get("DIM", "192"))
DEPTH = int(os.environ.get("DEPTH", "6"))
HEADS = int(os.environ.get("HEADS", "6"))
PATCH = int(os.environ.get("PATCH", "8"))
MLP_RATIO = float(os.environ.get("MLP_RATIO", "2.0"))

H = float(os.environ.get("H", "0.05"))
HEAD_DIRS = int(os.environ.get("HEAD_DIRS", "4"))
PE_DIRS = int(os.environ.get("PE_DIRS", "3"))
ABL_DIRS = int(os.environ.get("ABL_DIRS", "3"))
WIDE_DIRS = int(os.environ.get("WIDE_DIRS", "2"))
DIR_SEED = int(os.environ.get("DIR_SEED", "1000"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "360"))
INIT_SEED = int(os.environ.get("INIT_SEED", "0"))
ORDER_SEED = int(os.environ.get("ORDER_SEED", "1"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEOM = ViTGeometry(image_size=32, patch=PATCH, dim=DIM, depth=DEPTH,
                   heads=HEADS, mlp_ratio=MLP_RATIO)


# --------------------------------------------------------------------------- #
# The learning algorithm A(z): deterministic weighted ViT training on CIFAR-10
# --------------------------------------------------------------------------- #
def _lr_at(step: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to zero (mirrors metasmooth._lr_at)."""
    warmup = int(total_steps * WARMUP_FRAC)
    if step < warmup:
        return LR * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * LR * (1.0 + math.cos(math.pi * progress))


def _group_ids(subset: ms.CifarSubset, kind: str) -> Tensor:
    if kind == "per_cluster":
        return subset.train_labels.clone()
    return torch.arange(subset.n_train, device=subset.train_labels.device)


def _base_masses(dim: int) -> Tensor:
    return torch.full((dim,), 1.0 / dim, dtype=torch.float32, device=device)


def _flatten(params: dict[str, Tensor]) -> Tensor:
    return torch.cat([v.detach().reshape(-1).float().cpu() for v in params.values()])


def make_run_fn(routine: ViTRoutine, subset: ms.CifarSubset, kind: str):
    """Return a `z -> ms.RunResult` closure: train the menu'd ViT with per-group
    loss weights derived from `z` and report held-out CE / accuracy / theta."""
    num_classes = int(subset.train_labels.max().item()) + 1
    vit_config = routine.build_config(num_classes, GEOM)
    group_ids = _group_ids(subset, kind)
    base_cfg = SmoothAdamWConfig(
        learning_rate=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=WD
    )
    n = subset.n_train
    steps_per_epoch = max(1, n // BATCH)
    total_steps = EPOCHS * steps_per_epoch

    def run_fn(z: Tensor) -> ms.RunResult:
        ms.configure_determinism(INIT_SEED, tf32=False)
        model = VisionTransformerClassifier(vit_config).to(device=device, dtype=torch.float32)
        state = initialize_train_state(model)
        base_group_masses = _base_masses(z.numel())
        z = z.to(device=device, dtype=torch.float32)

        order_gen = torch.Generator().manual_seed(ORDER_SEED)
        step = 0
        last_train_loss = float("nan")
        for _ in range(EPOCHS):
            perm = torch.randperm(n, generator=order_gen).to(device)
            for s in range(steps_per_epoch):
                idx = perm[s * BATCH : (s + 1) * BATCH]
                opt_cfg = replace(base_cfg, learning_rate=_lr_at(step, total_steps))
                state, train_loss = weighted_inner_step(
                    model,
                    state,
                    subset.train_images[idx],
                    subset.train_labels[idx],
                    group_ids[idx],
                    z,
                    base_group_masses,
                    opt_cfg,
                    temperature=TEMP,
                    create_graph=False,
                )
                last_train_loss = float(train_loss.detach())
                step += 1

        with torch.no_grad():
            val_logits = functional_call(
                model, (state.parameters, state.buffers), (subset.val_images,)
            ).float()
            val_loss = float(cross_entropy_loss(val_logits, subset.val_labels).item())
            val_acc = float(
                (val_logits.argmax(1) == subset.val_labels).float().mean().item()
            )

        return ms.RunResult(
            theta=_flatten(state.parameters),
            val_loss=val_loss,
            train_loss=last_train_loss,
            val_acc=val_acc,
        )

    return run_fn, vit_config


# --------------------------------------------------------------------------- #
# Benchmark driver (mirrors _run_metasmooth_vm.py's record()/bench())
# --------------------------------------------------------------------------- #
subset = ms.load_cifar_subset(DATA_DIR, n_train=N_TRAIN, n_val=N_VAL, seed=0, device=device)

CONFIG = {
    "model": "vit",
    "n_train": N_TRAIN, "n_val": N_VAL, "epochs": EPOCHS, "batch_size": BATCH,
    "lr": LR, "eps": EPS, "betas": [BETA1, BETA2], "weight_decay": WD,
    "warmup_frac": WARMUP_FRAC, "temperature": TEMP,
    "dim": DIM, "depth": DEPTH, "heads": HEADS, "patch": PATCH, "mlp_ratio": MLP_RATIO,
    "h": H, "head_dirs": HEAD_DIRS, "pe_dirs": PE_DIRS, "abl_dirs": ABL_DIRS,
    "dir_seed": DIR_SEED, "amp": "off", "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
results: list[dict] = []
t0 = time.time()


def elapsed_min() -> float:
    return (time.time() - t0) / 60.0


def record(stage: str, routine: ViTRoutine, kind: str, h: float,
           ndirs: int, directions: list[ms.DirectionResult], vit_config: ViTConfig) -> None:
    s_cur = [d.s_curvature for d in directions]
    s_hat = [d.s_hat for d in directions]
    results.append({
        "stage": stage,
        "routine": routine.name,
        "metaparam": kind,
        "h": h,
        "num_directions": ndirs,
        "encoder_dim": vit_config.encoder_dim,
        "encoder_depth": vit_config.encoder_depth,
        "f0": directions[0].f0,
        "val_acc": directions[0].acc0,
        "s_curvature": s_cur,
        "s_curvature_mean": float(np.mean(s_cur)),
        "s_curvature_std": float(np.std(s_cur)),
        "s_curvature_median": float(np.median(s_cur)),
        "s_hat": s_hat,
        "s_hat_mean": float(np.nanmean(s_hat)),
        "s_hat_std": float(np.nanstd(s_hat)),
    })
    with open(OUT, "w") as fh:
        json.dump({"config": CONFIG, "results": results}, fh, indent=2)


def bench(routine: ViTRoutine, kind: str, ndirs: int, stage: str, h: float = H):
    if elapsed_min() > MAX_MINUTES:
        print(f"-- SKIP {routine.name}/{kind} ({stage}): over {MAX_MINUTES:.0f}min budget",
              flush=True)
        return None
    tb = time.time()
    run_fn, vit_config = make_run_fn(routine, subset, kind)
    dim = subset.n_train if kind == "per_example" else \
        int(subset.train_labels.max().item()) + 1
    z0 = torch.zeros(dim, device=device)
    directions: list[ms.DirectionResult] = []
    for k in range(ndirs):
        v = ms.sample_direction(dim, DIR_SEED + k, normalize=False, device=device)
        dr = ms.measure_direction(run_fn, z0, v, h)
        directions.append(dr)
        print(f"  [{routine.name}/{kind}] dir {k + 1}/{ndirs}: "
              f"S={dr.s_curvature:.3f}  S_hat={dr.s_hat:+.4f}", flush=True)
    record(stage, routine, kind, h, ndirs, directions, vit_config)
    s_cur = float(np.mean([d.s_curvature for d in directions]))
    s_hat = float(np.nanmean([d.s_hat for d in directions]))
    print(f"=> {routine.name:12s} | z={kind:<11s} | f0={directions[0].f0:.4f} "
          f"acc={directions[0].acc0:.3f} | S {s_cur:8.3f} (lower=smoother) | "
          f"S_hat {s_hat:+.4f} (higher=smoother)  "
          f"[{(time.time()-tb)/60:.1f}min bench, {elapsed_min():.1f}min total]",
          flush=True)
    return directions


print(f"### CONFIG {CONFIG}", flush=True)

print("\n### STAGE 1: head-to-head at scale", flush=True)
for kind in ("per_cluster", "per_example"):
    ndirs = HEAD_DIRS if kind == "per_cluster" else PE_DIRS
    bench(BASELINE_VIT, kind, ndirs, "head")
    bench(SMOOTH_VIT, kind, ndirs, "head")
bench(SMOOTH_WIDE_VIT, "per_cluster", WIDE_DIRS, "head")

print("\n### STAGE 2: per-cluster menu search (single knobs off baseline)", flush=True)
MENU = [
    (ViTRoutine(name="final/1", final_scale=1.0), ABL_DIRS),
    (ViTRoutine(name="final/3", final_scale=3.0), ABL_DIRS),
    (ViTRoutine(name="final/10", final_scale=10.0), ABL_DIRS),
    (ViTRoutine(name="final/30", final_scale=30.0), ABL_DIRS),
    (ViTRoutine(name="post_norm", pre_norm=False), ABL_DIRS),
    (ViTRoutine(name="cls_pool", pool="cls"), ABL_DIRS),
    (ViTRoutine(name="relu", smooth_act=False), ABL_DIRS),
    (ViTRoutine(name="width_x2", width_mult=2.0), WIDE_DIRS),
    (ViTRoutine(name="depth_x2", depth=DEPTH * 2), WIDE_DIRS),
]
for routine, nd in MENU:
    bench(routine, "per_cluster", nd, "menu")

print("\n### STAGE 3: per-example S_hat vs h for the smooth routine", flush=True)
for h in (0.1, 0.2):  # h=0.05 already covered in STAGE 1
    bench(SMOOTH_VIT, "per_example", PE_DIRS, "per_example_h", h=h)

print(f"\nALL DONE in {elapsed_min():.1f} min -> {OUT}", flush=True)
