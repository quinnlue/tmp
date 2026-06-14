"""Focused per-example metasmoothness probe for the ViT menu (CIFAR-10, local).

Phase A's main runner only probed per_example for baseline + smooth (plus an
h-sweep).  Per-example is the *real* curation target (one weight per training
image), so this script looks at it more carefully:

  * the per-example S / S_hat across a handful of menu routines, and
  * an h-sweep (the per-example S_hat noise floor moves with the perturbation
    size, as it did for ResNet-9).

It is deliberately standalone -- it carries its own small training core so it
can't disturb the verified `experiments.cifar10.vit_metasmooth`. fp32 throughout.

Config via env: N_TRAIN N_VAL EPOCHS BATCH LR EPS DIM DEPTH HEADS PATCH
                DIRS H_SWEEP MAX_MINUTES CIFAR_DIR OUT
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
from experiments.cifar10.vit_menu import ViTGeometry, ViTRoutine
from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from model import VisionTransformerClassifier, cross_entropy_loss

DATA_DIR = os.environ.get("CIFAR_DIR", "./data")
OUT = os.environ.get("OUT", "artifacts/vit_per_example_results.json")
N_TRAIN = int(os.environ.get("N_TRAIN", "4000"))
N_VAL = int(os.environ.get("N_VAL", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "500"))
LR = float(os.environ.get("LR", "2e-3"))
EPS = float(os.environ.get("EPS", "1e-4"))
DIRS = int(os.environ.get("DIRS", "4"))
WARMUP_FRAC = 0.3
DIR_SEED = int(os.environ.get("DIR_SEED", "1000"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "60"))

GEOM = ViTGeometry(
    image_size=32,
    patch=int(os.environ.get("PATCH", "8")),
    dim=int(os.environ.get("DIM", "192")),
    depth=int(os.environ.get("DEPTH", "6")),
    heads=int(os.environ.get("HEADS", "6")),
    mlp_ratio=2.0,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _lr_at(step: int, total: int) -> float:
    warm = int(total * WARMUP_FRAC)
    if step < warm:
        return LR * (step + 1) / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return 0.5 * LR * (1.0 + math.cos(math.pi * p))


def make_run_fn(routine: ViTRoutine, subset: ms.CifarSubset):
    """z (per-example, dim = n_train) -> ms.RunResult."""
    num_classes = int(subset.train_labels.max().item()) + 1
    cfg = routine.build_config(num_classes, GEOM)
    n = subset.n_train
    group_ids = torch.arange(n, device=device)
    base_masses = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
    base_opt = SmoothAdamWConfig(learning_rate=LR, betas=(0.9, 0.99), eps=EPS)
    spe = max(1, n // BATCH)
    total = EPOCHS * spe

    def run_fn(z: Tensor) -> ms.RunResult:
        ms.configure_determinism(0, tf32=False)
        model = VisionTransformerClassifier(cfg).to(device=device, dtype=torch.float32)
        state = initialize_train_state(model)
        z = z.to(device=device, dtype=torch.float32)
        gen = torch.Generator().manual_seed(1)
        step = 0
        last = float("nan")
        for _ in range(EPOCHS):
            perm = torch.randperm(n, generator=gen).to(device)
            for s in range(spe):
                idx = perm[s * BATCH : (s + 1) * BATCH]
                opt = replace(base_opt, learning_rate=_lr_at(step, total))
                state, loss = weighted_inner_step(
                    model, state, subset.train_images[idx], subset.train_labels[idx],
                    group_ids[idx], z, base_masses, opt, create_graph=False,
                )
                last = float(loss.detach())
                step += 1
        with torch.no_grad():
            logits = functional_call(
                model, (state.parameters, state.buffers), (subset.val_images,)
            ).float()
            vloss = float(cross_entropy_loss(logits, subset.val_labels).item())
            vacc = float((logits.argmax(1) == subset.val_labels).float().mean().item())
        theta = torch.cat([v.detach().reshape(-1).float().cpu()
                           for v in state.parameters.values()])
        return ms.RunResult(theta=theta, val_loss=vloss, train_loss=last, val_acc=vacc)

    return run_fn


subset = ms.load_cifar_subset(DATA_DIR, n_train=N_TRAIN, n_val=N_VAL, seed=0, device=device)
CONFIG = {"model": "vit", "metaparam": "per_example", "n_train": N_TRAIN,
          "n_val": N_VAL, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "eps": EPS,
          "dim": GEOM.dim, "depth": GEOM.depth, "heads": GEOM.heads,
          "patch": GEOM.patch, "dirs": DIRS, "amp": "off", "device": str(device)}
results: list[dict] = []
t0 = time.time()


def elapsed():
    return (time.time() - t0) / 60.0


def probe(routine: ViTRoutine, ndirs: int, h: float, stage: str):
    if elapsed() > MAX_MINUTES:
        print(f"-- SKIP {routine.name} h={h} ({stage}): over budget", flush=True)
        return
    tb = time.time()
    run_fn = make_run_fn(routine, subset)
    z0 = torch.zeros(N_TRAIN, device=device)
    drs = []
    for k in range(ndirs):
        v = ms.sample_direction(N_TRAIN, DIR_SEED + k, normalize=False, device=device)
        dr = ms.measure_direction(run_fn, z0, v, h)
        drs.append(dr)
        print(f"  [{routine.name} h={h}] dir {k+1}/{ndirs}: "
              f"S={dr.s_curvature:.3f} S_hat={dr.s_hat:+.4f}", flush=True)
    s_cur = [d.s_curvature for d in drs]
    s_hat = [d.s_hat for d in drs]
    rec = {"stage": stage, "routine": routine.name, "h": h, "num_directions": ndirs,
           "f0": drs[0].f0, "val_acc": drs[0].acc0,
           "s_curvature_mean": float(np.mean(s_cur)), "s_curvature_std": float(np.std(s_cur)),
           "s_hat_mean": float(np.nanmean(s_hat)), "s_hat_std": float(np.nanstd(s_hat))}
    results.append(rec)
    with open(OUT, "w") as fh:
        json.dump({"config": CONFIG, "results": results}, fh, indent=2)
    print(f"=> {routine.name:10s} h={h:<4} | f0={rec['f0']:.4f} acc={rec['val_acc']:.3f} "
          f"| S {rec['s_curvature_mean']:8.3f} | S_hat {rec['s_hat_mean']:+.4f} "
          f"[{(time.time()-tb)/60:.1f}min, {elapsed():.1f}min total]", flush=True)


# Routines: baseline + the per-cluster smoothness winners + a couple of knobs.
MENU = [
    ViTRoutine(name="baseline"),
    ViTRoutine(name="final/10", final_scale=10.0),
    ViTRoutine(name="final/30", final_scale=30.0),
    ViTRoutine(name="depth_x2", depth=GEOM.depth * 2),
    ViTRoutine(name="relu", smooth_act=False),
]

print(f"### CONFIG {CONFIG}", flush=True)
print("\n### per-example menu (h=0.05)", flush=True)
for r in MENU:
    probe(r, DIRS, 0.05, "menu")

print("\n### per-example h-sweep (baseline vs final/10)", flush=True)
for h in (0.1, 0.2, 0.4):
    probe(ViTRoutine(name="baseline"), DIRS, h, "h_sweep")
    probe(ViTRoutine(name="final/10", final_scale=10.0), DIRS, h, "h_sweep")

print(f"\nDONE in {elapsed():.1f} min -> {OUT}", flush=True)
