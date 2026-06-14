"""Phase B (ViT): does the smooth ViT menu cost held-out accuracy?

Phase A measured metasmoothness under fp32-deterministic, un-augmented training
(the metric needs that).  Phase B asks the production question: if we make the
ViT the *smooth* routine (post-norm + logits/10, optionally wider), what does it
cost in **test accuracy** under normal augmented training?

This mirrors `_train_smooth_vm.py` (the ResNet-9 Phase B) but for the ViT menu:
real RandomCrop+flip augmentation, AMP, AdamW, cosine+warmup, and a learning-rate
sweep -- the /10 logit scaling rescales the loss gradient, so the smooth routine
wants a larger LR (the key ResNet-9 finding to re-confirm).  Reports best val +
final test accuracy per (routine, lr); JSON written after every run.

Config via env: N_TRAIN N_VAL EPOCHS BATCH WD WARMUP_FRAC SEED PATCH DIM DEPTH
                HEADS MLP_RATIO MAX_MINUTES CIFAR_DIR OUT
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from _vit_menu import BASELINE_VIT, SMOOTH_VIT, SMOOTH_WIDE_VIT, ViTGeometry
from model import VisionTransformerClassifier

DATA_DIR = os.environ.get("CIFAR_DIR", "./data")
OUT = os.environ.get("OUT", "vit_train_results.json")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = int(os.environ.get("BATCH", "500"))
WD = float(os.environ.get("WD", "0.05"))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", "0.1"))
N_TRAIN = int(os.environ.get("N_TRAIN", "20000"))  # subset of the 50k train split
N_VAL = int(os.environ.get("N_VAL", "5000"))
SEED = int(os.environ.get("SEED", "0"))
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "240"))

GEOM = ViTGeometry(
    image_size=32,
    patch=int(os.environ.get("PATCH", "8")),
    dim=int(os.environ.get("DIM", "192")),
    depth=int(os.environ.get("DEPTH", "6")),
    heads=int(os.environ.get("HEADS", "6")),
    mlp_ratio=float(os.environ.get("MLP_RATIO", "2.0")),
)

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # real training: fast, non-deterministic OK
AMP_DTYPE = torch.bfloat16 if device.type == "cuda" else torch.float32


def _normalize(images_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images_uint8).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(_CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    cache = os.path.join(DATA_DIR, f"cifar10_{split}.npz")
    if os.path.exists(cache):
        blob = np.load(cache)
        return blob["images"], blob["labels"]
    from torchvision import datasets

    raw = datasets.CIFAR10(root=DATA_DIR, train=(split == "train"), download=True)
    return raw.data, np.asarray(raw.targets, dtype=np.int64)


def load_data():
    """N_TRAIN train / N_VAL val (disjoint subsets of the 50k train split) + 10k test."""
    tr_imgs, tr_labels = _load_split("train")
    te_imgs, te_labels = _load_split("test")
    x = _normalize(tr_imgs)
    y = torch.from_numpy(np.asarray(tr_labels)).long()
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(x.shape[0], generator=g)
    vai = perm[:N_VAL]
    tri = perm[N_VAL : N_VAL + N_TRAIN]
    xt = _normalize(te_imgs)
    yt = torch.from_numpy(np.asarray(te_labels)).long()
    return (x[tri].to(device), y[tri].to(device),
            x[vai].to(device), y[vai].to(device),
            xt.to(device), yt.to(device))


def augment(x: torch.Tensor) -> torch.Tensor:
    """GPU RandomHorizontalFlip + RandomCrop(32, padding=4, reflect), per-image."""
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (b,), device=x.device)
    ox = torch.randint(0, 9, (b,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar[None, :])
    cols = (ox[:, None] + ar[None, :])
    pad = pad.gather(2, rows[:, None, :, None].expand(b, 3, 32, 40))
    crop = pad.gather(3, cols[:, None, None, :].expand(b, 3, 32, 32))
    return crop


def lr_at(step: int, total: int, base_lr: float) -> float:
    warm = int(total * WARMUP_FRAC)
    if step < warm:
        return base_lr * (step + 1) / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for i in range(0, x.shape[0], 1000):
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=device.type == "cuda"):
            logits = model(x[i : i + 1000]).float()
        loss_sum += F.cross_entropy(logits, y[i : i + 1000], reduction="sum").item()
        correct += (logits.argmax(1) == y[i : i + 1000]).sum().item()
        total += y[i : i + 1000].shape[0]
    return loss_sum / total, correct / total


def train_one(routine, base_lr, data):
    xtr, ytr, xva, yva, xte, yte = data
    torch.manual_seed(SEED)
    num_classes = int(ytr.max().item()) + 1
    model = VisionTransformerClassifier(
        routine.build_config(num_classes, GEOM)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=WD)
    n = xtr.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    total = EPOCHS * steps_per_epoch
    step = 0
    best_val = 0.0
    vacc = 0.0
    g = torch.Generator(device=device).manual_seed(SEED + 1)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g, device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH : (s + 1) * BATCH]
            xb = augment(xtr[idx])
            yb = ytr[idx]
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, base_lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1
        _, vacc = evaluate(model, xva, yva)
        best_val = max(best_val, vacc)
    tloss, tacc = evaluate(model, xte, yte)
    dt = time.time() - t0
    print(f"  {routine.name:12s} lr={base_lr:<7.2g} -> val_best={best_val:.4f} "
          f"test_acc={tacc:.4f} test_loss={tloss:.3f}  [{dt/60:.1f}min]", flush=True)
    return {"routine": routine.name, "lr": base_lr, "epochs": EPOCHS,
            "encoder_dim": model.config.encoder_dim,
            "val_best_acc": best_val, "val_final_acc": vacc,
            "test_acc": tacc, "test_loss": tloss, "minutes": dt / 60.0}


def main():
    data = load_data()
    print(f"loaded CIFAR: train={data[0].shape[0]} val={data[2].shape[0]} "
          f"test={data[4].shape[0]}  epochs={EPOCHS} batch={BATCH} "
          f"dim={GEOM.dim} depth={GEOM.depth}", flush=True)
    # (routine, lr) grid.  The smooth routine's /10 logit scaling divides the loss
    # gradient by ~10, so it wants a higher LR -- sweep it upward (AdamW scale).
    GRID = [
        (BASELINE_VIT, 1e-3),
        (BASELINE_VIT, 2e-3),
        (SMOOTH_VIT, 1e-3),
        (SMOOTH_VIT, 3e-3),
        (SMOOTH_VIT, 6e-3),
        (SMOOTH_WIDE_VIT, 3e-3),
        (SMOOTH_WIDE_VIT, 6e-3),
    ]
    results = []
    t0 = time.time()
    for routine, lr in GRID:
        if (time.time() - t0) / 60.0 > MAX_MINUTES:
            print(f"-- SKIP {routine.name} lr={lr}: over {MAX_MINUTES:.0f}min budget",
                  flush=True)
            continue
        results.append(train_one(routine, lr, data))
        with open(OUT, "w") as fh:
            json.dump({"config": {"epochs": EPOCHS, "batch": BATCH, "wd": WD,
                                  "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL,
                                  "dim": GEOM.dim, "depth": GEOM.depth,
                                  "heads": GEOM.heads, "patch": GEOM.patch,
                                  "model": "vit"},
                       "results": results}, fh, indent=2)
    print("\n=== best test accuracy per routine ===", flush=True)
    by = {}
    for r in results:
        if r["routine"] not in by or r["test_acc"] > by[r["routine"]]["test_acc"]:
            by[r["routine"]] = r
    for name, r in by.items():
        print(f"  {name:12s} best test_acc={r['test_acc']:.4f} @ lr={r['lr']}", flush=True)


if __name__ == "__main__":
    main()
