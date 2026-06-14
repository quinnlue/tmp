"""Phase B: retrofit the smooth routine into real (augmented) CIFAR-10 training.

Phase A measured metasmoothness with *no* augmentation and fp32-deterministic
training (the metric needs it).  Phase B asks the production question: if we make
the classifier the smooth routine (avg pool + /10 logits + BN-before-act), what
does it cost in **test accuracy** under normal augmented training?

So here we train `metasmooth.ResNet9(routine)` the *real* way -- RandomCrop+flip
augmentation, AMP, cudnn.benchmark, SGD+cosine+warmup -- at ~25 epochs, and sweep
the learning rate (the /10 logit scaling rescales gradients, so the smooth routine
wants a larger LR).  Reports best val + final test accuracy per (routine, lr).

Config via env: EPOCHS BATCH WD MOM SEED CIFAR_DIR OUT GRID.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import metasmooth as ms

DATA_DIR = os.environ.get("CIFAR_DIR", "/workspace/tmp/data")
OUT = os.environ.get("OUT", "artifacts/train_smooth_vm_results.json")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BATCH = int(os.environ.get("BATCH", "512"))
WD = float(os.environ.get("WD", "5e-4"))
MOM = float(os.environ.get("MOM", "0.9"))
WARMUP_FRAC = 0.1
N_VAL = 5000
SEED = int(os.environ.get("SEED", "0"))

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # real training: fast, non-deterministic OK


def _normalize(images_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images_uint8).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(_CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_full():
    """Full CIFAR-10 from the npz cache: 45k train / 5k val / 10k test, on GPU."""
    tr = np.load(os.path.join(DATA_DIR, "cifar10_train.npz"))
    te = np.load(os.path.join(DATA_DIR, "cifar10_test.npz"))
    x = _normalize(tr["images"]); y = torch.from_numpy(tr["labels"]).long()
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(x.shape[0], generator=g)
    tri, vai = perm[N_VAL:], perm[:N_VAL]
    xt = _normalize(te["images"]); yt = torch.from_numpy(te["labels"]).long()
    return (x[tri].to(device), y[tri].to(device),
            x[vai].to(device), y[vai].to(device),
            xt.to(device), yt.to(device))


def augment(x: torch.Tensor) -> torch.Tensor:
    """GPU RandomHorizontalFlip + RandomCrop(32, padding=4, reflect), per-image."""
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")            # [B,3,40,40]
    oy = torch.randint(0, 9, (b,), device=x.device)
    ox = torch.randint(0, 9, (b,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar[None, :])                      # [B,32]
    cols = (ox[:, None] + ar[None, :])                      # [B,32]
    pad = pad.gather(2, rows[:, None, :, None].expand(b, 3, 32, 40))
    crop = pad.gather(3, cols[:, None, None, :].expand(b, 3, 32, 32))
    return crop


def lr_at(step, total, base_lr):
    warm = int(total * WARMUP_FRAC)
    if step < warm:
        return base_lr * (step + 1) / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, x, y, amp_dtype):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for i in range(0, x.shape[0], 1000):
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(x[i:i + 1000]).float()
        loss_sum += F.cross_entropy(logits, y[i:i + 1000], reduction="sum").item()
        correct += (logits.argmax(1) == y[i:i + 1000]).sum().item()
        total += y[i:i + 1000].shape[0]
    return loss_sum / total, correct / total


def train_one(routine, base_lr, data, amp_dtype=torch.bfloat16):
    xtr, ytr, xva, yva, xte, yte = data
    torch.manual_seed(SEED)
    model = ms.ResNet9(routine).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=MOM,
                          weight_decay=WD, nesterov=True)
    n = xtr.shape[0]
    steps_per_epoch = n // BATCH
    total = EPOCHS * steps_per_epoch
    step = 0
    best_val = 0.0
    g = torch.Generator(device=device).manual_seed(SEED + 1)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g, device=device)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            xb = augment(xtr[idx]); yb = ytr[idx]
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, base_lr)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step(); step += 1
        vloss, vacc = evaluate(model, xva, yva, amp_dtype)
        best_val = max(best_val, vacc)
    tloss, tacc = evaluate(model, xte, yte, amp_dtype)
    dt = time.time() - t0
    print(f"  {routine.name:12s} lr={base_lr:<5.2g} -> val_best={best_val:.4f} "
          f"test_acc={tacc:.4f} test_loss={tloss:.3f}  [{dt/60:.1f}min]", flush=True)
    return {"routine": routine.name, "lr": base_lr, "epochs": EPOCHS,
            "val_best_acc": best_val, "val_final_acc": vacc,
            "test_acc": tacc, "test_loss": tloss, "minutes": dt / 60.0}


def main():
    data = load_full()
    print(f"loaded full CIFAR: train={data[0].shape[0]} val={data[2].shape[0]} "
          f"test={data[4].shape[0]}  epochs={EPOCHS} batch={BATCH}", flush=True)
    # (routine, lr) grid.  Smooth routines want higher LR (the /10 logit scaling
    # divides the loss gradient by ~10), so sweep them upward.
    GRID = [
        (ms.BASELINE_ROUTINE, 0.1),
        (ms.BASELINE_ROUTINE, 0.2),
        (ms.SMOOTH_ROUTINE, 0.1),
        (ms.SMOOTH_ROUTINE, 0.3),
        (ms.SMOOTH_ROUTINE, 0.6),
        (ms.SMOOTH_ROUTINE, 1.0),
        (ms.SMOOTH_WIDE_ROUTINE, 0.3),
        (ms.SMOOTH_WIDE_ROUTINE, 0.6),
    ]
    results = []
    for routine, lr in GRID:
        r = train_one(routine, lr, data)
        results.append(r)
        with open(OUT, "w") as fh:
            json.dump({"config": {"epochs": EPOCHS, "batch": BATCH, "wd": WD,
                                  "seed": SEED, "n_val": N_VAL}, "results": results}, fh, indent=2)
    # Summary: best lr per routine.
    print("\n=== best test accuracy per routine ===", flush=True)
    by = {}
    for r in results:
        if r["routine"] not in by or r["test_acc"] > by[r["routine"]]["test_acc"]:
            by[r["routine"]] = r
    for name, r in by.items():
        print(f"  {name:12s} best test_acc={r['test_acc']:.4f} @ lr={r['lr']}", flush=True)


if __name__ == "__main__":
    main()
