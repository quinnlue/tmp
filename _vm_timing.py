"""Throwaway A40 timing calibration for the metasmoothness benchmark.

Times a single deterministic ResNet-9 training A(z) at a few (n_train, epochs)
scales so we can pick a Phase-A scale with sane wall-clock.  fp32 throughout.
"""
from __future__ import annotations

import time
import torch
import metasmooth as ms

DATA_DIR = "/workspace/tmp/data"
device = torch.device("cuda")

# warm CUDA + download/normalize CIFAR once at the biggest size we'll probe
big = ms.load_cifar_subset(DATA_DIR, n_train=20000, n_val=2000, seed=0, device=device)
print(f"loaded CIFAR subset: {big.n_train} train / {big.n_val} val", flush=True)


def time_run(n_train, epochs, routine, batch_size=500, reps=2):
    sub = ms.CifarSubset(
        big.train_images[:n_train], big.train_labels[:n_train],
        big.val_images, big.val_labels,
    )
    mp = ms.make_metaparam("per_cluster", sub)
    z = torch.zeros(mp.dim, device=device)
    cfg = ms.TrainConfig(epochs=epochs, batch_size=batch_size, lr=0.08, amp="off")
    # warmup (cudnn autotune off, but first call still pays allocation)
    ms.run_algorithm(sub, mp, z, routine, cfg, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        r = ms.run_algorithm(sub, mp, z, routine, cfg, device=device)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    print(f"  n={n_train:6d} ep={epochs:3d} bs={batch_size} {routine.name:12s} "
          f"-> {dt:6.2f}s/run  val_loss={r.val_loss:.3f}", flush=True)
    return dt


print("=== timing (one A(z) run; a probe direction = 3 of these) ===", flush=True)
for routine in (ms.BASELINE_ROUTINE, ms.SMOOTH_ROUTINE, ms.SMOOTH_WIDE_ROUTINE):
    time_run(3000, 16, routine)
print("--- candidate Phase-A scales ---", flush=True)
for n, ep in [(10000, 25), (20000, 25)]:
    for routine in (ms.BASELINE_ROUTINE, ms.SMOOTH_WIDE_ROUTINE):
        time_run(n, ep, routine, batch_size=512)
