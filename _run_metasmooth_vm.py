"""Phase A: solidify & scale the metasmoothness measurement (VM / A40).

Runs the paper's metasmoothness diagnostics (arXiv:2503.13751, Defs 1 & 2) at a
larger scale than the laptop demo, across four stages:

  1. head-to-head   : baseline vs smooth vs smooth_wide, per_cluster & per_example
  2. menu search    : single-knob ablation + final_scale / width sweeps (per_cluster)
  3. combos         : greedy combinations of the winning knobs (per_cluster)
  4. per-example    : push S_hat off its noise floor by varying h / #directions

All numbers (incl. held-out accuracy for the smoothness-vs-accuracy curve) stream
to a JSON for rendering.  fp32 throughout (amp="off"): fp16/bf16 flip the sign of
the tiny Definition-1 second difference.

Config via env vars (defaults chosen from A40 timing calibration):
  N_TRAIN N_VAL EPOCHS BATCH LR H HEAD_DIRS ABL_DIRS DIR_SEED CIFAR_DIR OUT
"""
from __future__ import annotations

import json
import os
import time

import torch

import metasmooth as ms

DATA_DIR = os.environ.get("CIFAR_DIR", "/workspace/tmp/data")
OUT = os.environ.get("OUT", "metasmooth_vm_results.json")

N_TRAIN = int(os.environ.get("N_TRAIN", "6000"))
N_VAL = int(os.environ.get("N_VAL", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "18"))
BATCH = int(os.environ.get("BATCH", "500"))
LR = float(os.environ.get("LR", "0.08"))
H = float(os.environ.get("H", "0.05"))
HEAD_DIRS = int(os.environ.get("HEAD_DIRS", "6"))    # per_cluster head-to-head
PE_DIRS = int(os.environ.get("PE_DIRS", "4"))        # per_example (noisier, costlier)
ABL_DIRS = int(os.environ.get("ABL_DIRS", "4"))      # menu / combo single-knobs
WIDE_DIRS = int(os.environ.get("WIDE_DIRS", "3"))    # 3x-cost wide routines
DIR_SEED = int(os.environ.get("DIR_SEED", "1000"))
# Soft wall-clock budget: stop launching new benches once exceeded (the JSON is
# written after every bench, so partial results are always saved).
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "90"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN = ms.TrainConfig(epochs=EPOCHS, batch_size=BATCH, lr=LR, amp="off")  # fp32
subset = ms.load_cifar_subset(DATA_DIR, n_train=N_TRAIN, n_val=N_VAL, seed=0, device=device)

CONFIG = {
    "n_train": N_TRAIN, "n_val": N_VAL, "epochs": EPOCHS, "batch_size": BATCH,
    "lr": LR, "amp": TRAIN.amp, "h": H, "head_dirs": HEAD_DIRS,
    "abl_dirs": ABL_DIRS, "dir_seed": DIR_SEED, "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
results: list[dict] = []


def record(res: ms.BenchmarkResult, stage: str, h: float) -> None:
    results.append({
        "stage": stage,
        "routine": res.routine,
        "metaparam": res.metaparam_kind,
        "h": h,
        "num_directions": res.num_directions,
        "f0": res.f0,
        "val_acc": res.val_acc,
        "s_curvature": res.s_curvature,
        "s_curvature_mean": res.s_curvature_mean,
        "s_curvature_std": res.s_curvature_std,
        "s_curvature_median": float(torch.tensor(res.s_curvature).median()),
        "s_hat": res.s_hat,
        "s_hat_mean": res.s_hat_mean,
        "s_hat_std": res.s_hat_std,
    })
    with open(OUT, "w") as fh:
        json.dump({"config": CONFIG, "results": results}, fh, indent=2)


t0 = time.time()


def elapsed_min() -> float:
    return (time.time() - t0) / 60.0


def bench(routine, kind, ndirs, stage, h=H):
    if elapsed_min() > MAX_MINUTES:
        print(f"-- SKIP {routine.name}/{kind} ({stage}): over {MAX_MINUTES:.0f}min budget", flush=True)
        return None
    tb = time.time()
    mp = ms.make_metaparam(kind, subset)
    res = ms.benchmark_metasmoothness(
        subset, mp, routine, TRAIN, h=h, num_directions=ndirs,
        direction_seed=DIR_SEED, device=device,
        progress=lambda s: print(s, flush=True),
    )
    record(res, stage, h)
    print(f"=> {res.summary()}  [{(time.time()-tb)/60:.1f}min bench, "
          f"{elapsed_min():.1f}min total]", flush=True)
    return res


print(f"### CONFIG {CONFIG}", flush=True)

print("\n### STAGE 1: head-to-head at scale", flush=True)
# baseline + smooth on both metaparams; smooth_wide (3x cost) per_cluster only.
for kind in ("per_cluster", "per_example"):
    ndirs = HEAD_DIRS if kind == "per_cluster" else PE_DIRS
    bench(ms.BASELINE_ROUTINE, kind, ndirs, "head")
    bench(ms.SMOOTH_ROUTINE, kind, ndirs, "head")
bench(ms.SMOOTH_WIDE_ROUTINE, "per_cluster", WIDE_DIRS, "head")

print("\n### STAGE 2: per-cluster menu search (single knobs + sweeps)", flush=True)
MENU = [
    (ms.Routine(pool="avg", name="avg_pool"), ABL_DIRS),
    (ms.Routine(final_scale=3.0, name="final/3"), ABL_DIRS),
    (ms.Routine(final_scale=10.0, name="final/10"), ABL_DIRS),
    (ms.Routine(final_scale=30.0, name="final/30"), ABL_DIRS),
    (ms.Routine(bn_before_act=False, name="bn_after_act"), ABL_DIRS),  # non-smooth dir
    (ms.Routine(smooth_act=True, name="gelu"), ABL_DIRS),
    (ms.Routine(width=2.0, name="width_x2"), WIDE_DIRS),  # 3x cost
]
for routine, nd in MENU:
    bench(routine, "per_cluster", nd, "menu")

print("\n### STAGE 3: greedy combos (per_cluster)", flush=True)
# avg+f10 == SMOOTH_ROUTINE (already in STAGE 1); test genuinely new combos:
# does f30 beat f10 when combined with avg, and does GELU add on top of smooth?
COMBOS = [
    (ms.Routine(pool="avg", final_scale=30.0, name="avg+f30"), ABL_DIRS),
    (ms.Routine(pool="avg", final_scale=10.0, smooth_act=True, name="smooth+gelu"), ABL_DIRS),
]
for routine, nd in COMBOS:
    bench(routine, "per_cluster", nd, "combo")

print("\n### STAGE 4: per-example S_hat vs h for the smooth routine", flush=True)
for h in (0.1, 0.2):  # h=0.05 already covered in STAGE 1
    bench(ms.SMOOTH_ROUTINE, "per_example", PE_DIRS, "per_example_h", h=h)

print(f"\nALL DONE in {elapsed_min():.1f} min -> {OUT}", flush=True)
