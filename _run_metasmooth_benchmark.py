"""Benchmark + maximize metasmoothness of the ResNet-9 / CIFAR-10 routine.

Runs the paper's metasmoothness diagnostics (arXiv:2503.13751, Defs 1 & 2) on
the notebook's training routine and on the paper's smoothness "menu", for both a
per-cluster and a per-example data-weight metaparameter.  Streams progress and
writes all numbers to metasmooth_results.json for the notebook to render.
"""
from __future__ import annotations

import json
import time

import torch

import metasmooth as ms

DATA_DIR = "C:/ml/dataset-curation/meta-grad-descent-w-clustering/data"
OUT = "metasmooth_results.json"

# Scale (laptop demo; scale up n_train/epochs/num_directions on the VM).
N_TRAIN = 3000
N_VAL = 1000
TRAIN = ms.TrainConfig(epochs=16, batch_size=500, lr=0.08, amp="off")  # fp32
H = 0.05
HEAD_DIRS = 3       # directions for the baseline-vs-smooth head-to-head
ABL_DIRS = 2        # directions for the per-knob ablation
DIR_SEED = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
subset = ms.load_cifar_subset(DATA_DIR, n_train=N_TRAIN, n_val=N_VAL, seed=0, device=device)

# The paper's menu: each single modification, then all together ("smooth").
ABLATION = [
    ms.Routine(pool="avg", name="avg_pool"),
    ms.Routine(final_scale=10.0, name="final/10"),
    ms.Routine(width=2.0, name="width_x2"),
    ms.Routine(bn_before_act=False, name="bn_after_act"),  # non-smooth direction
]

results: list[dict] = []


def record(res: ms.BenchmarkResult) -> None:
    results.append(
        {
            "routine": res.routine,
            "metaparam": res.metaparam_kind,
            "h": res.h,
            "num_directions": res.num_directions,
            "f0": res.f0,
            "s_curvature": res.s_curvature,
            "s_curvature_mean": res.s_curvature_mean,
            "s_curvature_median": float(torch.tensor(res.s_curvature).median()),
            "s_hat": res.s_hat,
            "s_hat_mean": res.s_hat_mean,
            "s_hat_std": res.s_hat_std,
        }
    )
    with open(OUT, "w") as fh:
        json.dump({"config": {
            "n_train": N_TRAIN, "n_val": N_VAL, "epochs": TRAIN.epochs,
            "batch_size": TRAIN.batch_size, "lr": TRAIN.lr, "amp": TRAIN.amp,
            "h": H, "dir_seed": DIR_SEED,
        }, "results": results}, fh, indent=2)


def bench(routine, kind, ndirs):
    mp = ms.make_metaparam(kind, subset)
    res = ms.benchmark_metasmoothness(
        subset, mp, routine, TRAIN, h=H, num_directions=ndirs,
        direction_seed=DIR_SEED, device=device, progress=lambda s: print(s, flush=True),
    )
    print("=> " + res.summary(), flush=True)
    record(res)
    return res


t0 = time.time()
print("### PHASE 1: baseline vs smooth (per_cluster first for early signal)", flush=True)
for kind in ("per_cluster", "per_example"):
    bench(ms.BASELINE_ROUTINE, kind, HEAD_DIRS)
    bench(ms.SMOOTH_ROUTINE, kind, HEAD_DIRS)

print("\n### PHASE 2: per-knob ablation (per_cluster)", flush=True)
for routine in ABLATION:
    bench(routine, "per_cluster", ABL_DIRS)

print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min -> {OUT}", flush=True)
