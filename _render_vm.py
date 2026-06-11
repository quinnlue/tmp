"""Render metasmooth_vm_results.json -> markdown tables + PNGs.

Tables: head-to-head (per_cluster & per_example), per-cluster menu/combo ranking,
per-example S_hat vs h.  Plots: ablation bars (S_hat, S) and a
smoothness-vs-accuracy scatter (the paper's Fig-4-style tradeoff curve).
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "metasmooth_vm_results.json"
with open(PATH) as fh:
    blob = json.load(fh)
cfg, rows = blob["config"], blob["results"]


def find(stage=None, routine=None, metaparam=None, h=None):
    out = []
    for r in rows:
        if stage is not None and r["stage"] != stage:
            continue
        if routine is not None and r["routine"] != routine:
            continue
        if metaparam is not None and r["metaparam"] != metaparam:
            continue
        if h is not None and abs(r["h"] - h) > 1e-9:
            continue
        out.append(r)
    return out


print(f"config: {cfg}\n")

# ---- Head-to-head ---------------------------------------------------------- #
print("### Head-to-head (phi = held-out cross-entropy)\n")
print("| metaparam | routine | f0 | val_acc | S (Def1) ↓ | S_hat (Def2) ↑ |")
print("|---|---|---|---|---|---|")
for kind in ("per_cluster", "per_example"):
    for r in find(stage="head", metaparam=kind):
        print(f"| {kind} | {r['routine']} | {r['f0']:.3f} | {r['val_acc']:.3f} | "
              f"{r['s_curvature_mean']:.2f} ± {r['s_curvature_std']:.2f} | "
              f"{r['s_hat_mean']:+.3f} ± {r['s_hat_std']:.3f} |")

# ---- Per-cluster menu + combos, ranked by S_hat ---------------------------- #
pc = [r for r in rows if r["metaparam"] == "per_cluster"]
# de-dup by routine name keeping the most-direction measurement
best = {}
for r in pc:
    k = r["routine"]
    if k not in best or r["num_directions"] > best[k]["num_directions"]:
        best[k] = r
ranked = sorted(best.values(), key=lambda r: r["s_hat_mean"], reverse=True)

print("\n### Per-cluster routines ranked by S_hat (higher = smoother)\n")
print("| routine | stage | dirs | f0 | val_acc | S (Def1) ↓ | S_hat (Def2) ↑ |")
print("|---|---|---|---|---|---|---|")
for r in ranked:
    print(f"| {r['routine']} | {r['stage']} | {r['num_directions']} | {r['f0']:.3f} | "
          f"{r['val_acc']:.3f} | {r['s_curvature_mean']:.2f} | "
          f"{r['s_hat_mean']:+.3f} ± {r['s_hat_std']:.3f} |")

# ---- Per-example S_hat vs h ------------------------------------------------ #
pe = sorted((r for r in rows if r["metaparam"] == "per_example"), key=lambda r: r["h"])
if pe:
    print("\n### Per-example S_hat vs h (smooth routine)\n")
    print("| h | dirs | S (Def1) | S_hat (Def2) |")
    print("|---|---|---|---|")
    for r in pe:
        print(f"| {r['h']:.2f} | {r['num_directions']} | {r['s_curvature_mean']:.2f} | "
              f"{r['s_hat_mean']:+.3f} ± {r['s_hat_std']:.3f} |")

# ---- Plots ----------------------------------------------------------------- #
names = [r["routine"] for r in ranked]
shat = [r["s_hat_mean"] for r in ranked]
scur = [r["s_curvature_mean"] for r in ranked]
acc = [r["val_acc"] for r in ranked]


def color(n):
    if n in ("smooth", "smooth_wide") or n.startswith("avg+"):
        return "#2a9d8f"
    if n == "bn_after_act":
        return "#e76f51"
    return "#5a6472"


colors = [color(n) for n in names]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(names, shat, color=colors); axes[0].axhline(0, color="k", lw=.6)
axes[0].set_title(r"empirical metasmoothness $\hat S$ (Def 2) — higher = smoother")
axes[1].bar(names, scur, color=colors)
axes[1].set_title(r"curvature $S$ (Def 1) — lower = smoother")
for ax in axes:
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
fig.suptitle(f"Metasmoothness ranking (per-cluster, n_train={cfg['n_train']}, "
             f"{cfg['epochs']} ep, fp32)")
plt.tight_layout()
plt.savefig("metasmooth_vm_ranking.png", dpi=110, bbox_inches="tight")
print("\nwrote metasmooth_vm_ranking.png")

# Smoothness vs accuracy (Fig-4 style tradeoff).
fig2, ax = plt.subplots(figsize=(6.2, 5))
ax.scatter(acc, shat, c=colors, s=70, zorder=3)
for n, a, s in zip(names, acc, shat):
    ax.annotate(n, (a, s), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("held-out accuracy at z=0")
ax.set_ylabel(r"empirical metasmoothness $\hat S$")
ax.set_title("smoothness vs accuracy (per-cluster)")
ax.grid(True, alpha=.3)
plt.tight_layout()
plt.savefig("metasmooth_vm_tradeoff.png", dpi=110, bbox_inches="tight")
print("wrote metasmooth_vm_tradeoff.png")
