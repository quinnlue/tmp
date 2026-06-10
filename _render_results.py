"""Render metasmooth_results.json -> a bar-chart PNG and a markdown summary."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("metasmooth_results.json") as fh:
    blob = json.load(fh)
cfg = blob["config"]
rows = blob["results"]


def get(routine, metaparam):
    for r in rows:
        if r["routine"] == routine and r["metaparam"] == metaparam:
            return r
    return None


print(f"config: {cfg}\n")

# Head-to-head table.
print("### Baseline vs smooth (held-out classification loss as phi)\n")
print("| metaparam | routine | val_loss f0 | S (Def1) lower=smoother | S_hat (Def2) higher=smoother |")
print("|---|---|---|---|---|")
for kind in ("per_cluster", "per_example"):
    for routine in ("baseline", "smooth"):
        r = get(routine, kind)
        if r:
            print(f"| {kind} | {routine} | {r['f0']:.3f} | "
                  f"{r['s_curvature_mean']:.2f} | {r['s_hat_mean']:+.3f} |")

# Ablation (per_cluster).
order = ["baseline", "avg_pool", "final/10", "bn_after_act", "width_x2", "smooth", "smooth_wide"]
abl = [get(n, "per_cluster") for n in order]
abl = [(n, r) for n, r in zip(order, abl) if r is not None]

print("\n### Per-cluster ablation\n")
print("| routine | S (Def1) | S_hat (Def2) |")
print("|---|---|---|")
for n, r in abl:
    print(f"| {n} | {r['s_curvature_mean']:.2f} | {r['s_hat_mean']:+.3f} |")

# Plot.
names = [n for n, _ in abl]
shat = [r["s_hat_mean"] for _, r in abl]
scur = [r["s_curvature_mean"] for _, r in abl]
colors = ["#2a9d8f" if n in ("smooth", "smooth_wide") else
          ("#e76f51" if n == "bn_after_act" else "#5a6472") for n in names]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].bar(names, shat, color=colors); axes[0].axhline(0, color="k", lw=.6)
axes[0].set_title(r"empirical metasmoothness $\hat S$ (Def 2) — higher = smoother")
axes[1].bar(names, scur, color=colors)
axes[1].set_title(r"curvature $S$ (Def 1) — lower = smoother")
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
fig.suptitle("Metasmoothness of the ResNet-9 / CIFAR-10 routine (per-cluster data weights)")
plt.tight_layout()
plt.savefig("metasmooth_results.png", dpi=110, bbox_inches="tight")
print("\nwrote metasmooth_results.png")
