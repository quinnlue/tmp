"""Phase C: per-cluster metagradient data curation on the smooth routine.

Now that the routine is metasmooth (Phase A) we compute *real* metagradients
d phi / d z of the held-out cross-entropy through smooth-routine training, and run
metagradient descent (MGD) on the per-cluster data weights z in R^10.

Reuse: the repo's differentiable engine -- `functional_train` (smooth AdamW inner
optimizer + `weighted_inner_step`, eps inside the sqrt), the `weighting`
group-softmax reparam, and the metagradient backend chosen by METHOD:
  * METHOD=replay (default): `recursive_replay.recursive_replay_state` -- O(k log T)
    memory, so the inner training can be long (store-all OOMs past ~50 steps here).
  * METHOD=store_all: `metagrad.train_unrolled` -- differentiate the whole unroll;
    only fits short horizons, useful for a quick smoke.
The curation model is `metasmooth.ResNet9(SMOOTH_GN_ROUTINE)` (GroupNorm => no BN
running-stat buffers, so exact metagradients through training are clean).

z parameterizes per-class loss weights via the count/softmax relaxation:
example i gets multiplier softmax(z)[class_i] / base[class_i], with base = uniform
(1/10), so z = 0 is ordinary training.  MGD descends z on held-out CE.

Config via env: N_POOL N_OBJ N_VAL INNER_EPOCHS BATCH INNER_LR META_STEPS META_LR
                SEED CIFAR_DIR OUT.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import metasmooth as ms
from functional_train import (
    SmoothAdamWConfig, initialize_train_state, weighted_inner_step,
)
from metagrad import InnerBatch, ObjectiveBatch, train_unrolled, classification_objective
from recursive_replay import recursive_replay_state

DATA_DIR = os.environ.get("CIFAR_DIR", "/workspace/tmp/data")
OUT = os.environ.get("OUT", "artifacts/phase_c_mgd_results.json")
N_POOL = int(os.environ.get("N_POOL", "4000"))      # curation training pool
N_OBJ = int(os.environ.get("N_OBJ", "2000"))        # held-out objective (MGD target)
N_VAL = int(os.environ.get("N_VAL", "2000"))        # separate validation (generalization)
INNER_EPOCHS = int(os.environ.get("INNER_EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "250"))
INNER_LR = float(os.environ.get("INNER_LR", "5e-3"))
META_STEPS = int(os.environ.get("META_STEPS", "20"))
META_LR = float(os.environ.get("META_LR", "0.1"))
METHOD = os.environ.get("METHOD", "replay")          # "replay" | "store_all"
BRANCHING = int(os.environ.get("BRANCHING", "4"))    # REPLAY k-ary branching factor
SEED = int(os.environ.get("SEED", "0"))
# Optional label-noise stress test: randomize a fraction of the POOL labels for the
# given true classes (objective/val stay clean).  MGD should down-weight exactly
# these classes.  e.g. CORRUPT_CLASSES="3,5" CORRUPT_FRAC=1.0
CORRUPT_CLASSES = [int(c) for c in os.environ.get("CORRUPT_CLASSES", "").split(",") if c.strip()]
CORRUPT_FRAC = float(os.environ.get("CORRUPT_FRAC", "0.0"))
# Optional distribution-shift test: restrict the held-out objective/val to these
# target classes (the pool stays balanced over all 10).  MGD should UP-weight the
# target classes -- the clean per-cluster curation demo.  e.g. TARGET_CLASSES="0,1,8,9"
TARGET_CLASSES = [int(c) for c in os.environ.get("TARGET_CLASSES", "").split(",") if c.strip()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ms.configure_determinism(SEED)

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def load_split():
    """Disjoint pool / objective / val subsets from the CIFAR-10 train split.

    Returns pool = (images, train_labels, group_labels): train_labels feed the CE
    loss (optionally corrupted), group_labels are the TRUE class used for per-cluster
    weighting, so MGD can down-weight a class even when its labels are noisy.
    """
    full = ms.load_cifar_subset(DATA_DIR, n_train=N_POOL + N_OBJ + N_VAL,
                                n_val=1, seed=SEED, device=device)
    x, y = full.train_images, full.train_labels
    xp, ytrue = x[:N_POOL], y[:N_POOL]
    ytrain = ytrue.clone()
    if CORRUPT_CLASSES and CORRUPT_FRAC > 0:
        gen = torch.Generator(device=ytrue.device).manual_seed(SEED + 7)
        in_bad = torch.zeros_like(ytrue, dtype=torch.bool)
        for c in CORRUPT_CLASSES:
            in_bad |= ytrue == c
        flip = in_bad & (torch.rand(ytrue.shape, generator=gen, device=ytrue.device) < CORRUPT_FRAC)
        offset = torch.randint(1, 10, ytrue.shape, generator=gen, device=ytrue.device)
        ytrain = torch.where(flip, (ytrue + offset) % 10, ytrue)  # always a wrong label
        print(f"corrupted {int(flip.sum())} pool labels in classes {CORRUPT_CLASSES} "
              f"(frac={CORRUPT_FRAC})", flush=True)
    pool = (xp, ytrain, ytrue)
    obj = (x[N_POOL:N_POOL + N_OBJ], y[N_POOL:N_POOL + N_OBJ])
    val = (x[N_POOL + N_OBJ:], y[N_POOL + N_OBJ:])

    if TARGET_CLASSES:
        def keep(xy):
            xx, yy = xy
            m = torch.zeros_like(yy, dtype=torch.bool)
            for c in TARGET_CLASSES:
                m |= yy == c
            return xx[m], yy[m]
        obj, val = keep(obj), keep(val)
        print(f"distribution shift: held-out objective restricted to classes "
              f"{TARGET_CLASSES} (obj={obj[0].shape[0]} val={val[0].shape[0]})", flush=True)
    return pool, obj, val


def build_trajectory(pool):
    """Fixed deterministic inner-training trajectory (only z varies across MGD)."""
    images, train_labels, group_labels = pool
    n = images.shape[0]
    steps_per_epoch = n // BATCH
    gen = torch.Generator().manual_seed(SEED + 1)
    traj = []
    for _ in range(INNER_EPOCHS):
        perm = torch.randperm(n, generator=gen).to(device)
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            traj.append(InnerBatch(images[idx], train_labels[idx], group_labels[idx]))
    return traj


@torch.no_grad()
def eval_ce(state, model, xy):
    x, y = xy
    ce = 0.0
    for i in range(0, x.shape[0], 1000):
        ce += classification_objective(
            model, state, ObjectiveBatch(x[i:i + 1000], y[i:i + 1000])
        ).item() * min(1000, x.shape[0] - i)
    return ce / x.shape[0]


def main():
    pool, obj, val = load_split()
    print(f"pool={pool[0].shape[0]} obj={obj[0].shape[0]} val={val[0].shape[0]} "
          f"inner_steps={INNER_EPOCHS * (N_POOL // BATCH)} meta_steps={META_STEPS}",
          flush=True)
    model = ms.ResNet9(ms.SMOOTH_GN_ROUTINE, num_classes=10).to(device)
    model.train()
    traj = build_trajectory(pool)
    total_steps = len(traj)
    obj_batch = ObjectiveBatch(*obj)
    opt_cfg = SmoothAdamWConfig(learning_rate=INNER_LR, weight_decay=0.0)
    base = torch.full((10,), 0.1, device=device)   # uniform base group masses

    def replay_step(state, idx, zz, create_graph):
        b = traj[idx]
        next_state, _ = weighted_inner_step(
            model, state, b.images, b.labels, b.group_ids, zz, base, opt_cfg,
            temperature=1.0, create_graph=create_graph,
        )
        return next_state

    def trained_state(zz, differentiable):
        """Run the inner training and return the final TrainState (which carries a
        grad path to zz when differentiable)."""
        if METHOD == "replay":
            return recursive_replay_state(initialize_train_state(model), zz,
                                          total_steps, replay_step,
                                          branching_factor=BRANCHING)
        return train_unrolled(model, initialize_train_state(model), traj, zz, base,
                              opt_cfg, temperature=1.0, create_graph=differentiable)

    z = torch.zeros(10, device=device, requires_grad=True)
    meta_opt = torch.optim.Adam([z], lr=META_LR)

    history = []
    for step in range(META_STEPS + 1):
        differentiable = step < META_STEPS
        final_state = trained_state(z, differentiable)
        obj_ce = classification_objective(model, final_state, obj_batch)
        val_ce = eval_ce(final_state, model, val)
        weights = (10.0 * torch.softmax(z, 0)).detach().cpu().tolist()  # per-class mult
        rec = {"step": step, "obj_ce": float(obj_ce.detach()), "val_ce": val_ce,
               "z": z.detach().cpu().tolist(), "class_weights": weights}
        history.append(rec)
        print(f"step {step:2d}: obj_CE={rec['obj_ce']:.4f} val_CE={val_ce:.4f}  "
              f"w[min={min(weights):.2f} max={max(weights):.2f}]", flush=True)
        with open(OUT, "w") as fh:
            json.dump({"config": {"method": METHOD, "n_pool": N_POOL, "n_obj": N_OBJ,
                                  "n_val": N_VAL, "inner_epochs": INNER_EPOCHS,
                                  "batch": BATCH, "total_steps": total_steps,
                                  "inner_lr": INNER_LR, "meta_lr": META_LR,
                                  "meta_steps": META_STEPS,
                                  "corrupt_classes": CORRUPT_CLASSES,
                                  "corrupt_frac": CORRUPT_FRAC,
                                  "target_classes": TARGET_CLASSES}, "history": history}, fh, indent=2)
        if step == META_STEPS:
            break
        (g,) = torch.autograd.grad(obj_ce, z)
        z.grad = g
        meta_opt.step()
        meta_opt.zero_grad()

    base_ce, final_ce = history[0]["obj_ce"], history[-1]["obj_ce"]
    base_val, final_val = history[0]["val_ce"], history[-1]["val_ce"]
    print(f"\n=== MGD per-cluster summary ===", flush=True)
    print(f"held-out objective CE: {base_ce:.4f} (uniform) -> {final_ce:.4f} "
          f"({100*(base_ce-final_ce)/base_ce:+.1f}%)", flush=True)
    print(f"separate val CE:       {base_val:.4f} -> {final_val:.4f}", flush=True)
    final_w = history[-1]["class_weights"]
    order = sorted(range(10), key=lambda i: final_w[i])
    print("class weights (low->high):", flush=True)
    for i in order:
        tag = "  <-- CORRUPTED" if i in CORRUPT_CLASSES else (
            "  <-- TARGET" if i in TARGET_CLASSES else "")
        print(f"  {CLASSES[i]:11s} {final_w[i]:.3f}{tag}", flush=True)
    if CORRUPT_CLASSES:
        bad = sum(final_w[i] for i in CORRUPT_CLASSES) / len(CORRUPT_CLASSES)
        good = sum(final_w[i] for i in range(10) if i not in CORRUPT_CLASSES) / (10 - len(CORRUPT_CLASSES))
        print(f"mean weight: corrupted={bad:.3f}  clean={good:.3f}  "
              f"(MGD should make corrupted < clean)", flush=True)
    if TARGET_CLASSES:
        tw = sum(final_w[i] for i in TARGET_CLASSES) / len(TARGET_CLASSES)
        ow = sum(final_w[i] for i in range(10) if i not in TARGET_CLASSES) / (10 - len(TARGET_CLASSES))
        print(f"mean weight: target={tw:.3f}  off-target={ow:.3f}  "
              f"(MGD should make target > off-target)", flush=True)


if __name__ == "__main__":
    main()
