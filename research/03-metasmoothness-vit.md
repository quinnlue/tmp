# Metasmoothness of a Vision Transformer backbone (CIFAR-10)

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

## Goal

Measure how *metasmooth* a Vision Transformer training routine is on CIFAR-10
— i.e. how amenable it is to metagradient-based data curation — and maximize
it with a transformer "menu". This is the **transformer analogue** of the
ResNet-9 metasmoothness study, reusing the paper's two finite-difference
diagnostics (arXiv:2503.13751, Defs 1 & 2; 3 deterministic trainings per
probe, no metagradients). The diagnostics live in `metasmooth.py` and are
model-agnostic; only the learning algorithm `A(z)` changes to a ViT trained
with the repo's functional **smooth AdamW** (eps inside the sqrt).

Scope: **Phases A + B** only, run locally on the RTX 4050 laptop overnight in
fp32 — mirroring the ResNet-9 procedure's Phase A (ranking) and Phase B
(accuracy check), with menu levers = final-logit scale {1, 3, 10, 30},
pre- vs post-norm, mean vs CLS pooling, GELU vs ReLU, width, depth.

- `φ` = held-out **cross-entropy** on a disjoint CIFAR-10 split.
- `z` = continuous **data weights at z=0** (count relaxation): a per-group
  softmax reweighting (`weighting.weighted_example_loss`); `z=0` ⇒ uniform =
  ordinary training. Probed for **per-cluster** (one weight per CIFAR-10
  class, dim 10) and **per-example** (one weight per image, dim = n_train).

## Setup

- Repo: `C:\ml\dataset-curation\meta-grad-descent-w-clustering` — functional
  (not JAX) PyTorch.
- Environment: `C:/Users/luequ/micromamba/envs/torch311/python.exe` — torch
  2.8.0+cu128, torchvision 0.23.0+cu128, CUDA on an **RTX 4050 Laptop GPU**.
- Data: CIFAR-10 npz cache at `./data/cifar10_{train,test}.npz` (build with
  `python _build_cifar_cache.py ./data`).
- Diagnostics: Definition 1 (curvature) `S = |f(z+2hv) - 2f(z+hv) + f(z)| / h^2`
  (**lower = smoother**); Definition 2 (empirical metasmoothness)
  `Ŝ = sign(Δ_A(z;v))^T diag(d/‖d‖₁) sign(Δ_A(z+hv;v)) ∈ [-1,1]` (**higher =
  smoother**).

## Method

### The ViT "menu" (`ViTRoutine`, `_vit_menu.py`)

A frozen dataclass `ViTRoutine`, the transformer analogue of
`metasmooth.Routine`:

| lever | field | smooth choice | notes |
|---|---|---|---|
| final-logit scale | `final_scale` | 10.0 (÷10) | the **dominant** lever (ResNet-9 finding, reproduced) |
| norm placement | `pre_norm` | `False` (post-norm) | transformer-specific; worktree probes favored post-norm for clean metagradients |
| token pooling | `pool` | `"mean"` | average pooling is the paper's smooth choice; `"cls"` is the standard default |
| activation | `smooth_act` | `True` (GELU) | GELU vs ReLU |
| width | `width_mult` | 2.0 | wider == more metasmooth |
| depth | `depth` | — | extra capacity lever |

(Note: the `pre_norm`/`smooth choice` cell above reflects the *original*
worktree assumption; Phase A overturned the post-norm part — see Findings.)

Pre-defined routines (as originally written): `BASELINE_VIT` (mean, pre-norm,
GELU, scale 1), `SMOOTH_VIT` (post-norm + scale 10), `SMOOTH_WIDE_VIT` (+
width ×2). `build_config(num_classes, geom)` snaps `encoder_dim` to a
multiple of `lcm(heads, 4)` so `ViTConfig`'s invariants hold.

Fixed geometry (`ViTGeometry`, everything a routine does *not* vary):
`image_size=32, patch=8, dim=192, depth=6, heads=6, mlp_ratio=2.0`.

The backbone is `model.VisionTransformerClassifier` / `ViTConfig`, extended
in this work with the menu fields `pre_norm` (default `True`), `pool`
(`"mean"`/`"cls"`, default `"mean"`), `smooth_activation` (default `True`,
GELU), `final_logit_scale` (default `1.0`, divides the head's logits), plus a
CLS token and a post-norm transformer-block path. Defaults reproduce the
original behavior exactly, so `tests/test_model.py` is unchanged. Sinusoidal
2-D positions + MATH-backend SDPA attention (deterministic) were already
present.

**Corrected `SMOOTH_VIT`** (post-sweep, in `_vit_menu.py`):
`ViTRoutine(name="smooth", pre_norm=True, final_scale=10.0)` — mean pooling,
**pre-norm**, GELU, logits/10. Post-norm increased curvature and is omitted.
`SMOOTH_WIDE_VIT` = smooth + `width_mult=2.0`.

### Deliverables

| file | what |
|---|---|
| `model.py` | edited: ViT smooth-menu fields + CLS token + post-norm path (backward-compatible defaults). |
| `_vit_menu.py` | `ViTRoutine` / `ViTGeometry` menu + `BASELINE_VIT` / `SMOOTH_VIT` / `SMOOTH_WIDE_VIT`. Shared by both runners. |
| `_run_vit_metasmooth_local.py` | Phase A: CIFAR-10 ViT metasmoothness. `A(z)` = deterministic weighted training via functional smooth AdamW; drives `ms.measure_direction`. Stages: head-to-head, per-cluster menu search, per-example h-sweep. |
| `_run_vit_smooth_train_local.py` | Phase B: augmented AdamW training + LR sweep; best test/val accuracy per (routine, lr). |
| `_render_vit_results.py` | JSON → markdown tables (ranked by `Ŝ`) + PNG renderings. |
| `tests/test_vit_metasmooth.py` | 6 CPU tests: training determinism, menu construction (CLS + post-norm paths), `final_scale` divides logits, estimator end-to-end on a toy ViT. |

Tests: `…/torch311/python.exe -m pytest tests/test_vit_metasmooth.py
tests/test_model.py -q` → **12 passed**.

### Methodology / key engineering facts

- **fp32 is mandatory** for Phase A (model + SmoothAdamW in float32,
  `configure_determinism(..., tf32=False)`). fp16/bf16 flip the sign of the
  tiny Definition-1 second difference.
- **Bit-determinism:** only `z` varies across the 3 runs of a probe. Fixed
  `init_seed`/`order_seed`, no augmentation, `cudnn.deterministic=True`, MATH
  SDPA backend. Verified by `test_training_is_deterministic_for_equal_z`
  (max|Δθ| = 0).
- **No BatchNorm:** ViT is LayerNorm-only, so there are no running-stat
  buffers and **no BN recalibration** is needed (the ResNet-9 gotcha doesn't
  apply); the functional path is clean.
- **Weighting** is softmax-based (`weighting.weighted_example_loss`): at
  `z=0` every group mass is uniform ⇒ multipliers = 1 ⇒ exactly standard
  training. This matches the Phase-C curation parameterization, not the
  ResNet-9 `exp(z)` form.
- **Probe directions:** `v ~ N(0, I)` (unnormalized), same `DIR_SEED` reused
  across routines for apples-to-apples; `h=0.05`.

### Run configuration

Run 1: `n_train=4000, n_val=2000, epochs=20, batch=500, fp32, dim=192 depth=6
heads=6 patch=8, SmoothAdamW lr=2e-3 eps=1e-4, h=0.05, dir_seed=1000`. Phase A
finished in **25 min**; Phase B (`n_train=20000`, 30 ep, AdamW) in ~6 min.

## Results

### Phase A — per-cluster ranking by Ŝ (higher = smoother)

| routine | S (Def1) ↓ | Ŝ (Def2) ↑ | f0 | val_acc |
|---|---|---|---|---|
| baseline (mean, pre-norm, GELU, scale 1) | 15.92 | **+0.368** | 1.960 | 0.414 |
| final/1 | 16.64 | +0.330 | 1.960 | 0.414 |
| final/30 | **4.97** | +0.287 | 1.863 | 0.336 |
| depth_x2 | 7.02 | +0.287 | 1.855 | **0.435** |
| cls_pool | 19.37 | +0.285 | 2.155 | 0.397 |
| final/3 | 10.11 | +0.265 | 1.768 | 0.405 |
| **final/10** | **6.68** | **+0.253** | 1.698 | 0.383 |
| post_norm | **24.24** | +0.184 | 1.936 | 0.415 |
| smooth (post-norm + /10) | 9.49 | +0.148 | 1.685 | 0.387 |
| relu | 21.58 | +0.137 | 1.896 | 0.408 |
| width_x2 | 19.68 | +0.135 | 2.198 | 0.407 |
| smooth_wide | 11.69 | +0.022 | 1.662 | 0.404 |

Head-to-head also ran per_example; per-example Ŝ is near its noise floor and
`S` falls sharply with `h` (S: 6.11 → 1.59 → 0.25 for h = 0.05/0.10/0.20),
exactly the ResNet-9 pattern — use per-example `S`, not `Ŝ`.

### Phase B — best test accuracy per routine (AdamW, augmented)

| routine | best test_acc | @ lr | note |
|---|---|---|---|
| baseline | **0.6685** | 1e-3 | |
| smooth | 0.6481 | 1e-3 | higher LR *hurts* (0.59 @ 3e-3, collapses @ 6e-3) |
| smooth_wide | 0.10 (chance) | — | collapsed at every tested LR (only ≥3e-3 swept) |

### Per-example (`_run_vit_per_example_local.py`)

Per-example (one weight per training image, dim = n_train) is the real
curation target. Probed across the menu (3 dirs, h=0.05) plus an h-sweep:

| routine | S @ h=.05 | Ŝ @ h=.05 | S @ h=.1 | S @ h=.2 | S @ h=.4 |
|---|---|---|---|---|---|
| baseline | 3.69 | **+0.78** | 1.56 | 1.18 | 0.27 |
| final/10 | 5.06 | +0.18 | **1.14** | **0.30** | **0.20** |
| final/30 | 3.51 | +0.10 | — | — | — |
| depth_x2 | 13.10 | +0.29 | — | — | — |
| relu | 4.15 | +0.75 | — | — | — |

## Findings

1. **`final_scale` is the dominant lever for curvature `S`** — monotone
   16.6 → 10.1 → 6.7 → 5.0 for scale 1 → 3 → 10 → 30. This *does* replicate
   the paper / ResNet-9 on Definition 1.
2. **But the composite `smooth` routine (post-norm + /10) is not the
   smoothest, because post-norm is a *negative* lever for the ViT.**
   `post_norm` alone has the **worst** `S` (24.2) — pairing it with /10 (the
   original `smooth` routine) lands at `S=9.5`, *worse* than **`final/10`
   alone (`S=6.7`, `Ŝ=+0.253`)**, which is the actual sweet spot. The
   worktree's "post-norm is cleaner for ViT" assumption does **not** hold
   here. **Corrected `SMOOTH_VIT` = mean + pre-norm + GELU + final_scale=10**
   (drop post-norm; the `final/10` routine), optionally with added depth.
3. **`Ŝ` (Def 2) does not cleanly rank ViT routines at this scale** — the
   baseline has the *highest* per-cluster `Ŝ` (+0.368), and `Ŝ` only weakly
   tracks `S`. Prefer **curvature `S`** as the ViT smoothness metric for now;
   push `Ŝ` with more directions / larger `n_train` before trusting its
   ordering.
4. Confirmed-as-expected levers: **mean ≥ CLS** pool (cls `S`=19.4 > baseline
   `S`=15.92), **GELU ≥ ReLU** (relu `S`=21.6), **depth helps** (`S`=7.0, best
   val_acc 0.435). **width hurt** here (`S`=19.7, high `f0`) — likely an
   optimization/LR artifact at this scale, not a true smoothness loss.
5. **Phase B reverses the ResNet-9 LR finding.** With **AdamW** the update is
   already normalized by the gradient's second moment, so the ÷10 logit
   scaling does *not* call for a higher LR — raising LR just destabilizes
   (`smooth` collapses by 6e-3; `smooth_wide` collapsed entirely, though it
   was only swept at ≥3e-3). At matched LR (1e-3) the smooth ViT is *slightly
   below* baseline (0.6481 vs 0.6685), i.e. a mild accuracy cost, not a gain.
6. **Per-example `Ŝ` is not a trustworthy routine-ranking metric — it's
   confounded by update magnitude.** It *inverts* the per-cluster ordering:
   baseline/relu score high (+0.75-0.78) while the genuinely-smoother scaled
   routines score low (+0.10-0.18). Reason: the ÷10 logit scale shrinks
   per-step parameter updates, so under a tiny per-example nudge θ moves less
   and sign-agreement is dominated by optimizer noise. The big-step baseline
   just *looks* coherent. As `h` grows the two converge (baseline Ŝ 0.78 →
   0.41), confirming the small-h value is a magnitude artifact. **Don't rank
   ViT routines by per-example `Ŝ`.**
7. **Per-example curvature `S` is strongly h-dependent; probe at h ≥ 0.1.** At
   the tiny h=0.05 it's in the noise (final/10 ≈ or > baseline). But at **h ≥
   0.1 the final-scale lever clearly lowers `S`** — final/10 < baseline at
   h=0.1 (1.14 vs 1.56), h=0.2 (0.30 vs 1.18), and h=0.4 (0.20 vs 0.27) —
   recovering the per-cluster Definition-1 story. So for per-example
   smoothness, use **curvature `S` at h ≥ 0.1**, not `Ŝ` and not h=0.05.

## Actionable correction

Redefine `SMOOTH_VIT` as **mean + pre-norm + GELU + final_scale=10** (drop
post-norm; the `final/10` sweet spot), as now reflected in `_vit_menu.py`.
This architecture correction (mean pooling, pre-norm, GELU,
`final_scale=10`) fed directly into the CIFAR100-LT compute-scale study — see
[04-cifar100lt-vit-scale-study.md](04-cifar100lt-vit-scale-study.md).

Open items not pursued here: re-sweep Phase B at lower LR for the wide
variant (only tried at ≥3e-3, which collapsed); raise `HEAD_DIRS`/`ABL_DIRS`
and/or `N_TRAIN` if per-cluster `Ŝ` is to be trusted; compare against plain
SGD to check whether the ranking and Phase-B LR behavior are
optimizer-dependent.

## Reproduction

```powershell
# one-time: build the CIFAR-10 cache
C:/Users/luequ/micromamba/envs/torch311/python.exe _build_cifar_cache.py ./data

# Phase A — metasmoothness ranking
C:/Users/luequ/micromamba/envs/torch311/python.exe _run_vit_metasmooth_local.py
#   defaults: N_TRAIN=4000 N_VAL=2000 EPOCHS=20 BATCH=500
#             DIM=192 DEPTH=6 HEADS=6 PATCH=8 MLP_RATIO=2.0
#             H=0.05 HEAD_DIRS=4 PE_DIRS=3 ABL_DIRS=3 MAX_MINUTES=360
#   -> vit_metasmooth_results.json   (written after every bench)

# Phase B — does the smooth menu cost accuracy?
C:/Users/luequ/micromamba/envs/torch311/python.exe _run_vit_smooth_train_local.py
#   defaults: N_TRAIN=20000 N_VAL=5000 EPOCHS=30 ... MAX_MINUTES=240
#   -> vit_train_results.json

# render tables + PNGs
C:/Users/luequ/micromamba/envs/torch311/python.exe _render_vit_results.py
```

Override any knob via env vars (PowerShell: `$env:EPOCHS=15; python ...`).

Smoke test (~10 s, proves the pipeline + ordering):
`N_TRAIN=400 N_VAL=200 EPOCHS=2 DIM=48 DEPTH=2 HEADS=4 HEAD_DIRS=1 PE_DIRS=1 ABL_DIRS=1 WIDE_DIRS=1 OUT=smoke.json python _run_vit_metasmooth_local.py`.

## Gotchas

- **Windows env-var gotcha:** the temperature knob is `TEMPERATURE`, **not**
  `TEMP` (Windows reserves `TEMP` for the temp dir).
- **No BatchNorm / no BN recalibration:** ViT is LayerNorm-only, so the
  ResNet-9 BN-recalibration step is not needed here.
- fp32 + bit-determinism (fixed seeds, no augmentation,
  `cudnn.deterministic=True`, MATH SDPA backend) are required for Phase A;
  fp16/bf16 flip the sign of the Definition-1 second difference.

## Source documents

- `HANDOFF_vit_metasmoothness.md`
- `_vit_menu.py`
- `model.py`
