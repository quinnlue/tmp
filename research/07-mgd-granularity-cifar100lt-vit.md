# MGD Granularity on CIFAR100-LT / ViT-Tiny: Per-class vs Per-cluster vs Per-example

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

This is the capstone comparison: it runs all three weighting granularities on a
*trainable* transformer (not a frozen linear head), against a balanced long-tail
objective, with the curation anchored to a fully validated training recipe. It
builds directly on the backbone/recipe chosen in
[04-cifar100lt-vit-scale-study.md](04-cifar100lt-vit-scale-study.md) and the
engine described in [01-metagradient-engine.md](01-metagradient-engine.md).

## Question/Goal

On CIFAR100-LT (`r-100`), under the corrected-smooth ViT-Tiny 75-epoch recipe,
does metagradient data curation improve balanced final-test accuracy over the
uniform baseline — and which weighting granularity (per-class, per-cluster,
per-example) is best?

## Setup

- **Dataset:** `tomas-gajarsky/cifar100-lt`, config `r-100`. 10,847-image
  long-tailed training pool across 100 classes. Balanced official test split
  divided into 50 validation + 50 final-test images per class. Seed 0.
- **Model:** corrected-smooth ViT-Tiny — patch 4, embed dim 192, depth 12,
  3 heads, MLP ratio 4, mean pooling, pre-norm, GELU, final logits ÷10.
  5,367,460 parameters.
- **Uniform baseline = the verified recipe:** AdamW, peak LR 1e-3, weight decay
  0.05, 75 epochs, batch 256, 10% linear warmup + cosine decay, bf16, random
  horizontal flip + reflection-padded random crop. Reproduces the scale study
  exactly: best epoch 58, balanced val 24.76%, balanced test **23.42%**, tiers
  many/medium/few = **45.77 / 19.49 / 1.93**.

## Method — two-stage curation

The exact metagradient through a full 75-epoch ViT-Tiny REPLAY is expensive, so
the search horizon is decoupled from the evaluation recipe (mirrors the
ImageNet-LT driver's external-reeval pattern in
[06-mgd-granularity-imagenetlt.md](06-mgd-granularity-imagenetlt.md)).

**Stage 0 — baseline + cluster features.** Train the uniform recipe once. It is
both the uniform comparison point and the source of penultimate ViT-Tiny
embeddings (mean-pooled pre-head tokens, 192-d) for the 10,847-image pool.

**Stage 1 — MGD search (per granularity, fp32, deterministic REPLAY).**
- Inner training: the 10,847-image pool with a fixed seeded batch schedule,
  **15 search-epochs** (~1050 inner steps, batch 256), functional Smooth AdamW
  with the recipe-shaped cosine+warmup LR (peak 1e-3, weight decay 0.05).
  Augmentation disabled (REPLAY requires a bit-identical image stream across
  replays; only `z` varies).
- Objective φ: mean cross-entropy over **25/class** held out from the balanced
  validation split (a balanced set, so mean CE = balanced CE).
- Meta-validation (step selection): the other **25/class**. Final test (50/class)
  stays fully untouched.
- Metaparameter `z` is the deviation from the base distribution
  (`effective_group_logits = z + T·log(base)`), so `z = 0` is ordinary training
  for every granularity. Granularities differ only in the `group_ids` map:
  - **per_class:** 100 groups, base = long-tail class frequencies.
  - **per_cluster:** 128 groups, label-agnostic 2-level hierarchical k-means on
    the Stage-0 embeddings, base = cluster frequencies.
  - **per_example:** 10,847 groups, base = uniform 1/n.
- Outer loop: **15 meta-steps**, exact REPLAY metagradient (`branching_factor=32`),
  fixed signed-KL update with target KL 1e-3. Selection: minimum balanced
  meta-validation CE.

**Stage 2 — full-recipe re-evaluation.** Convert each method's selected group
masses to per-example loss multipliers and retrain the **full 75-epoch recipe**
(bf16, augmented, seed 0) with the weighted cross-entropy. Report balanced
val/test + tiers. Uniform = Stage 0.

Precision note: the **search is fp32/deterministic** (exact metagradients require
it); the **baseline and all reevals are bf16** (the recipe). The learned weights
are scalar per-example loss multipliers, so carrying fp32-derived weights into a
bf16 retrain is sound, and the comparison is apples-to-apples (all four final
runs use identical precision and recipe).

## Results

Balanced final-test accuracy under the identical 75-epoch recipe, differing only
in the learned per-example data weights:

| Method | Val bal | Test bal | Many | Medium | Few | Best epoch | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform (baseline) | 24.76 | 23.42 | 45.77 | 19.49 | 1.93 | 58 | — |
| per_class | 23.60 | 23.58 | 46.00 | 19.49 | 2.20 | 62 | +0.16 |
| **per_cluster** | **24.16** | **24.74** | **48.74** | **20.34** | 1.87 | 58 | **+1.32** |
| per_example | 24.76 | 23.42 | 45.77 | 19.49 | 1.93 | 58 | +0.00 |

MGD search summary (balanced meta-validation CE; 15 search-epochs × 15 meta-steps):

| Granularity | Groups | Selected step | Meta-val CE (start → selected) | Final entropy | Final ESS |
|---|---:|---:|---|---:|---:|
| per_class | 100 | 8/15 | 4.4034 → 4.3868 | 3.999 | 41.3 |
| per_cluster | 128 | 7/15 | 4.4034 → 4.3905 | 4.273 | 57.2 |
| per_example | 10847 | 0/15 | 4.4035 → 4.4035 | 9.292 | 10847.0 |

## Findings

1. **Per-cluster wins (+1.32 balanced-test points, 23.42% → 24.74%).** Gains
   concentrate in many- and medium-shot (+2.97 / +0.85); few-shot is flat
   (−0.06). 128 label-agnostic clusters are coarse enough to yield a *coherent*
   metagradient, yet finer than the 100 class knobs.
2. **Per-example collapses to the baseline.** Its 10,847-dim metaparameter never
   improved balanced meta-validation CE, so the min-CE selection rule kept
   step 0 (uniform). Its reeval is consequently **bit-identical** to the baseline
   (23.42%, best epoch 58) — a clean confirmation that "no curation" maps exactly
   back to the baseline and that the bf16 reeval harness is deterministic.
3. **Per-class barely moves (+0.16).** Reweighting only 100 class knobs against an
   already class-balanced objective has little headroom; ESS falls to ~41 of 100.
4. **Consistency check.** The ordering from the independent full-recipe reevals
   (per_cluster > per_class > per_example) matches the search trajectories
   (per_cluster/per_class found small meta-val CE improvements at interior steps;
   per_example found none).

## Gotchas

- **Determinism vs CUDA ops.** The search uses `configure_replay_determinism(tf32=False)`,
  which enables `use_deterministic_algorithms(True)`. `scatter_add`/`bincount` on
  CUDA raise under it, so mass summaries, class base masses, and clustering run on
  CPU. `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set before CUDA initializes (done
  at module import). Ordinary phases call `torch.use_deterministic_algorithms(False)`.
- **REPLAY cost is backward-dominated and largely branching-independent.** At
  25 search-epochs a meta-step was ~11 min (forward ~102 s + exact backward
  ~9 min); raising `branching_factor` cuts recomputation but not the per-step
  create-graph autograd, so the practical lever is the search horizon. The reported
  run used 15 search-epochs × 15 meta-steps (~5 hr for all three granularities + reevals).
- **Two-stage caveat.** Per-example's null result is "no signal at this 15-epoch
  search budget," not "per-example curation is impossible" — a longer search
  horizon, more meta-steps, or averaging might move it off the noise floor (the
  ResNet-9 work found per-example signal only lifts above noise at larger `h`/scale).

## Reproduction

Code (main checkout): `_run_cifar100_lt_vit_mgd.py` (`--phase {baseline,search,reeval,all}`),
`_cluster_basis.py` (label-agnostic hierarchical k-means), `_render_cifar100_lt_vit_mgd.py`,
`tests/test_cifar100_lt_vit_mgd.py`. Reuses the scale-sweep recipe
(`_run_cifar100_lt_vit_scale_sweep.py`) and the engine (`recursive_replay.py`,
`functional_train.py`, `weighting.py`, `determinism.py`).

```bash
# baseline + embeddings, then search all granularities, then full-recipe reeval
python _run_cifar100_lt_vit_mgd.py --phase baseline --epochs 75
python _run_cifar100_lt_vit_mgd.py --phase search \
  --granularities per_class,per_cluster,per_example \
  --search-epochs 15 --meta-steps 15 --num-clusters 128 --branching-factor 32
python _run_cifar100_lt_vit_mgd.py --phase reeval \
  --granularities per_class,per_cluster,per_example --epochs 75
python _render_cifar100_lt_vit_mgd.py   # the 4-way table
```

Artifacts: `artifacts/cifar100_lt_vit_mgd/{baseline,search_<g>,reeval_<g>}.json`.

## Source documents

- `artifacts/cifar100_lt_vit_mgd/baseline.json`, `reeval_{per_class,per_cluster,per_example}.json`, `search_{per_class,per_cluster,per_example}.json`
- `CIFAR100_LT_VIT_SCALE_STUDY.md` (backbone/recipe selection; verbatim in [archive/original-handoffs.md](archive/original-handoffs.md))
- `_run_cifar100_lt_vit_mgd.py`, `_cluster_basis.py`, `_render_cifar100_lt_vit_mgd.py`
