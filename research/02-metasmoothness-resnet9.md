# Metasmoothness and Metagradient Curation on ResNet-9 / CIFAR-10

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

## Question/Goal

How metasmooth is a CIFAR-10 ResNet-9 training routine — i.e. how amenable is it to metagradient-based data curation — and can it be made smoother without sacrificing accuracy? Then: does per-cluster metagradient descent (MGD) on the resulting smooth routine actually improve held-out performance, using the project's REPLAY engine?

## Setup

**The paper.** "Optimizing ML Training with Metagradient Descent" — Engstrom, Ilyas, Chen, Feldmann, Moses, Mądry, [arXiv:2503.13751](https://arxiv.org/abs/2503.13751), 17 Mar 2025. Defines the **metagradient** `∇_z φ(A(z))` and **REPLAY** (memory-efficient exact metagradients via segment-tree-like reverse traversal). Proposes **metasmoothness** diagnostics so metagradients are only trusted when the training function `f = φ∘A` is smooth in `z`, each computed from **3 deterministic training runs** at `z`, `z+hv`, `z+2hv`:

- **Def 1 (Eq 5) — curvature `S`**: `S_{h,v}(f;z) = |f(z+2hv) − 2 f(z+hv) + f(z)| / h²` (2nd-order finite difference along direction `v`). **Lower = smoother** (β-smooth ⇒ `S ≤ β`).
- **Def 2 (Eq 6) — empirical metasmoothness `Ŝ`** in parameter space: `Ŝ_{h,v}(A;z) = sign(Δ_A(z;v))ᵀ · diag(d/‖d‖₁) · sign(Δ_A(z+hv;v))`, where `Δ_A(z;v)=(θ_h−θ_0)/h`, `Δ_A(z+hv;v)=(θ_2h−θ_h)/h`, `d=|θ_2h−θ_0|` — a range-weighted average sign-agreement of consecutive finite-difference metagradients. `Ŝ ∈ [−1,1]`; **higher (→1) = smoother**. This is the paper's preferred (bounded, interpretable) metric.

**Discovery.** Despite its name, `train_imagenet_vit_tiny.ipynb` is actually a **ResNet-9 / CIFAR-10 cross-entropy classifier** (SGD + cosine + warmup + AMP + wandb, 100-epoch full run with a held-out val split) — essentially the paper's own Sec-3.2/Fig-4 case study. `φ` = held-out cross-entropy. `z` = continuous data weights at `z=0` (paper's count relaxation, Sec 4.1.2): example `i` trained with loss weight `exp(z_i)`, `z=0` ⇒ ordinary training. Both **per-cluster** (one weight per CIFAR class, dim 10) and **per-example** (one weight per training image) were analyzed.

**Engineering facts (apply throughout):**
- **fp32 only** for metasmoothness/finite-difference work. fp16/bf16 are ~2x faster and bit-deterministic but **flip the sign of the tiny Definition-1 second difference** — unfaithful. TF32 is also too imprecise. (`metasmooth.py` supports `amp="fp16"|"bf16"`/`tf32=True` as documented speed options, but all reported numbers are fp32.)
- **BatchNorm running stats must be recalibrated** over the train set before eval (`run_algorithm(..., recalibrate_bn=True)`, default) — otherwise too-few BN updates leave eval-mode logits exploding, an artifact that masquerades as non-smoothness.
- Probe directions `v ~ N(0, I)` (unnormalized), `h=0.05`, with the same `direction_seed` reused across routines for apples-to-apples comparison.

## Method

### The smooth-model "menu"

A frozen dataclass `Routine(pool, bn_before_act, final_scale, width, smooth_act, name)` (plus, added on the VM, `norm: "bn"|"gn"`, `gn_groups`) parameterizes the design-space menu from Engstrom et al.'s Remark 4:

- `BASELINE_ROUTINE`: `pool="max"`, `bn_before_act=True`, `final_scale=1.0`, `width=1.0` (mirrors the original notebook's ResNet-9 — note it **already** uses BN-before-ReLU).
- `SMOOTH_ROUTINE`: `pool="avg"`, `bn_before_act=True`, `final_scale=10.0`, `width=1.0` (capacity-matched; only architectural levers, width unchanged).
- `SMOOTH_WIDE_ROUTINE`: `SMOOTH_ROUTINE` + `width=2.0`.
- `SMOOTH_GN_ROUTINE` (added in the VM phase): smooth routine + GroupNorm. GroupNorm has **0 buffers**, so the functional engine (which holds buffers fixed) gives clean exact metagradients through training — this is the **Phase C curation model**.

**`final_scale ÷10` (scaling down the last layer's output) is the dominant smoothness lever** — alone it collapses curvature `S` from 53 to 2.3 in the original benchmark (§ "Initial findings" below). Average pooling helps modestly/noisily; combined with BN-before-activation it is complementary for `Ŝ`.

### Phase A — scaled metasmoothness measurement (VM)

Two VM runs at larger scale than the laptop benchmark, same routine definitions / `h=0.05` / `dir_seed=1000` for apples-to-apples comparison:
- `metasmooth_vm_results.json` — n_train=6000, 18 epochs, 6/4 directions.
- `metasmooth_vm_deep_results.json` — n_train=8000, 20 epochs, **12 directions**, plus `smooth_gn` (the tighter, more reliable run; 218 minutes on one A40).

**Cost note**: fp32 + `cudnn.deterministic` is the bottleneck on A40 (baseline 6000×18 = 20.6 s/run; ~3.2× for width-2; one probe = 3 runs). The metric is **relative** — scaling **directions** is the cheap reliability lever, not `n_train`. Do not switch to TF32/fp16/bf16.

### Phase B — retrofit smooth routine into real training (VM)

25-epoch augmented training (full 45k/5k/10k CIFAR-10 splits), AMP, with an LR sweep over `(routine, lr)` pairs. No fp32 constraint here (real training, AMP is fine).

### Phase C — per-cluster metagradient curation via REPLAY (VM)

**Engine reuse**: the repo's differentiable engine is already classification-based (the WIP converted MAE → classifier). `weighted_inner_step` (`functional_train.py`) does one functional Smooth-AdamW step with the group-softmax reparameterization (`weighting.py`); `metagrad.train_unrolled` is store-all; `recursive_replay.recursive_replay_state` is REPLAY. The curation model is `metasmooth.ResNet9(SMOOTH_GN_ROUTINE)` (GroupNorm → 0 buffers). `z` = per-cluster logits (`ℝ¹⁰`); weights = `10·softmax(z)`; `z=0` ⇒ uniform training.

**Memory**: store-all OOMs past ~50 inner steps on the A40. **Use REPLAY** (`recursive_replay_state`, `branching_factor=4`) → `O(k log_k T)` memory → `T=192` inner steps fits easily. REPLAY does **not** support higher-order diff (raises if grad is enabled), so the first-order metagradient is computed with `autograd.grad(obj, z)` and a plain Adam step on `z` — sufficient for MGD.

Three experiments probe where per-cluster curation helps: balanced data, label-noise (classes 3,5 randomized), and distribution shift (objective = vehicle classes 0,1,8,9).

## Results

### Initial findings (laptop benchmark, head-to-head)

Config: `n_train=3000, n_val=1000, epochs=16, batch_size=500, lr=0.08, amp=off, h=0.05, dir_seed=1000`. `f0` = held-out cross-entropy at `z=0` (same trained model for both metaparameterizations, since `z=0` ⇒ all weights 1).

| metaparameter z | routine | val loss f0 | S (Def 1) ↓ | Ŝ (Def 2) ↑ |
|---|---|---|---|---|
| per-cluster (class) | baseline | 2.078 | 53.06 ± 12.85 | +0.092 ± 0.130 |
| per-cluster (class) | **smooth** | **1.425** | **4.79 ± 2.32** | **+0.511 ± 0.049** |
| per-example | baseline | 2.078 | 43.41 ± 9.27 | +0.135 ± 0.064 |
| per-example | **smooth** | **1.425** | **7.69 ± 2.74** | +0.148 ± 0.005 |

Per-cluster ablation (one knob at a time):

| routine | val loss f0 | S (Def 1) ↓ | Ŝ (Def 2) ↑ |
|---|---|---|---|
| baseline (max pool, full-magnitude logits) | 2.078 | 53.06 | +0.092 |
| average pooling only | 1.21 | 30.23 | +0.073 |
| **final-layer output ÷10 only** | **0.99** | **2.29** | +0.123 |
| **smooth (avg + ÷10 + BN-before)** | 1.42 | 4.79 | **+0.511** |
| width_x2 only | — | (≈66, 1-dir probe; not finished) | — |
| bn_after_act (non-smooth direction) | — | (not finished) | — |

### Phase A — scaled per-cluster ranking (deep run, ranked by `Ŝ`)

| routine | acc | S (Def1) ↓ | Ŝ (Def2) ↑ |
|---|---|---|---|
| avg+f30 | 0.601 | 4.3 | **+0.485** |
| smooth+gelu | 0.747 | 4.2 | +0.315 |
| final/30 | 0.749 | 3.1 | +0.300 |
| smooth_wide | 0.764 | 3.5 | +0.210 |
| smooth | 0.745 | 7.8 | +0.207 ± 0.068 |
| **smooth_gn** (GroupNorm) | 0.443 | 47.9 | **+0.196** |
| final/10 | 0.759 | 5.0 | +0.097 |
| avg_pool | 0.759 | 7.7 | +0.068 |
| gelu | 0.686 | 23.4 | +0.065 |
| width_x2 | 0.508 | 101.7 | +0.054 |
| final/3 | 0.774 | 11.7 | +0.027 |
| baseline | 0.686 | 34.1 | +0.025 |
| bn_after_act | 0.544 | 37.7 | +0.016 |

Per-example smooth (noise floor lifts with `h`): `Ŝ` +0.036 → +0.102 → +0.122 as `h` 0.05 → 0.1 → 0.2 (`S` 3.8 → 1.8 → 0.3). Baseline per-example `Ŝ ≈ 0`.

### Phase B — retrofit + accuracy (`train_smooth_vm_results.json`)

25-epoch augmented training, best test accuracy per routine across the LR sweep:

| routine | best test acc | @ lr |
|---|---|---|
| baseline | 91.2% | 0.2 |
| smooth | **93.1%** | 0.6 |
| smooth_wide | **93.7%** | 0.6 |

The ÷10 logits divide the loss gradient by ~10, so the smooth routine wants a **higher LR** (`0.1→91.2`, `0.6→93.1`). With LR retuned, the smooth routine **beats** the baseline — the "accuracy cost" implied by the architectural change is in fact a gain once the LR is retuned.

### Phase C — per-cluster metagradient curation

| setup | held-out CE | result | file |
|---|---|---|---|
| balanced | 1.813 → 1.831 | no exploitable signal (uniform ≈ optimal) | `phase_c_balanced_results.json` |
| label-noise (cls 3,5 randomized) | 2.153 → 2.227 | **degenerate**: held-out *contains* the corrupted classes → MGD boosts their logit bias (cat↓0.58 but dog↑2.15) instead of removing noise | `phase_c_corrupt_results.json` |
| **distribution shift** (objective = vehicles 0,1,8,9) | **1.860 → 1.709 (−8.1%)**; val 1.834 → 1.681 (−8.3%) | **MGD measurably cuts held-out CE and it generalizes** | `phase_c_shift_results.json` |

Winning config: `METHOD=replay N_POOL=4000 N_OBJ=3000 N_VAL=3000 INNER_EPOCHS=12 BATCH=250 INNER_LR=0.02 META_STEPS=20 META_LR=0.05 TARGET_CLASSES=0,1,8,9` (T=192 inner steps; the inner trajectory is fixed and only `z` varies, so the CE drop is purely attributable to the learned weighting).

## Findings

1. **Baseline is not metasmooth** (`Ŝ ≈ 0.09–0.14`, `S ≈ 43–53` at small scale; `Ŝ ≈ 0.025`, `S ≈ 34.1` at scale). Metagradients on it would be unreliable.
2. **The paper's menu fixes it**: at small scale, curvature drops ~6–11x and per-cluster `Ŝ` rises `+0.09 → +0.51`. At scale, the smooth menu's ranking is stable and reproducible.
3. **`final_scale` is the dominant `Ŝ` lever** (monotone `f3 → f10 → f30`); `avg+f30` maximizes `Ŝ` (+0.485) but at an accuracy cost (acc 0.601) — a real **smoothness↔accuracy frontier**. `smooth`/`smooth_wide` are the production sweet spot (high `Ŝ` and high accuracy: 0.745/0.764).
4. **Non-smooth directions verify as non-smooth at scale**: `bn_after_act`, `gelu`-alone, `width`-alone all sit near `Ŝ ≈ 0` (0.016, 0.065, 0.054).
5. **Smoothness tracks optimizability**: the smooth routine also reaches lower held-out loss at small scale (`2.08 → 1.42`; final/10 alone `→ 0.99`).
6. **per-cluster vs per-example are two metaparameterizations of the same routine, not competing techniques** — per-cluster is the restriction of per-example to the subspace where weights are constant within a class. They **agree on curvature `S`** (both diagnose baseline as non-smooth and both improve ~6–11x under the smooth routine at small scale) but **diverge sharply on `Ŝ`**: per-cluster `+0.09 → +0.51` vs per-example `+0.14 → +0.15` (flat). This is a signal-to-noise effect: `Ŝ` averages sign-agreement over ~6.5M parameters, and a per-cluster nudge reweights a whole class (large, structured, sign-stable response) whereas a per-example nudge spreads `h·v` over thousands of images (~1% each → minuscule, incoherent response → `Ŝ ≈ 0.15` noise floor for any routine). Practical implication: per-cluster is the better smoothness *probe* and a coarse curation knob; per-example is the real curation *target* but its `Ŝ` is pinned at the noise floor at this scale — use curvature `S` (not `Ŝ`) for per-example, and per-example `Ŝ` does improve with larger `h` (0.036→0.102→0.122 as h: 0.05→0.1→0.2).
7. **GroupNorm stays metasmooth**: `smooth_gn` `Ŝ=+0.196 ≈ smooth`'s `+0.207` — re-validated before trusting Phase C metagradients. Its curvature `S` is high (47.9) and its accuracy is low (0.443) under the benchmark's SGD/lr=0.08 (GN wants a different optimizer); Phase C trains it with functional AdamW and that's fine. The `S`-vs-`Ŝ` divergence for GN is noted as worth a closer look but unresolved.
8. **Phase B**: smooth routine beats baseline once LR is retuned upward (93.1%/93.7% vs 91.2% at 25 epochs) — the architectural smoothness changes are a net accuracy win, not a tradeoff, given the LR retune.
9. **Phase C**: per-cluster MGD via REPLAY on the smooth GroupNorm routine cuts held-out CE by 8.1% (val 8.3%) under a distribution-shift objective, and this generalizes from objective set to validation set. Balanced data gives no signal (uniform already near-optimal); label-noise overlapping the held-out classes produces a degenerate bias-boosting solution rather than denoising. The learned per-class weights for the winning shift run are a non-trivial reweighting (down-weights truck, up-weights automobile/ship and some animal classes) reflecting real confusion structure rather than naive distribution-matching — flagged as a finding needing more meta-steps / lower `META_LR` / longer inner training for a cleaner weight story.

## Reproduction

```bash
# from the repo root, in the torch311 env (laptop benchmark):
C:/Users/luequ/micromamba/envs/torch311/python.exe -m experiments.cifar10.metasmooth_benchmark   # writes artifacts/metasmooth_results.json
C:/Users/luequ/micromamba/envs/torch311/python.exe -m experiments.cifar10.render_metasmooth      # writes artifacts/metasmooth_results.png + tables
C:/Users/luequ/micromamba/envs/torch311/python.exe -m pytest tests/test_metasmooth.py -q   # 10 passed
```

VM scaling: raise `N_TRAIN`, `TRAIN.epochs`, `HEAD_DIRS`/`ABL_DIRS`/`PE_DIRS`/`WIDE_DIRS` etc. in `experiments/cifar10/metasmooth_vm.py` (env-var driven); keep `amp="off"`. Render with `python -m experiments.cifar10.render_metasmooth_vm artifacts/metasmooth_vm_deep_results.json`. CIFAR data on the VM must come from an HF-parquet `.npz` cache (`python -m tools.build_cifar_cache`) — the torchvision mirror is throttled to ~30 KB/s.

Phase B: `experiments/cifar10/metasmooth_train.py` (routine, lr) grid over real augmented training. Phase C: `experiments/cifar10/metasmooth_mgd.py` with env vars `METHOD=replay|store_all N_POOL N_OBJ N_VAL INNER_EPOCHS BATCH INNER_LR META_STEPS META_LR BRANCHING`, plus `CORRUPT_CLASSES/CORRUPT_FRAC` (label-noise demo) and `TARGET_CLASSES` (distribution-shift demo); `experiments/cifar10/metasmooth_smoke.py` is a CPU/laptop check that the engine differentiates held-out CE w.r.t. per-cluster logits (no CIFAR needed).

Artifacts on the laptop (under `artifacts/`): `metasmooth_vm_results.json`, `metasmooth_vm_deep_results.json`, `metasmooth_vm_ranking_deep.png`, `metasmooth_vm_tradeoff_deep.png`, `train_smooth_vm_results.json`, `phase_c_balanced_results.json`, `phase_c_corrupt_results.json`, `phase_c_shift_results.json`.

## Gotchas

- **fp32 only** for metasmoothness/finite-difference work — TF32/fp16/bf16 flip the sign of the Def-1 second difference and invalidate exact metagradients.
- **BatchNorm running-stat buffers are held fixed by the functional engine** → exact metagradients through training require **GroupNorm** (`SMOOTH_GN_ROUTINE`, 0 buffers). BN must also be recalibrated before eval in the finite-difference probes (`recalibrate_bn=True`), or eval-mode logits can explode and masquerade as non-smoothness.
- CIFAR mirror (cs.toronto.edu) is throttled on the VM boxes — pre-build an `.npz` cache from the HuggingFace CDN.
- Store-all (`train_unrolled`) OOMs past ~50 inner steps on an A40 — use REPLAY for `T=192`.
- `recursive_replay_state` raises if higher-order autograd is attempted; Phase C uses first-order `autograd.grad(obj, z)` + a plain Adam step on `z`.
- Phase-C demo design matters: balanced data gives per-cluster MGD nothing to do; label-noise on classes present in the held-out objective creates a degenerate bias-boosting optimum; distribution shift (held-out = a target subset of classes) is the clean demo.
- All Phase A/B/C code is **uncommitted** in the source repo as of this writing.

## Source documents

- `HANDOFF_metasmoothness.md` (verbatim in [archive/original-handoffs.md](archive/original-handoffs.md))
- `HANDOFF_vm_phaseABC.md` (verbatim in [archive/original-handoffs.md](archive/original-handoffs.md))
