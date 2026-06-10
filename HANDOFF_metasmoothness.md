# Handoff: Metasmoothness of `train_imagenet_vit_tiny.ipynb`

Paste this whole file into a new chat to continue the work. It is self-contained:
background, the paper, what was built, full results, the per-cluster vs
per-example analysis, the API, how to reproduce/scale, and open items.

---

## 0. One-paragraph summary

We measured how *metasmooth* a CIFAR-10 ResNet-9 training routine is — i.e. how
amenable it is to metagradient-based data curation — and then maximized it using
the design "menu" from Engstrom et al. (2025). Metasmoothness is a finite-difference
diagnostic (3 deterministic training runs per probe; no metagradients needed). The
baseline routine is **not** metasmooth; the paper's menu (average pooling +
scaling the final-layer logits down ~10x + BatchNorm-before-activation) makes it
dramatically smoother (curvature `S` down ~6–11x; empirical metasmoothness `Ŝ` up
`+0.09 → +0.51` for per-cluster data weights) **and** trains to a lower held-out
loss. All code is in `metasmooth.py` + a new section in the notebook.

---

## 1. Project background

- Repo: `C:\ml\dataset-curation\meta-grad-descent-w-clustering` (a.k.a.
  "meta-grad-descent-w-clustering"). A **PyTorch** (functional, not JAX)
  metagradient **data-curation** experiment: treat the training dataset /
  weighting as a continuous metaparameter and optimize it with metagradients.
- Has a working metagradient engine: `functional_train.py` (functional smooth
  AdamW — eps kept *inside* the sqrt for smooth backward-over-backward),
  `metagrad.py`, `paper_mgd.py` (paper-faithful count relaxation + Algorithm 1),
  `replay.py` / `recursive_replay.py` (REPLAY = segmented `torch.utils.checkpoint`).
  This engine was originally built around an **MAE-ViT** model with masked
  *reconstruction* loss; it is **not used** by the metasmoothness work below.
- Workflow: **dev-on-laptop / train-on-VM**. Heavy runs go to a VM/L4; the laptop
  (RTX 4050) is for development and small demos.
- Environment: micromamba env **`torch311`**:
  `C:/Users/luequ/micromamba/envs/torch311/python.exe` — torch 2.8.0+cu128,
  torchvision 0.23.0+cu128, CUDA on an **RTX 4050 Laptop GPU**.
- Data: CIFAR-10 already downloaded at
  `C:\ml\dataset-curation\meta-grad-descent-w-clustering\data`.

### Important repo-state caveat
The most-recent version of the work lives as **uncommitted WIP in the main
checkout** (not committed to git). That WIP converted the repo from MAE/recon to a
classifier: `mae.py`→`model.py` (`VisionTransformerClassifier`, mean-pooled),
`tests/test_mae.py`→`tests/test_model.py`,
`train_tiny_mae.ipynb`→`train_tiny_classifier.ipynb`, plus a new
`train_imagenet_vit_tiny.ipynb`. The metasmoothness work was done on a worktree and
the deliverables were **copied back into the main checkout**.

---

## 2. The paper

**"Optimizing ML Training with Metagradient Descent"** — Engstrom, Ilyas, Chen,
Feldmann, Moses, Mądry (MIT/Stanford/UIUC), [arXiv:2503.13751](https://arxiv.org/abs/2503.13751), 17 Mar 2025.

- **Metagradient**: `∇_z φ(A(z))` — gradient of a downstream metric `φ` through the
  *entire training process* `A`, w.r.t. a metaparameter `z` (e.g. per-datapoint
  weights). Enables gradient-based dataset selection / poisoning / HP search.
- **REPLAY**: memory-efficient exact metagradients (`O(k log_k T)` space) via a
  segment-tree-like reverse traversal of optimizer states.
- **Metasmoothness** (Sec 3): metagradients are only useful if the training
  function `f = φ∘A` is smooth in `z`. Two cheap, metagradient-free diagnostics,
  each needing only **3 deterministic training runs** at `z`, `z+hv`, `z+2hv`:
  - **Def 1 (Eq 5) — curvature**: `S_{h,v}(f;z) = |f(z+2hv) − 2 f(z+hv) + f(z)| / h²`
    (a 2nd-order finite difference along `v`). **Lower = smoother** (β-smooth → `S ≤ β`).
  - **Def 2 (Eq 6) — empirical metasmoothness** in *parameter* space:
    `Ŝ_{h,v}(A;z) = sign(Δ_A(z;v))ᵀ · diag(d/‖d‖₁) · sign(Δ_A(z+hv;v))`, where
    `Δ_A(z;v)=(θ_h−θ_0)/h`, `Δ_A(z+hv;v)=(θ_2h−θ_h)/h`, `d=|θ_2h−θ_0|`.
    A range-weighted average sign-agreement of consecutive finite-difference
    metagradients. `Ŝ ∈ [−1,1]`; **higher (→1) = smoother**. This is the paper's
    preferred metric (bounded, interpretable).
- **Smooth model training** (Sec 3.2, Remark 4): make a routine metasmooth by
  exploring a menu of design changes and keeping those that raise `Ŝ`. For
  ResNet/CIFAR: **average instead of max pooling**, **BatchNorm before activation**,
  **scale the last layer's output down ~10x**, **wider networks**, and a smooth
  optimizer (eps inside the sqrt). Smoothness also correlates with optimizability.

---

## 3. The task (as given)

> In `train_imagenet_vit_tiny.ipynb`: check how metasmooth it is and modify it to be
> as metasmooth as possible. Benchmark first, then implement the paper's metasmooth
> objective. Goal: maximize metasmoothness. Use **just the classification loss** as
> the objective (on the held-out set). User clarification: analyze **both per-cluster
> and per-data(example) weights**.

### Key discovery
Despite its name, `train_imagenet_vit_tiny.ipynb` is **not** a ViT or an MAE — it is
a **ResNet-9 / CIFAR-10 cross-entropy classifier** (SGD + cosine + warmup + AMP +
wandb, 100-epoch full run, with a held-out val split). This is essentially the
paper's own Sec-3.2 / Fig-4 metasmoothness case study. So:
- `φ` = held-out **cross-entropy** (a disjoint CIFAR-10 split). ✓ matches the ask.
- `z` = continuous **data weights at z=0** (paper's count relaxation, Sec 4.1.2):
  example `i` trained with loss weight `exp(z_i)`; `z=0` ⇒ ordinary training.
  Analyzed for **per-cluster** (one weight per CIFAR class, dim 10) and
  **per-example** (one weight per training image, dim = `n_train`).
- Metasmoothness is finite-difference, so the repo's REPLAY/functional engine is
  **not needed** here — just 3 ordinary (deterministic) trainings per probe.

---

## 4. What was built (deliverables)

In the **main checkout** `C:\ml\dataset-curation\meta-grad-descent-w-clustering\`
(and on worktree branch `claude/optimistic-goldberg-0ac9e8`):

| file | what |
|---|---|
| `metasmooth.py` | Core module: deterministic weighted-CE ResNet-9 `A(z)`, the routine "menu", and both estimators (Def 1 `S`, Def 2 `Ŝ`). Standalone — imports neither `mae.py` nor `model.py`. |
| `tests/test_metasmooth.py` | 10 passing unit tests (analytic training functions; no GPU). |
| `train_imagenet_vit_tiny.ipynb` | Original 10 cells **unchanged** + 11 new cells: a "Metasmoothness" section (intro, benchmark baseline, apply menu, ablation, plot, results + conclusions with real numbers and the PNG). Cells are runnable but **not executed** (recompute ≈30 min on a laptop GPU). |
| `_run_metasmooth_benchmark.py` | Reproducible benchmark runner; writes `metasmooth_results.json`. |
| `_render_results.py` | Renders the JSON → markdown tables + `metasmooth_results.png`. |
| `metasmooth_results.json` / `metasmooth_results.png` | The measured results below. |

(`_add_section.py`, `_add_conclusions.py` are one-shot notebook builders, present on
the worktree only.)

Tests: `C:/Users/luequ/micromamba/envs/torch311/python.exe -m pytest tests/test_metasmooth.py -q` → 10 passed.

---

## 5. Methodology / key engineering facts (don't relearn these)

- **Bit-determinism is required.** The only thing that may differ across the 3 runs
  of a probe is `z`. We fix seeds, data order, disable random augmentation, set
  `cudnn.deterministic=True`, and pool with atomics-free reshape+`amax`/`mean`.
  Verified: `A(z)` twice with equal `z` → identical θ (max|Δ|=0) and val loss.
- **fp32 is mandatory** (`amp="off"`). fp16/bf16 are ~2x faster and bit-deterministic
  but they **flip the sign of the tiny Definition-1 second difference** — unfaithful.
  (TF32 is also too imprecise.) `metasmooth.py` supports `amp="fp16"|"bf16"` and
  `tf32=True` as documented speed options, but the reported numbers are fp32.
- **BatchNorm running stats must be recalibrated** over the train set before eval.
  After short training, too-few BN updates leave running stats far from the
  activation statistics and eval-mode logits explode — an artifact that *masquerades
  as non-smoothness*. `run_algorithm` does this when `recalibrate_bn=True` (default).
- **Weighting**: per-example loss multiplier `exp(z_{coord(i)})`, mean-reduced
  (`(weights * per_example_ce).mean()`), so `z=0` is exactly standard training.
- **Probe directions**: `v ~ N(0, I)` (unnormalized), same `direction_seed` reused
  across routines so comparisons are apples-to-apples. `h=0.05`.
- **The routine "menu"** is a frozen dataclass `Routine(pool, bn_before_act,
  final_scale, width, smooth_act, name)`:
  - `BASELINE_ROUTINE` = pool="max", bn_before_act=True, final_scale=1.0, width=1.0
    (mirrors the notebook's ResNet-9 — note it **already** uses BN-before-ReLU).
  - `SMOOTH_ROUTINE` = pool="avg", bn_before_act=True, final_scale=10.0, width=1.0
    (capacity-matched: only architectural levers, width unchanged).
  - `SMOOTH_WIDE_ROUTINE` = SMOOTH + width=2.0.
- **Benchmark scale used (laptop demo)**: `n_train=3000`, `n_val=1000`, `epochs=16`,
  `batch_size=500`, `lr=0.08` (SGD+momentum+cosine+warmup), fp32, `h=0.05`,
  3 probe directions head-to-head (2 for the ablation). The metric is **relative** —
  scale up `n_train`/`epochs`/`num_directions` on the VM; the ordering is what matters.
- **Caveat**: the laptop GPU thermal-throttled after ~40 min of continuous fp32
  ResNet-9 training, so the `width_x2` and `bn_after_act` ablation knobs were **not
  finished** (only a 1-direction probe of `width_x2` ≈ baseline, `S≈66`). They remain
  as runnable cells.

---

## 6. Results

Config: `n_train=3000, n_val=1000, epochs=16, batch_size=500, lr=0.08, amp=off, h=0.05, dir_seed=1000`.
`f0` = held-out cross-entropy at `z=0` (same trained model for both metaparams, since
`z=0` ⇒ all weights 1 — a sanity check that only the probe directions differ).

### 6.1 Baseline vs the paper's menu (head-to-head)

| metaparameter z | routine | val loss f0 | S (Def 1) ↓ | Ŝ (Def 2) ↑ |
|---|---|---|---|---|
| per-cluster (class) | baseline | 2.078 | 53.06 ± 12.85 | +0.092 ± 0.130 |
| per-cluster (class) | **smooth** | **1.425** | **4.79 ± 2.32** | **+0.511 ± 0.049** |
| per-example | baseline | 2.078 | 43.41 ± 9.27 | +0.135 ± 0.064 |
| per-example | **smooth** | **1.425** | **7.69 ± 2.74** | +0.148 ± 0.005 |

### 6.2 Per-cluster ablation (one knob at a time)

| routine | val loss f0 | S (Def 1) ↓ | Ŝ (Def 2) ↑ |
|---|---|---|---|
| baseline (max pool, full-magnitude logits) | 2.078 | 53.06 | +0.092 |
| average pooling only | 1.21 | 30.23 | +0.073 |
| **final-layer output ÷10 only** | **0.99** | **2.29** | +0.123 |
| **smooth (avg + ÷10 + BN-before)** | 1.42 | 4.79 | **+0.511** |
| width_x2 only | — | (≈66, 1-dir probe; not finished) | — |
| bn_after_act (non-smooth direction) | — | (not finished) | — |

### 6.3 Takeaways
1. The **baseline routine is not metasmooth** (`Ŝ≈0.09–0.14`, `S≈43–53`).
   Metagradients on it would be unreliable.
2. The **paper's menu fixes it**: curvature down ~6–11x; per-cluster `Ŝ` up
   `+0.09 → +0.51`.
3. **Scaling the final-layer output down (÷10) is the dominant lever** — alone it
   collapses `S` from 53 to 2.3. Average pooling helps modestly/noisily; the two are
   **complementary for `Ŝ`** (only the combination reaches +0.51). BN-before-activation
   was already on in the baseline.
4. **Smoothness tracks optimizability**: the smooth routine also reaches lower
   held-out loss (`2.08→1.42`; final/10 alone `→0.99`).

---

## 7. per-cluster vs per-example — how they compare

These are not competing techniques; they are two **metaparameterizations** (probe
families) of the *same* routine. per-cluster is per-example restricted to the 10-dim
subspace where weights are constant within a class (`z_example = z_cluster[label]`).

- **They agree on curvature `S`**: both diagnose the baseline as non-smooth (~43–53)
  and both improve ~6–11x under the smooth routine. Interchangeable on this metric.
- **They diverge sharply on `Ŝ`**: per-cluster `+0.09→+0.51`; per-example
  `+0.14→+0.15` (flat). This is a **signal-to-noise** effect, not a real difference in
  the routine's smoothness:
  - `Ŝ` averages sign-agreement over ~6.5M parameters; it needs each parameter to move
    coherently when `z` is nudged.
  - A per-cluster nudge reweights a whole class — a big, structured shove → large,
    sign-stable per-parameter response → `Ŝ` meaningful, and the smooth routine makes
    it consistent (0.51).
  - A per-example nudge spreads `h·v` over 3000 images (~1% each) → minuscule,
    incoherent per-parameter response → consecutive metagradient signs are ~coin-flips
    → `Ŝ≈0.15` (noise floor) for any routine. The tiny ±0.005 spread means
    "consistently uninformative," not "consistently smooth."
- **Practical implication**:
  - **per-cluster = the better smoothness *probe*** (cheap, low-variance, cleanly
    separates routines) and a coarse curation knob (= class reweighting).
  - **per-example = the real curation *target*** (full dataset selection), but its `Ŝ`
    is pinned at the noise floor at this scale. Per-example metagradient curation will
    need the smooth routine **and** scale (more data/epochs/directions, larger
    effective perturbations); use **curvature `S`** (not `Ŝ`) as the per-example
    smoothness metric.

---

## 8. Recommendation

To make this routine amenable to metagradient data curation, change the ResNet-9
classifier to **scale logits down ~10x** and **use average (not max) pooling**,
keeping BatchNorm-before-activation (optionally widen). In code: use
`ms.SMOOTH_ROUTINE` / `ms.SMOOTH_WIDE_ROUTINE`. For the actual training notebook,
that means: divide the final `nn.Linear` output by ~10 and swap
`AdaptiveMaxPool2d`/`MaxPool2d` for the average-pool equivalents.

---

## 9. `metasmooth.py` API quick reference

```python
import metasmooth as ms

# Routines (the paper's menu as a frozen dataclass)
ms.Routine(pool="max"|"avg", bn_before_act=bool, final_scale=float, width=float,
           smooth_act=bool, name=str)
ms.BASELINE_ROUTINE, ms.SMOOTH_ROUTINE, ms.SMOOTH_WIDE_ROUTINE
ms.ResNet9(routine, num_classes=10, in_ch=3)            # nn.Module

# Data + metaparameter
subset = ms.load_cifar_subset(data_dir, n_train=, n_val=, seed=, device=)  # CifarSubset
mp = ms.make_metaparam("per_cluster"|"per_example", subset)                # Metaparam (dim, coord_of_example)

# Training config / the learning algorithm A(z)
cfg = ms.TrainConfig(epochs=16, batch_size=500, lr=0.08, momentum=0.9,
                     weight_decay=5e-4, nesterov=True, warmup_frac=0.3,
                     recalibrate_bn=True, amp="off"|"fp16"|"bf16", tf32=False,
                     init_seed=0, order_seed=1)
res = ms.run_algorithm(subset, mp, z, routine, cfg, device=device)         # -> RunResult(theta, val_loss, train_loss)

# Estimators
dr = ms.measure_direction(run_fn, z0, v, h)   # run_fn: z->RunResult; -> DirectionResult(s_curvature, s_hat, f0, fh, f2h, ...)
v  = ms.sample_direction(dim, seed, normalize=False, device=device)
b  = ms.benchmark_metasmoothness(subset, mp, routine, cfg, h=0.05,
                                 num_directions=3, direction_seed=1000, device=device,
                                 progress=print)  # -> BenchmarkResult(.s_curvature_mean, .s_hat_mean, .f0, ...)
```

Determinism is configured inside `run_algorithm` via `ms.configure_determinism(seed, tf32=False)`.

---

## 10. Reproduce / scale up

```bash
# from the repo root, in the torch311 env:
C:/Users/luequ/micromamba/envs/torch311/python.exe _run_metasmooth_benchmark.py   # writes metasmooth_results.json
C:/Users/luequ/micromamba/envs/torch311/python.exe _render_results.py             # writes metasmooth_results.png + tables
C:/Users/luequ/micromamba/envs/torch311/python.exe -m pytest tests/test_metasmooth.py -q
```
To scale on the VM: raise `N_TRAIN`, `TRAIN.epochs`, `HEAD_DIRS`/`ABL_DIRS` in
`_run_metasmooth_benchmark.py`; keep `amp="off"`. Per-example `Ŝ` should improve with
scale; per-cluster is already a clean signal.

---

## 11. Open items / suggested next steps

1. **Finish the ablation** (`width_x2`, `bn_after_act`) on the VM (laptop throttled).
   Expectation: `bn_after_act` should be *less* smooth (the wrong direction); widening
   should help modestly.
2. **Apply the smooth routine to the real notebook training** (not just the
   metasmoothness probe): scale logits ÷10 and use avg pooling in the actual ResNet-9,
   then check that full-run accuracy holds (paper: smoothness/accuracy trade-off is mild).
3. **Wire metagradient data curation on the smooth routine**: now that the routine is
   metasmooth, compute actual metagradients `∇_z φ` (via the repo's REPLAY engine or a
   short unrolled fp32 run) and do a few MGD steps on per-cluster weights first
   (clean signal), then per-example at scale.
4. **Execute the notebook cells** end-to-end on a faster machine to embed live outputs
   (currently the heavy cells are runnable-but-unexecuted; the Results section has the
   real measured numbers + PNG).
5. **Per-example at scale**: re-measure `Ŝ` with larger `n_train`/`h`/`num_directions`
   to see how far above the noise floor it can be pushed.

---

## 12. Environment cheat-sheet

- Repo root: `C:\ml\dataset-curation\meta-grad-descent-w-clustering`
- Worktree (branch `claude/optimistic-goldberg-0ac9e8`):
  `C:\ml\dataset-curation\meta-grad-descent-w-clustering\.claude\worktrees\optimistic-goldberg-0ac9e8`
- Python: `C:/Users/luequ/micromamba/envs/torch311/python.exe` (torch 2.8.0+cu128,
  torchvision 0.23.0+cu128, CUDA, RTX 4050 Laptop GPU)
- CIFAR-10 data: `C:\ml\dataset-curation\meta-grad-descent-w-clustering\data`
- Paper: arXiv:2503.13751 (Engstrom et al., 2025).
