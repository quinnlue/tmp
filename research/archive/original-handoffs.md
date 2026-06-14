# Original handoff & study documents (archived)

These are the **original, verbatim** per-workstream handoff and study
documents that predate the consolidated write-ups in [`../`](../README.md).
They are kept here unedited as raw provenance: the refined, organized,
cross-referenced versions of this material live in `research/01`–`research/08`.
If anything in the consolidated docs is ambiguous, the source of truth is here.

Archived 2026-06-14. Contents (in original form):

1. `HANDOFF_metasmoothness.md` — ResNet-9/CIFAR-10 metasmoothness (see [research/02](../02-metasmoothness-resnet9.md))
2. `HANDOFF_vm_phaseABC.md` — VM Phases A/B/C continuation (see [research/02](../02-metasmoothness-resnet9.md))
3. `HANDOFF_vit_metasmoothness.md` — ViT metasmoothness (see [research/03](../03-metasmoothness-vit.md))
4. `CIFAR100_LT_VIT_SCALE_STUDY.md` — CIFAR100-LT ViT scale study (see [research/04](../04-cifar100lt-vit-scale-study.md))


---

<!-- ====== Begin verbatim: HANDOFF_metasmoothness.md ====== -->

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


---

<!-- ====== Begin verbatim: HANDOFF_vm_phaseABC.md ====== -->

# Handoff: metasmoothness → curation, VM continuation (Phases A/B/C)

Paste this whole file into a new chat to continue. It is self-contained: what was
built on the VMs, all results, the engineering gotchas, file locations, and the
open threads. It assumes the background in `HANDOFF_metasmoothness.md` and the plan
in the Phase-A/B/C steering doc (both in this repo / earlier context).

---

## 0. One-paragraph summary

We continued the metasmoothness → metagradient-curation work (Engstrom et al. 2025,
arXiv:2503.13751) on rented GPUs and finished all three planned phases. **(A)** Scaled
and tightened the metasmoothness measurement of the CIFAR-10 ResNet-9 routine — the
smooth menu's ranking is stable and reproducible at larger scale, and a GroupNorm
variant stays metasmooth. **(B)** Retrofitted the smooth routine into real augmented
training: with its learning rate retuned up (the ÷10 logits rescale gradients) it
**beats** the baseline (93.1% vs 91.2% test acc at 25 epochs). **(C)** Ran real
per-cluster metagradient descent through smooth-routine training (REPLAY backend) and
**measurably reduced held-out cross-entropy** (−8.1%) under a distribution shift,
using the repo's existing differentiable engine. All code is working but **uncommitted**.

---

## 1. Environment (VMs are ephemeral — likely gone next session)

- Two **A40 (46 GB)** vast.ai boxes were used:
  - VM1 `ssh -p 4462 root@160.250.70.29` — Phases B and C.
  - VM2 `ssh -p 4146 root@160.250.70.25` — the deeper Phase-A run.
  - Both: `torch 2.12.0+cu126`, interpreter **`/venv/main/bin/python`**, repo at
    `/workspace/tmp` cloned from the **public** `https://github.com/quinnlue/tmp.git`.
- Dev-on-laptop / run-on-VM. Laptop repo root: `C:\ml\dataset-curation\meta-grad-descent-w-clustering`.
  Laptop interpreter (for tests/smoke): `C:/Users/luequ/micromamba/envs/torch311/python.exe`.
- **Per-VM setup that must be redone on a fresh box** (≈5 min):
  1. `git clone --depth 1 https://github.com/quinnlue/tmp.git` into `/workspace`.
  2. `/venv/main/bin/pip install -q pytest pyarrow pillow matplotlib`.
  3. **CIFAR data**: the torchvision mirror (cs.toronto.edu) is throttled to ~30 KB/s
     on these boxes. Do **not** let torchvision download. Build an `.npz` cache from
     the HuggingFace CDN (~130 MB/s): `python _build_cifar_cache.py /workspace/tmp/data`
     → `data/cifar10_train.npz` (50k) + `cifar10_test.npz` (10k). `load_cifar_subset`
     has a fast-path that reads `{data_dir}/cifar10_train.npz` when present.
  4. **scp the laptop's updated files over the cloned ones** (the github copies are
     older): `metasmooth.py`, `_run_metasmooth_vm.py`, `_render_vm.py`,
     `_build_cifar_cache.py`, plus the Phase B/C scripts as needed.
  5. Validate: `python -m pytest tests/test_metasmooth.py -q` (10 pass) and
     `python -c "import metasmooth as ms; print(ms.SMOOTH_GN_ROUTINE.norm)"` → `gn`.

---

## 2. Code changes (laptop, all UNCOMMITTED)

### `metasmooth.py` (edited in place; tests still pass)
1. **CIFAR `.npz` fast-path** in `load_cifar_subset` (reads `cifar10_train.npz` if
   present, else torchvision download). Backward compatible.
2. **`val_acc`** plumbed through: field on `RunResult` (computed in `run_algorithm`),
   `acc0` on `DirectionResult`, `BenchmarkResult.val_acc` property, shown in `summary()`.
   Enables the smoothness-vs-accuracy curve.
3. **GroupNorm support**: `Routine` gains `norm: "bn"|"gn"` and `gn_groups` fields;
   `ConvBlock` builds BN or GN via `_make_norm`; `ResNet9.block` passes them; new
   module-level **`SMOOTH_GN_ROUTINE`** (smooth + GroupNorm). GroupNorm has **0 buffers**,
   so exact metagradients through training are clean (the functional engine holds
   buffers fixed) — this is the Phase-C curation model.

### New scripts (all top-level `_*.py`)
- `_build_cifar_cache.py` — HF-parquet → `.npz` CIFAR cache.
- `_run_metasmooth_vm.py` — **parameterized** Phase-A benchmark (env vars: `N_TRAIN
  N_VAL EPOCHS BATCH LR H HEAD_DIRS PE_DIRS ABL_DIRS WIDE_DIRS DIR_SEED MAX_MINUTES
  CIFAR_DIR OUT`). 4 staged sections (head-to-head / menu / combos / per-example-h),
  writes the JSON **after every bench**, soft wall-clock budget guard. fp32 default.
- `_render_vm.py` — JSON → markdown tables + `metasmooth_vm_ranking.png` +
  `metasmooth_vm_tradeoff.png` (Ŝ-vs-accuracy scatter).
- `_train_smooth_vm.py` — Phase B: real augmented training (GPU RandomCrop+flip, AMP,
  cudnn.benchmark, SGD+cosine+warmup), (routine, lr) grid, writes test acc per config.
- `_phase_c_mgd_vm.py` — Phase C: per-cluster MGD. Env: `METHOD=replay|store_all`,
  `N_POOL N_OBJ N_VAL INNER_EPOCHS BATCH INNER_LR META_STEPS META_LR BRANCHING`,
  and demo switches `CORRUPT_CLASSES/CORRUPT_FRAC` (label noise) and `TARGET_CLASSES`
  (distribution shift). Writes history (obj_CE, val_CE, z, class_weights) per step.
- `_phase_c_smoke.py` — CPU/laptop validation that the engine differentiates held-out
  CE w.r.t. per-cluster logits z (no CIFAR needed).
- `_vm_timing.py`, `_vm_calib.py` — throwaway A40 timing calibration.

---

## 3. Phase A — metasmoothness, scaled

Two runs, same routine definitions / `h=0.05` / `dir_seed=1000` (apples-to-apples):
- `metasmooth_vm_results.json` — n_train=6000, 18 ep, 6/4 directions.
- `metasmooth_vm_deep_results.json` — n_train=8000, 20 ep, **12** directions, +`smooth_gn`
  (the tighter, more reliable run; 218 min on one A40).

**Cost note:** fp32 + `cudnn.deterministic` is the bottleneck on A40 (baseline
6000×18 = 20.6 s/run; ~3.2× for width-2; a probe = 3 runs). The metric is **relative** —
scaling **directions** is the cheap reliability lever, not n_train. Do **not** switch
to TF32/fp16/bf16 — they flip the sign of the tiny Definition-1 second difference.

**Deep run, per-cluster, ranked by Ŝ (higher = smoother):**

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

Per-example smooth (noise floor lifts with h): Ŝ +0.036→+0.102→+0.122 as h 0.05→0.1→0.2
(S 3.8→1.8→0.3). baseline per-example Ŝ ≈ 0.

**Takeaways (stable across both scales):**
1. Baseline is **not** metasmooth (Ŝ≈0); the smooth menu fixes it.
2. **`final_scale` is the dominant Ŝ lever** (monotone f3→f10→f30); `avg+f30` maxes Ŝ
   (+0.49) but at an accuracy cost — a real **smoothness↔accuracy frontier**
   (see `metasmooth_vm_tradeoff_deep.png`). `smooth`/`smooth_wide` are the production
   sweet spot (high Ŝ **and** high accuracy).
3. **Non-smooth directions verify**: `bn_after_act`, `gelu`-alone, `width`-alone all
   sit near Ŝ≈0.
4. **GroupNorm stays metasmooth** (`smooth_gn` Ŝ=+0.196 ≈ smooth's +0.207) — required
   re-check before trusting Phase C metagradients. NB its *curvature S* is high and its
   *accuracy is low under the benchmark's SGD/lr=0.08* — GN wants a different optimizer;
   Phase C trains it with functional AdamW and it's fine. The S-vs-Ŝ divergence for GN
   is worth a closer look.

Render: `python _render_vm.py metasmooth_vm_deep_results.json`.

---

## 4. Phase B — retrofit + accuracy (VM1, `train_smooth_vm_results.json`)

25-epoch augmented training (full 45k/5k/10k splits), AMP, LR sweep:

| routine | best test acc | @ lr |
|---|---|---|
| baseline | 91.2% | 0.2 |
| smooth | **93.1%** | 0.6 |
| smooth_wide | **93.7%** | 0.6 |

The ÷10 logits divide the loss gradient by ~10, so the smooth routine wants a **higher
LR** (0.1→91.2, 0.6→93.1). With it retuned, the smooth routine **beats** the baseline —
the "accuracy cost" is a gain. (No fp32 constraint here — real training, AMP is fine.)

---

## 5. Phase C — per-cluster metagradient curation (VM1)

**Engine reuse (key discovery):** the repo's differentiable engine is **already
classification-based** (the WIP converted MAE→classifier). `weighted_inner_step`
(`functional_train.py`) does one functional smooth-AdamW step with the
group-softmax reparam (`weighting.py`); `metagrad.train_unrolled` is store-all;
`recursive_replay.recursive_replay_state` is REPLAY. The curation model is
`metasmooth.ResNet9(SMOOTH_GN_ROUTINE)` (GroupNorm → 0 buffers). z = per-cluster
logits (R¹⁰); weights = `10·softmax(z)`; z=0 ⇒ uniform training.

**Memory:** store-all OOMs past ~50 inner steps on the A40. **Use REPLAY**
(`recursive_replay_state`, `branching_factor=4`) → O(k logₖT) memory → T=192 inner
steps fits easily. REPLAY does **not** support higher-order diff (it raises if grad is
enabled), so compute the first-order metagradient with `autograd.grad(obj, z)` and a
plain Adam step on z — fine for MGD.

**Three experiments map where per-cluster curation helps:**

| setup | held-out CE | result | file |
|---|---|---|---|
| balanced | 1.813 → 1.831 | no exploitable signal (uniform ≈ optimal) | `phase_c_balanced_results.json` |
| label-noise (cls 3,5 randomized) | 2.153 → 2.227 | **degenerate**: held-out *contains* the corrupted classes → MGD boosts their logit bias (cat↓0.58 but dog↑2.15) instead of removing noise | `phase_c_corrupt_results.json` |
| **distribution shift** (objective = vehicles 0,1,8,9) | **1.860 → 1.709 (−8.1%)**; val 1.834 → 1.681 (−8.3%) | **MGD measurably cuts held-out CE and it generalizes** | `phase_c_shift_results.json` |

Config for the winning run: `METHOD=replay N_POOL=4000 N_OBJ=3000 N_VAL=3000
INNER_EPOCHS=12 BATCH=250 INNER_LR=0.02 META_STEPS=20 META_LR=0.05 TARGET_CLASSES=0,1,8,9`
(T=192 inner steps; inner trajectory is fixed, only z varies, so the CE drop is purely
the learned weighting). The learned per-class weights are a **non-trivial** reweighting,
not naive distribution-matching (it down-weights truck while up-weighting
automobile/ship and even some animals — exploiting the real confusion structure). That
nuance is itself a finding; a cleaner weight story may need more meta-steps / lower
META_LR / longer inner training.

---

## 6. Engineering gotchas (don't relearn these)

- **fp32 only** for metasmoothness/finite-difference (TF32/fp16/bf16 flip the Def-1 sign).
- **CIFAR mirror throttled** → HF parquet cache (`_build_cifar_cache.py`).
- **BN + metagradients**: BN running-stat buffers are held fixed by the functional
  engine → use **GroupNorm** (`SMOOTH_GN_ROUTINE`, 0 buffers) for curation. Re-validated
  smooth in Phase A.
- **nohup-over-ssh**: the ssh channel stays open until the backgrounded process's fds
  close. For a clean immediate-return detach use `setsid <cmd> > log 2>&1 < /dev/null &`.
- **`pgrep -f <pattern>` self-matches** the polling command (its own cmdline contains
  the pattern) → infinite "waiting" loops. Poll by grepping a **log-file marker**
  (e.g. `"ALL DONE"`, `"MGD per-cluster summary"`, `Traceback`) instead.
- **tail/head buffering**: piping a long-running command through `tail`/`head` buffers
  until exit — read the log file directly.
- **Phase C demo design**: balanced data gives per-cluster MGD nothing; label-noise on
  classes that are present in the held-out objective creates a degenerate bias-boosting
  optimum; **distribution shift** (held-out = a target subset of classes) is the clean demo.

---

## 7. Open threads / suggested next steps

1. **Per-example MGD at scale** — the natural next step. Use `paper_mgd.py` (count
   relaxation + Algorithm 1 + REPLAY) rather than the group-softmax path; per-example z
   needs REPLAY for any real horizon. Expect to need a corruption/shift signal (per the
   Phase-C findings) to show a clean win.
2. **Commit** the uncommitted files (the `metasmooth.py` edits + the `_*_vm.py` scripts)
   to a branch.
3. **GroupNorm follow-up**: retune GN's optimizer/LR for accuracy in the benchmark, and
   understand the high-S / high-Ŝ divergence for `smooth_gn`.
4. **Cleaner Phase-C weight story**: more meta-steps / lower META_LR / longer (REPLAY)
   inner training; try a class-imbalance (rather than subset) shift.
5. Re-run Phase A at even larger n / more directions if a publication-grade ranking is
   wanted (cost ~linear in n_train×epochs×directions; budget accordingly).

## 8. Artifacts on the laptop (repo root)

`metasmooth_vm_results.json` (n=6000), `metasmooth_vm_deep_results.json` (n=8000/12-dir)
+ `metasmooth_vm_ranking_deep.png` / `metasmooth_vm_tradeoff_deep.png`;
`train_smooth_vm_results.json` (Phase B);
`phase_c_balanced_results.json` / `phase_c_corrupt_results.json` /
`phase_c_shift_results.json` (Phase C). Reproduce any run with the `_*_vm.py` scripts
above (env-var configs are in §2–§5).


---

<!-- ====== Begin verbatim: HANDOFF_vit_metasmoothness.md ====== -->

# Handoff: Metasmoothness of a Vision Transformer backbone (CIFAR-10, local/overnight)

Paste this whole file into a new chat to continue the work. It is self-contained:
background, the paper, the ViT "menu", what was built, how to run it locally
overnight, the methodology caveats, and where to put the results.

This is the **transformer analogue** of the ResNet-9 study in
`HANDOFF_metasmoothness.md` (laptop Phase A) and `HANDOFF_vm_phaseABC.md`
(VM Phases A/B/C). Read those for the original numbers; this doc mirrors their
structure for a ViT.

---

## 0. One-paragraph summary

We measure how *metasmooth* a Vision Transformer training routine is on CIFAR-10
— i.e. how amenable it is to metagradient-based data curation — and maximize it
with a transformer "menu", reusing the paper's two finite-difference diagnostics
(arXiv:2503.13751, Defs 1 & 2; 3 deterministic trainings per probe, no
metagradients). The diagnostics live in `metasmooth.py` and are model-agnostic;
we only swap in a ViT learning algorithm `A(z)` trained with the repo's
functional **smooth AdamW** (eps inside the sqrt — the paper's "smooth
optimizer"). **Phase A** ranks the ViT menu (logit-scale, pre/post-norm,
mean/CLS pool, GELU/ReLU, width, depth) by smoothness; **Phase B** checks the
smooth menu doesn't cost held-out accuracy. Both are scoped to run on the local
RTX 4050 laptop overnight in fp32. The toy-scale smoke run already reproduces the
ResNet-9 headline: **scaling the final-layer logits down is the dominant
smoothness lever** (curvature `S`: scale 1 → 1.60, /3 → 0.72, /10 → 0.09, /30 →
0.02), and the smooth routine is far smoother than the baseline ViT.

---

## 1. Project background

- Repo: `C:\ml\dataset-curation\meta-grad-descent-w-clustering`. A **PyTorch**
  (functional, not JAX) metagradient **data-curation** experiment.
- Working metagradient engine already present: `functional_train.py` (functional
  smooth AdamW), `metagrad.py`, `paper_mgd.py`, `replay.py` / `recursive_replay.py`
  (REPLAY). The metasmoothness diagnostics do **not** need the engine — only
  ordinary deterministic trainings — but Phase A trains via
  `functional_train.weighted_inner_step(create_graph=False)` so the same
  parameterization carries over to (future) ViT curation.
- Workflow: **dev-and-run-on-laptop** here (overnight), unlike the ResNet-9 work
  which scaled on a VM. Metasmoothness is a *relative* metric, so modest local
  scale is fine — the routine *ordering* is the result.
- Environment: `C:/Users/luequ/micromamba/envs/torch311/python.exe` — torch
  2.8.0+cu128, torchvision 0.23.0+cu128, CUDA on an **RTX 4050 Laptop GPU**.
- Data: CIFAR-10 npz cache at `./data/cifar10_{train,test}.npz` (build with
  `python _build_cifar_cache.py ./data` — pulls the HuggingFace parquet mirror).

---

## 2. The paper

**"Optimizing ML Training with Metagradient Descent"** — Engstrom et al.,
[arXiv:2503.13751](https://arxiv.org/abs/2503.13751), 2025.

- **Metasmoothness** (Sec 3): metagradients are only useful if `f = φ∘A` is smooth
  in `z`. Two cheap, metagradient-free diagnostics, each from **3 deterministic
  trainings** at `z`, `z+hv`, `z+2hv`:
  - **Def 1 (curvature)** `S = |f(z+2hv) − 2f(z+hv) + f(z)| / h²`. **Lower = smoother.**
  - **Def 2 (empirical metasmoothness)** in parameter space:
    `Ŝ = sign(Δ_A(z;v))ᵀ diag(d/‖d‖₁) sign(Δ_A(z+hv;v)) ∈ [−1,1]`. **Higher = smoother.**
- **Smooth-model menu** (Sec 3.2, Remark 4): make a routine metasmooth by trying
  design changes and keeping those that raise `Ŝ`. For ResNet/CIFAR: average
  pooling, BN-before-activation, **scale the last layer's output down ~10×**,
  wider networks, and a smooth optimizer. We port the architecture-agnostic
  levers to a ViT (Section 4).

---

## 3. The task (as given)

> Evaluate the metasmoothness of transformer / ViT backbones, scoped to local
> compute, running overnight; mirror the ResNet-9 procedure. Decisions confirmed:
> **Phases A + B**, optimizer = **functional SmoothAdamW**, menu levers =
> logit-scale {1,3,10,30}, pre- vs post-norm, mean vs CLS pooling, GELU vs ReLU,
> width, depth.

- `φ` = held-out **cross-entropy** on a disjoint CIFAR-10 split.
- `z` = continuous **data weights at z=0** (the paper's count relaxation): a
  per-group softmax reweighting (`weighting.weighted_example_loss`); `z=0` ⇒
  uniform = ordinary training. Probed for **per-cluster** (one weight per CIFAR
  class, dim 10) and **per-example** (one weight per image, dim = n_train).

---

## 4. The ViT metasmoothness "menu" (`_vit_menu.py`)

The transformer analogue of `metasmooth.Routine`, a frozen dataclass `ViTRoutine`:

| lever | field | smooth choice | notes |
|---|---|---|---|
| final-logit scale | `final_scale` | 10.0 (÷10) | the **dominant** lever (ResNet-9 finding, reproduced) |
| norm placement | `pre_norm` | `False` (post-norm) | transformer-specific; worktree probes favored post-norm for clean metagradients |
| token pooling | `pool` | `"mean"` | average pooling is the paper's smooth choice; `"cls"` is the standard default |
| activation | `smooth_act` | `True` (GELU) | GELU vs ReLU |
| width | `width_mult` | 2.0 | wider == more metasmooth |
| depth | `depth` | — | extra capacity lever |

Pre-defined: `BASELINE_VIT` (mean, pre-norm, GELU, scale 1), `SMOOTH_VIT`
(post-norm + scale 10), `SMOOTH_WIDE_VIT` (+ width ×2). `build_config(num_classes,
geom)` snaps `encoder_dim` to a multiple of `lcm(heads, 4)` so `ViTConfig`'s
invariants hold.

The backbone is `model.VisionTransformerClassifier`. **`model.py` was extended**
(this work) to add the menu fields `pre_norm`, `pool`, `smooth_activation`,
`final_logit_scale`, a CLS token, and the post-norm block path — with defaults
that reproduce the original behavior exactly (so `tests/test_model.py` is
unchanged). Sinusoidal 2-D positions + MATH-backend SDPA attention (deterministic)
were already present.

---

## 5. What was built (deliverables)

In the main checkout:

| file | what |
|---|---|
| `model.py` | **edited**: ViT smooth-menu fields + CLS token + post-norm path (backward-compatible defaults). |
| `_vit_menu.py` | `ViTRoutine` / `ViTGeometry` menu + `BASELINE_VIT` / `SMOOTH_VIT` / `SMOOTH_WIDE_VIT`. Shared by both runners. |
| `_run_vit_metasmooth_local.py` | **Phase A**: CIFAR-10 ViT metasmoothness. `A(z)` = deterministic weighted training via functional smooth AdamW; drives `ms.measure_direction`. Stages: head-to-head, per-cluster menu search, per-example h-sweep. Env-var config, soft `MAX_MINUTES`, JSON after every bench. |
| `_run_vit_smooth_train_local.py` | **Phase B**: augmented AdamW training + LR sweep; best test/val accuracy per (routine, lr). |
| `_render_vit_results.py` | JSON → markdown tables (ranked by `Ŝ`) + `vit_metasmooth_{ranking,tradeoff}.png`. Also prints the Phase-B accuracy table if present. |
| `tests/test_vit_metasmooth.py` | 6 CPU tests: training determinism (equal `z` ⇒ identical θ), menu construction (CLS + post-norm paths), `final_scale` divides logits, estimator end-to-end on a toy ViT. |

Tests: `…/torch311/python.exe -m pytest tests/test_vit_metasmooth.py tests/test_model.py -q` → **12 passed**.

---

## 6. Methodology / key engineering facts (don't relearn these)

- **fp32 is mandatory** for Phase A (`amp="off"` equivalent: model + SmoothAdamW
  in float32, `configure_determinism(..., tf32=False)`). fp16/bf16 flip the sign
  of the tiny Definition-1 second difference.
- **Bit-determinism:** the only thing varying across the 3 runs of a probe is `z`.
  Fixed `init_seed`/`order_seed`, no augmentation, `cudnn.deterministic=True` (via
  `ms.configure_determinism`), MATH SDPA backend. Verified by
  `test_training_is_deterministic_for_equal_z` (max|Δθ| = 0).
- **No BatchNorm:** ViT is LayerNorm-only, so there are **no running-stat buffers**
  and **no BN recalibration** is needed (the ResNet-9 gotcha doesn't apply); the
  functional path is clean.
- **Weighting** is softmax-based (`weighting.weighted_example_loss`): at `z=0`
  every group mass is uniform ⇒ multipliers = 1 ⇒ exactly standard training. This
  matches the Phase-C curation parameterization, not the ResNet-9 `exp(z)` form.
- **Windows env-var gotcha:** the temperature knob is `TEMPERATURE`, **not** `TEMP`
  (Windows reserves `TEMP` for the temp dir).
- **Probe directions:** `v ~ N(0, I)` (unnormalized), same `DIR_SEED` reused across
  routines for apples-to-apples; `h=0.05`.
- **Per-example `Ŝ`** will sit near its noise floor at this scale (as for ResNet-9:
  a per-example nudge spreads `h·v` over thousands of images ⇒ incoherent
  per-parameter response). Use **curvature `S`** as the per-example metric and
  per-cluster `Ŝ` for the clean routine ranking.

---

## 7. How to run (local, overnight)

```powershell
# one-time: build the CIFAR-10 cache
C:/Users/luequ/micromamba/envs/torch311/python.exe _build_cifar_cache.py ./data

# Phase A — metasmoothness ranking (recommended overnight defaults baked in)
C:/Users/luequ/micromamba/envs/torch311/python.exe _run_vit_metasmooth_local.py
#   defaults: N_TRAIN=4000 N_VAL=2000 EPOCHS=20 BATCH=500
#             DIM=192 DEPTH=6 HEADS=6 PATCH=8 MLP_RATIO=2.0
#             H=0.05 HEAD_DIRS=4 PE_DIRS=3 ABL_DIRS=3 MAX_MINUTES=360
#   -> vit_metasmooth_results.json   (written after every bench)

# Phase B — does the smooth menu cost accuracy? (run after A, or a 2nd night)
C:/Users/luequ/micromamba/envs/torch311/python.exe _run_vit_smooth_train_local.py
#   defaults: N_TRAIN=20000 N_VAL=5000 EPOCHS=30 ... MAX_MINUTES=240
#   -> vit_train_results.json

# render tables + PNGs
C:/Users/luequ/micromamba/envs/torch311/python.exe _render_vit_results.py
```

Override any knob via env vars (PowerShell: `$env:EPOCHS=15; python …`). To dial
the overnight budget, trade `N_TRAIN`/`EPOCHS`/`*_DIRS` against `MAX_MINUTES`;
the JSON is always left consistent because it's rewritten after each bench.

Smoke test (≈10 s, proves the pipeline + ordering):
`N_TRAIN=400 N_VAL=200 EPOCHS=2 DIM=48 DEPTH=2 HEADS=4 HEAD_DIRS=1 PE_DIRS=1 ABL_DIRS=1 WIDE_DIRS=1 OUT=smoke.json python _run_vit_metasmooth_local.py`.

---

## 8. Results

Run 1: `n_train=4000, n_val=2000, epochs=20, batch=500, fp32, dim=192 depth=6
heads=6 patch=8, SmoothAdamW lr=2e-3 eps=1e-4, h=0.05, dir_seed=1000`. Phase A
finished in **25 min**; Phase B (n_train=20000, 30 ep, AdamW) in ~6 min.

### 8.1 Phase A — per-cluster ranking by Ŝ (higher = smoother)

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

Head-to-head also ran per_example; per-example Ŝ is near its noise floor and `S`
falls sharply with `h` (S: 6.11 → 1.59 → 0.25 for h = 0.05/0.10/0.20), exactly
the ResNet-9 pattern — use per-example `S`, not `Ŝ`.

### 8.2 Phase B — best test accuracy per routine (AdamW, augmented)

| routine | best test_acc | @ lr | note |
|---|---|---|---|
| baseline | **0.6685** | 1e-3 | |
| smooth | 0.6481 | 1e-3 | higher LR *hurts* (0.59 @ 3e-3, collapses @ 6e-3) |
| smooth_wide | 0.10 (chance) | — | collapsed at every tested LR (only ≥3e-3 swept) |

### 8.3 Takeaways (these differ from the ResNet-9 story — report honestly)

1. **`final_scale` is the dominant lever for curvature `S`** — monotone
   16.6 → 10.1 → 6.7 → 5.0 for scale 1 → 3 → 10 → 30. This *does* replicate the
   paper / ResNet-9 on Definition 1.
2. **But the composite `smooth` routine is not the smoothest, because post-norm
   is a *negative* lever for the ViT.** `post_norm` alone has the **worst** `S`
   (24.2) — pairing it with /10 (the `smooth` routine) lands at `S=9.5`, *worse*
   than **`final/10` alone (`S=6.7`, `Ŝ=+0.253`)**, which is the actual sweet
   spot. The worktree's "post-norm is cleaner for ViT" assumption does **not**
   hold here.
3. **`Ŝ` (Def 2) does not cleanly rank ViT routines at this scale** — the baseline
   has the *highest* per-cluster `Ŝ` (+0.368), and `Ŝ` only weakly tracks `S`.
   Prefer **curvature `S`** as the ViT smoothness metric for now; push `Ŝ` with
   more directions / larger `n_train` before trusting its ordering.
4. Confirmed-as-expected levers: **mean ≥ CLS** pool (cls `S`=19.4 > baseline),
   **GELU ≥ ReLU** (relu `S`=21.6), **depth helps** (`S`=7.0, best val_acc 0.435).
   **width hurt** here (`S`=19.7, high `f0`) — likely an optimization/LR artifact
   at this scale, not a true smoothness loss.
5. **Phase B reverses the ResNet-9 LR finding.** With **AdamW** the update is
   already normalized by the gradient's second moment, so the ÷10 logit scaling
   does *not* call for a higher LR — raising LR just destabilizes (`smooth`
   collapses by 6e-3; `smooth_wide` collapsed entirely, though it was only swept
   at ≥3e-3). At matched LR the smooth ViT is *slightly below* baseline
   (0.648 vs 0.669), i.e. a mild accuracy cost, not a gain.

**Actionable correction:** redefine `SMOOTH_VIT` as **mean + pre-norm + GELU +
final_scale=10** (i.e. drop post-norm; optionally add depth), and re-sweep Phase
B from **lr ≤ 1e-3 downward** for the wide variant. See §9.

### 8.4 Per-example, examined more closely (`_run_vit_per_example_local.py`)

Per-example (one weight per training image, dim = n_train) is the real curation
target. Probed across the menu (3 dirs, h=0.05) plus an h-sweep:

| routine | S @ h=.05 | Ŝ @ h=.05 | S @ h=.1 | S @ h=.2 | S @ h=.4 |
|---|---|---|---|---|---|
| baseline | 3.69 | **+0.78** | 1.56 | 1.18 | 0.27 |
| final/10 | 5.06 | +0.18 | **1.14** | **0.30** | **0.20** |
| final/30 | 3.51 | +0.10 | — | — | — |
| depth_x2 | 13.10 | +0.29 | — | — | — |
| relu | 4.15 | +0.75 | — | — | — |

Two clean conclusions:

1. **Per-example `Ŝ` is not a trustworthy routine-ranking metric — it's
   confounded by update magnitude.** It *inverts* the per-cluster ordering:
   baseline/relu score high (+0.75–0.78) while the genuinely-smoother scaled
   routines score low (+0.10–0.18). Reason: the ÷10 logit scale shrinks the
   per-step parameter updates, so under a tiny per-example nudge θ moves less and
   sign-agreement is dominated by optimizer noise. The big-step baseline just
   *looks* coherent. As `h` grows the two converge (baseline Ŝ 0.78 → 0.41),
   confirming the small-h value is a magnitude artifact. **Don't rank ViT
   routines by per-example `Ŝ`.**
2. **Per-example curvature `S` is strongly h-dependent; probe at h ≥ 0.1.** At the
   tiny h=0.05 it's in the noise (final/10 ≈ or > baseline). But at **h ≥ 0.1 the
   final-scale lever clearly lowers `S`** — final/10 < baseline at h=0.1 (1.14 vs
   1.56), h=0.2 (0.30 vs 1.18), and h=0.4 (0.20 vs 0.27) — recovering the
   per-cluster Definition-1 story. So for per-example smoothness, use **curvature
   `S` at h ≥ 0.1**, not `Ŝ` and not h=0.05.

---

## 9. Open items / suggested next steps

1. **Redefine `SMOOTH_VIT` to drop post-norm** (mean + pre-norm + GELU +
   final_scale=10, the `final/10` sweet spot) in `_vit_menu.py`, and re-run Phase
   A to confirm it beats baseline on `S`. Consider a `smooth_deep` adding depth.
2. **Re-sweep Phase B at lower LR** — the smooth/wide ViT wants lr ≤ 1e-3 (AdamW
   normalizes gradient scale, so /10 logits do *not* need a higher LR). The
   current grid only tried wide at ≥3e-3, which collapsed.
3. If per-cluster `Ŝ` is to be trusted as a ranking metric, raise
   `HEAD_DIRS`/`ABL_DIRS` and/or `N_TRAIN` — at the current scale `Ŝ` is noisy and
   doesn't track `S`; `S` is the reliable ViT metric for now.
4. **Phase C analogue** (not in scope here): with a confirmed-smooth ViT, wire
   per-cluster metagradient descent via the REPLAY engine (`recursive_replay.py`)
   under distribution shift — mirror `_phase_c_mgd_vm.py`. The ViT is LayerNorm-
   only, so it needs no GroupNorm swap (unlike the ResNet-9 Phase C).
5. Compare optimizers: re-run the menu under plain SGD to see whether the
   smoothness ranking (and the Phase-B LR behavior) is optimizer-dependent.

---

## 10. Environment cheat-sheet

- Repo root: `C:\ml\dataset-curation\meta-grad-descent-w-clustering`
- Python: `C:/Users/luequ/micromamba/envs/torch311/python.exe` (torch 2.8.0+cu128,
  RTX 4050 Laptop GPU)
- CIFAR-10 cache: `./data/cifar10_{train,test}.npz` (`_build_cifar_cache.py`)
- Paper: arXiv:2503.13751 (Engstrom et al., 2025).
- Reusable estimators: `metasmooth.py` (`measure_direction`, `sample_direction`,
  `RunResult`, `DirectionResult`, `configure_determinism`, `load_cifar_subset`).


---

<!-- ====== Begin verbatim: CIFAR100_LT_VIT_SCALE_STUDY.md ====== -->

# CIFAR100-LT corrected-smooth ViT scale study

## Setup

- Dataset: `tomas-gajarsky/cifar100-lt`, `r-100`
- Long-tailed training pool: 10,847 images
- Balanced official test split: 50 images/class for validation and 50 images/class
  for the untouched final test
- Architecture correction from the ViT metasmoothness study:
  mean pooling, pre-norm, GELU, and `final_scale=10`
- Optimizer: AdamW, bf16, random crop/flip, 10% warmup, cosine decay
- Profiles:
  - Tiny: 5,367,460 parameters, dim 192, depth 12, 3 heads
  - Small: 21,351,652 parameters, dim 384, depth 12, 6 heads
  - Base: 85,170,532 parameters, dim 768, depth 12, 12 heads

The 25- and 75-epoch matrices sweep learning rates `1e-3`, `5e-4`, and `2e-4`.
The 150-epoch runs extend the scale-matched winner for each profile.

## Results

### Full 75-epoch learning-rate matrix

| Profile | LR | Balanced test accuracy | Many | Medium | Few | Minutes |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 1e-3 | **23.42%** | 45.77% | 19.49% | 1.93% | 3.0 |
| Tiny | 5e-4 | 22.62% | 48.86% | 15.77% | 0.00% | 3.0 |
| Tiny | 2e-4 | 19.32% | 48.11% | 7.09% | 0.00% | 3.1 |
| Small | 1e-3 | 20.58% | 40.23% | 16.51% | 2.40% | 6.2 |
| Small | 5e-4 | **23.94%** | 46.29% | 20.17% | 2.27% | 6.2 |
| Small | 2e-4 | 23.22% | 50.11% | 16.17% | 0.07% | 6.2 |
| Base | 1e-3 | 16.50% | 33.71% | 12.57% | 1.00% | 14.4 |
| Base | 5e-4 | 21.92% | 42.17% | 18.74% | **2.00%** | 14.4 |
| Base | 2e-4 | **25.82%** | 50.29% | 22.63% | 1.00% | 14.4 |

Capacity only helps when the learning rate falls with scale. The corrected
lower-LR sweep is essential for ViT-Base.

### Best scale-matched configuration across epoch budgets

| Profile | Selected LR | 25 epochs | 75 epochs | 150 epochs | 150-epoch best val epoch |
|---|---:|---:|---:|---:|---:|
| Tiny | 1e-3 | 14.00% | 23.42% | 25.96% | 90 |
| Small | 5e-4 | 15.52% | 23.94% | 27.58% | 138 |
| Base | 2e-4 | 16.90% | 25.82% | **27.64%** | 123 |

The independent 150-epoch finalists were run concurrently, so their wall-clock
times are not directly comparable. Use the 75-epoch matrix for clean relative
runtime comparisons.

### 150-epoch final-test tiers

| Profile | Balanced | Many | Medium | Few |
|---|---:|---:|---:|---:|
| Tiny | 25.96% | 47.49% | 22.80% | **4.53%** |
| Small | 27.58% | 50.97% | 24.00% | 4.47% |
| Base | **27.64%** | **51.09%** | **24.40%** | 4.07% |

## Decision

- **Ordinary-training efficiency knee: ViT-Small at `5e-4`.** It reaches 99.8%
  of Base accuracy with 25.1% of Base parameters.
- **Primary full-model MGD backbone: ViT-Tiny at `1e-3`.** It retains 93.9% of
  Base accuracy with 6.3% of Base parameters. Exact REPLAY cost is dominated by
  model size and trajectory length, so Tiny permits meaningfully deeper and
  better-converged MGD instead of spending the budget on model width.
- **Capacity confirmation: ViT-Small at `5e-4`.** Run it after the Tiny MGD
  comparison has stable settings.
- **Do not use Base for the initial MGD matrix.** At 150 epochs it adds only
  0.06 balanced-accuracy points over Small and 1.68 points over Tiny.

## Full-model MGD plan

Use the corrected-smooth ViT-Tiny with a 100-epoch inner horizon. The 150-epoch
curve peaks at epoch 90, so 100 epochs captures convergence while avoiding the
flat tail. The inner training must use the repo's functional SmoothAdamW and
exact REPLAY path; the ordinary scale sweep used PyTorch AdamW only to choose
the backbone and horizon.

Compare all methods under the same initialization, deterministic batch order,
inner horizon, balanced validation-CE objective, and untouched final test:

1. Uniform unweighted baseline.
2. Label groups: one learned weight per CIFAR-100 class.
3. Each learned clustering technique: one learned weight per cluster.
4. Per-example MGD: one learned weight per training example.

Use enough outer MGD steps to demonstrate convergence rather than a single
metagradient update. Track objective/test balanced CE and accuracy, tier
accuracy, weight entropy/effective sample size, and generalization gap at every
outer step. Confirm the final MGD setting once on ViT-Small.

