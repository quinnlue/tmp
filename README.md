# Cluster-Reparameterized Metagradient Data Curation for MAE-ViT

> **Nature of this document.** This is a *design / intent* document for an exploratory research
> experiment — not an implementation plan and not publication-bound. Its job is to pin down the
> motivation, the background, the precise experimental setup, and the testing/evaluation strategy
> so that implementation (later) is unambiguous. Priorities, in order: **get feet wet with a real
> metagradient implementation → maximize interpretability → maximize efficiency.** Faithfulness to
> a realistic downstream pipeline is explicitly *not* a goal.

---

## 1. Context & Motivation

The paper **"Optimizing ML Training with Metagradient Descent"** (Engstrom, Ilyas, Chen, Feldmann,
Moses, Mądry; arXiv:2503.13751) treats *the training configuration itself* as something to optimize
by gradient descent. It defines the **metagradient** `∇_z f(z)` of a training function `f = φ ∘ A`,
where `z` are **metaparameters**, `A` is the training algorithm mapping `z` to trained weights `θ`,
and `φ` is an output function mapping `θ` to a scalar (e.g. a held-out loss). Their **REPLAY**
algorithm computes this metagradient exactly and at scale, and their **metasmoothness** framework
explains when such metagradients are usable.

One headline application is **data curation / selection** (§4.1–4.2): the metaparameters are
per-example data weights, and metagradient descent (MGD) reweights the training set to minimize a
target loss. It works, but the authors flag a clear failure mode in the instruction-tuning
experiment (§4.2):

> *"…our method improves validation loss over MGD steps … but also exhibits signs of overfitting.
> Given intuition from overparameterized learning, we might expect this behavior: we optimize a
> total of **270,679 'weights'** — each corresponding to a count for a datapoint — to minimize loss
> on only **a handful of test samples**."*

**The overfitting/high-variance is a dimensionality problem**: one free metaparameter per training
example, fit against a tiny objective set. **This project's core idea** is to *reparameterize the
curated distribution onto a low-dimensional basis* so there are far fewer metaparameters to overfit.

Working in a base, unfiltered, unweighted dataset `D`, there is some ideal reweighting `D'` that
maximizes downstream eval performance. Instead of approximating `D'` with one weight per sample, we
construct it as a weighted sum over **clusters** of `D`:

```
D'  ≈  w₀·d₀ + w₁·d₁ + … + w_k·d_k ,     Σ w_i = 1
```

where each `d_i` is a cluster of `D` — in the spirit of expressing a function by its series
expansion. The hypothesis is that this coarse basis retains most of the downstream gain while
sharply reducing the overfitting and variance seen with per-sample weights.

For interpretability we make the clusters **the class labels of a labelled dataset**. Labels are
used *only* to (a) define clusters and (b) select the held-out target-class eval set — the training
objective remains pure self-supervised reconstruction; labels never enter the loss.

---

## 2. Background (precise, as it bears on this project)

**2.1 Metagradient & the data-selection surrogate (§2.1, §4.1).**
The paper's data-selection metaparameters are non-negative counts `c ∈ ℤⁿ`. Because counts are
discrete, they use a **differentiable surrogate**: at a chosen step they add `Σ_i z_i·ℓ(x_i; θ)` to
the loss and take the metagradient at `z = 0`; then update counts via a *signed-gradient*
block-coordinate step `c ← c − sign(g) ⊙ m`, `m ~ Bernoulli(p)`. **Our "differentiable
loss-reweighting" is the same family of trick**, with two deliberate differences: the weight is
**tied across a whole cluster** (k parameters, not n), and it is applied as a **persistent
reweighting at every step** (it literally defines `D'`), not a single-step perturbation.

The repository also contains a separate, paper-faithful baseline in `paper_mgd.py`. It keeps
non-negative integer counts, builds a fixed-budget trajectory from repeated samples, injects a
zero-initialized continuous perturbation at one chosen inner step, and updates counts with either
Algorithm 1's projected signed step or Appendix C.2's fixed-budget ranked exchange. This baseline
is intentionally isolated from the persistent cluster/sample softmax-weighting APIs.

**2.2 REPLAY (§2.3).** Computes *exact* metagradients through iterative training using a lazy
k-ary-tree (segment-tree-like) reverse traversal: **`O(k·log_k T)` memory**, re-running the training
algorithm **`1 + log_k T` times** for a trajectory of length `T`. It requires a **bit-deterministic**
training loop (fixed data order, fixed augmentation, fixed all RNG) so that intermediate states can
be replayed rather than stored. The reverse pass is a **backward-over-backward** computation
(~2–3× the ops of a normal backward pass) and the paper's implementation requires **fp32 / tf32**.

> **Key consequence for this project:** REPLAY's cost is dominated by trajectory length and model
> size, **not** by the dimensionality of `z`. So clustering does **not** make the metagradient
> cheaper than the per-sample baseline — both pay the same REPLAY cost. What clustering buys is
> purely **statistical**: a smaller, better-posed meta-optimization with less overfitting and lower
> variance. This framing is central to how we evaluate success (§6).

**2.3 Metasmoothness (§3).** Applying REPLAY to a *standard* training routine often yields `±∞` or
useless metagradients because the metaparameter→metric map is non-smooth. The paper proposes an
**empirical metasmoothness metric** (Definition 2) computed from **three finite-difference calls** to
`A` along a direction, and a menu of interventions that helped for CNNs: **BatchNorm before
activations, average (not max) pooling, scaling down the last layer's output, network width, batch
size.** A ViT has none of BN/pooling, so this is a real hurdle (see §4.5).

**2.4 PyTorch and attention double-backward.** PyTorch supports the required higher-order
differentiation through `torch.autograd.grad(..., create_graph=True)` and functional model execution
through `torch.func.functional_call`. The default attention backend for this project is PyTorch's
**math scaled-dot-product attention (SDPA)**, because it follows the unfused reference formulation
and is the intended correctness oracle for the backward-over-backward path; its higher-order
behavior is still verified explicitly. Faster SDPA backends (FlashAttention, memory-efficient
attention, cuDNN attention) are **optional optimizations** and may be adopted only after their first-
and second-order gradients match the math backend in our exact configuration. The paper's
observation that softmax double-backward can be expensive remains relevant; sigmoid attention is a
possible custom fallback, not part of the initial implementation.

**2.5 Related work to position against (informational).** *DataRater: Meta-Learned Dataset Curation*
(arXiv:2505.17895) meta-learns a data-rating network via meta-gradients — a learned-function approach
to the same goal. Our cluster-basis is a deliberately simpler, more interpretable parameterization.

---

## 3. Research Question & Hypotheses

**RQ.** Does reparameterizing the curated distribution `D'` onto a low-dimensional **cluster basis**
(one weight per class) reduce the overfitting and variance of metagradient data curation, while
retaining most of the downstream benefit, relative to **per-sample** metagradient curation?

- **H1 (less overfitting).** Cluster-MGD has a smaller *generalization gap* — (held-out test
  reconstruction loss) − (objective-set reconstruction loss) — than per-sample MGD over meta-steps.
- **H2 (lower variance).** The converged weights and the held-out reconstruction loss vary less
  across seeds / objective-set resamples for cluster-MGD than for per-sample MGD.
- **H3 (retained benefit).** Cluster-MGD still improves held-out target reconstruction over the
  uniform/base-`D` distribution, at competitive levels.
- **H4 (interpretability).** The converged cluster weights are semantically sensible — the target's
  own class and visually-related classes (e.g. wolves/foxes for a dog target) are upweighted;
  orthogonal classes (e.g. vehicles) are downweighted.

---

## 4. Experimental Design

### 4.1 Data, clusters, target, and splits
- **Base dataset `D`:** an **even (class-balanced) subset of ImageNet-1k**, **medium scale ≈ 50–100
  classes**. Class set deliberately spans three tiers relative to the target: the **target class
  itself**, **near-neighbors** (visually/semantically related), and **orthogonal distractors**.
- **Clusters `d_i` = ImageNet class labels** (one cluster per class), `k ≈ 50–100`.
- **Target class `c*`:** **configurable** (the "dogs/wolves" story is just the running illustration).
  `c*` **is present in the training pool** as one of the clusters (so the method *can* upweight it —
  a built-in sanity check), **but its evaluation images are held out** — strict train/eval disjointness
  for the target class, no leakage.
- **Four disjoint splits:**
  1. **Training pool** — the clustered images actually trained on (includes a train split of `c*`).
  2. **Objective set `S_obj`** — held-out `c*` images that define `φ` and **drive the metagradient**.
  3. **Meta-validation set `S_val`** — separate held-out `c*` images for choosing meta-hyperparameters
     (temperature `τ`, meta-LR `α`, number of meta-steps) and early stopping.
  4. **Test set `S_test`** — separate held-out `c*` images for **final reported numbers only**;
     never seen by the metagradient or by meta-hyperparameter selection.
  > The `S_obj` vs `S_test` distinction is what makes the overfitting claim (H1) measurable and
  > honest: the "eval that is never seen" must be `S_test`, *not* the objective set that `φ` is
  > computed on.
- **Scale knobs to fix at implementation:** images/class (≈ few hundred → `n` ≈ 15k–30k total),
  image resolution, patch size.

### 4.2 Model & deterministic training
- **MAE-ViT**, **ViT-Tiny** backbone first (ViT-Small as a stretch), pure self-supervised
  **masked-patch reconstruction** loss. Decoder per standard MAE.
- **Output function `φ`** = mean masked-reconstruction MSE over `S_obj` (for the metagradient) /
  `S_test` (for reporting), using **fixed, deterministic eval masks** so `φ` is low-variance and
  smooth.
- **Bit-determinism (required by REPLAY):** fixed data ordering and fixed batch composition;
  **MAE mask seeded per `(image, step)`**; **no data augmentation** beyond that mask; **dropout off**;
  **fp32/tf32**; call `configure_replay_determinism(seed, tf32=...)` before CUDA initialization to
  set seeds, deterministic PyTorch/CUDA/cuDNN settings, and the required cuBLAS workspace mode.

### 4.3 Constructing `D'` — the cluster reweighting
- **Parameterization:** unconstrained logits `θ ∈ ℝᵏ`, weights `w = softmax(θ / τ)` with a **fixed
  temperature `τ`** (tuned on `S_val`; a `τ` ablation is in scope). `Σ w_i = 1` automatically; the
  map is smooth (good for metagradients).
- **Application:** each sample's reconstruction loss is multiplied by `w_{cluster(sample)}`, **at every
  training step** — a persistent reweighting that *is* `D'`. (Contrast with the paper's single-step
  surrogate.)
- **Initialization:** `θ = 0` → uniform `w = 1/k` → the base distribution `D` (mirrors the paper's
  "start = the standard training algorithm").

### 4.4 Metagradient & meta-optimization
- **Flow:** `θ → w = softmax(θ/τ) → (per-cluster loss weights) → T-step deterministic MAE training →
  trained weights → φ on S_obj`. The **metagradient `∇_θ φ`** is computed by **REPLAY**, with
  PyTorch autograd providing the backward-over-backward path. Training parameters and optimizer
  state are explicit tensors, and each update is a functional, differentiable state transition.
- **Meta-update (paper-faithful):** `θ ← θ − α · sign(∇_θ φ)`. Meta-LR `α` and the number of
  meta-steps tuned/early-stopped on `S_val`.
- **Trajectory length `T`:** start modest (order hundreds of steps) given recursive replay
  recomputation and backward-over-backward cost on a single A100/H100; treat as a tunable.

### 4.5 Metasmoothness handling (paper-faithful, pragmatic)
Because this is an early experiment, we **follow the paper's existing playbook rather than inventing a
ViT-specific smoothness taxonomy**:
- **Measure** metasmoothness with the paper's finite-difference metric (Definition 2) before trusting
  metagradients.
- **Intervene only as the paper prescribes**, mapping its CNN interventions to the ViT where they
  transfer (most directly: **scaling down the reconstruction-head / last-layer output**; norm
  behavior; smaller meta-LR; fp32). 
- The **softmax-vs-sigmoid attention** smoothness angle is **noted as a fallback lever** if softmax
  double-backward proves ill-behaved. It would require a custom PyTorch attention implementation and
  is *not* a planned sweep.

---

## 5. Baselines
1. **Uniform / base `D`** (`θ = 0`, no curation) — the floor for H3.
2. **Paper-faithful count MGD:** one non-negative integer count per candidate, with a one-step
   continuous surrogate and signed count updates. Both projected-sign and fixed-budget ranked
   policies are available for comparison.
3. **Per-sample persistent-softmax MGD — same mechanism, sample granularity** (the key contrast for H1/H2): the
   *identical* softmax-loss-weighting training loop and REPLAY metagradient, but with **one weight per
   sample** (`θ ∈ ℝⁿ`) instead of per cluster (`θ ∈ ℝᵏ`). This isolates the **single variable =
   granularity (k vs n)**, but is a smooth proxy rather than the paper's count-based method.
4. *(Optional reference)* **Train-on-target-only "oracle"** — an upper-ish bound on `φ` from training
   only on `c*` (and/or near-neighbors), for context.

---

## 6. Evaluation & Testing Strategy

### 6.1 Scientific metrics (the experiment's verdict)
Measured for **cluster-MGD vs per-sample MGD vs uniform**, across the meta-optimization trajectory:
- **Generalization gap (H1):** `φ(S_test) − φ(S_obj)` over meta-steps. Expect cluster-MGD's gap to be
  smaller and to grow more slowly than per-sample MGD's.
- **Variance (H2):** spread of final `φ(S_test)` **and** of the converged weight vector across
  **N ≈ 5 seeds** and across **objective-set resamples**. Expect cluster-MGD lower.
- **Downstream gain (H3):** `φ(S_test)` of cluster-MGD vs uniform (and vs per-sample at its best
  pre-overfitting point).
- **Interpretability (H4):** inspect the converged `w` — is `c*` and are its near-neighbors
  upweighted, orthogonal classes downweighted? Optionally correlate `w` with an embedding-similarity-
  to-`c*` prior.

### 6.2 Software / correctness tests (must pass before trusting any result)
- **Finite-difference gradcheck** of the metagradient: compare REPLAY's `∇_θ φ` against numerical
  perturbation of `θ` on a tiny config. Doubles as a metasmoothness probe.
- **Determinism / replay-consistency:** two runs with the same seed bit-match; a replayed
  intermediate state equals the originally-computed one (the property REPLAY depends on).
- **Overfit-a-target sanity:** with `T` tiny and the target trivially separable, the metagradient
  should drive `w` toward the obviously-correct cluster(s).
- **Runtime/memory budget check:** confirm lazy k-ary replay recomputation ×
  backward-over-backward fit the A100/H100 at the chosen `T`, model size, and batch.

### 6.3 Success criteria
- **Primary:** gradcheck + determinism tests pass, and the cluster-MGD vs per-sample comparison
  produces a **clear, interpretable verdict on H1–H4** (whether confirming or refuting them).
- **Secondary:** converged cluster weights are human-legible (H4), and a working functional-PyTorch
  REPLAY loop exists end-to-end (the "feet wet" goal).

---

## 7. Scope, Scale & Compute
- **Hardware:** single **A100/H100 (40–80GB)**.
- **Scale:** medium, **~50–100 even ImageNet-1k classes**, **ViT-Tiny** first (ViT-Small as stretch),
  modest `T` to keep full REPLAY tractable.
- **Deliverable of this phase:** **this document only.** Implementation follows after approval.

---

## 8. Implementation Approach (high-level only — not a build plan)
- **From-scratch PyTorch MAE-ViT**, written functionally for the differentiable inner training loop:
  model parameters and optimizer state are explicit tensors, model evaluation uses
  `torch.func.functional_call`, and inner gradients use
  `torch.autograd.grad(..., create_graph=True)`.
- Components: deterministic data pipeline (seeded masks, fixed order) · MAE-ViT model · per-cluster
  softmax-temperature loss weighting · lazy k-ary **REPLAY** metagradient engine with an explicit
  `branching_factor` memory/compute tradeoff · signed-gradient meta-optimizer · the
  eval/test harness from §6.
- The paper-faithful count baseline is implemented separately in `paper_mgd.py`; it reuses the MAE,
  functional Smooth AdamW state, and held-out objective without changing the persistent-weighting
  path.
- Lazy-tree checkpoints use `ReplayCheckpointConfig`. The default `backend="memory"` preserves the
  fastest existing behavior. `backend="disk"` writes detached checkpoint states to unique,
  ephemeral run directories under `directory` or the system temporary directory, reloads them on
  demand, and removes them after traversal. Disk mode spills only lazy-tree child roots; the active
  state, initial/final states, adjoints, and current differentiable step remain in memory. Pass the
  config as `checkpoint_config=` to replay APIs or set it on `PaperMGDConfig`. Setting
  `interval_steps=N` writes coarse disk checkpoints every N steps, then recursively replays each
  block using in-memory lazy-tree checkpoints; this increases peak disk space and block-local
  memory in exchange for much lower disk write/read volume.
- Attention is routed through one swappable interface. PyTorch math SDPA is the default and
  correctness oracle; optimized SDPA backends are optional and must pass the verification ladder.
- Greenfield repository (currently documentation-only); no existing model code to reuse.

---

## 9. Risks & Open Questions
- **Metasmoothness of a ViT** is unproven; if metagradients are `±∞`/useless, the paper-prescribed
  interventions (esp. last-layer scaling) and, as a fallback, **sigmoid attention** are the levers.
- **Softmax double-backward cost** may dominate runtime → test optimized PyTorch SDPA backends, then
  consider a custom sigmoid-attention fallback only if necessary.
- **REPLAY engineering** (functional optimizer state, correct lazy k-ary traversal, deterministic
  recomputation, and fp32 stability) is the bulk of the build risk.
- **`φ` stability:** fixed eval masks and averaging over `S_obj` chosen to keep `φ` smooth.
- **Compute:** backward-over-backward plus recursive replay recomputation can blow up with `T`; start
  small.
- **Open knobs to finalize at implementation:** exact class list (target + near + orthogonal),
  images/class, resolution/patch size, mask ratio, `T`, `τ`, `α`, N seeds.

## 10. Recorded Defaults / Assumptions
- `φ` = masked-recon MSE on held-out target-class images (fixed eval masks); labels never in the loss.
- `θ` init 0 (uniform start); softmax **with fixed temperature `τ`** (tuned on `S_val`, `τ` ablation in
  scope); cluster weight applied **every step**.
- Meta-update = **signed gradient** `θ ← θ − α·sign(∇_θ φ)`.
- Target class **in the training pool**, eval images **held out** (no leakage); target is **configurable**.
- No augmentation beyond seeded MAE mask; dropout off; fp32/tf32; full bit-determinism.
- N ≈ 5 seeds for variance (raisable).

## 11. Verification (how we'll know it works)
1. **Correctness:** finite-difference gradcheck of `∇_θ φ` matches REPLAY; determinism/replay-consistency
   bit-match tests pass.
2. **Smoothness:** Definition-2 metasmoothness measured and acceptable (or interventions applied until it
   is).
3. **Scientific:** cluster-MGD vs per-sample MGD vs uniform run to convergence; H1 (gap), H2 (variance),
   H3 (downstream gain), H4 (interpretable weights) each get a clear yes/no with plots over meta-steps.
4. **Systems:** the full PyTorch REPLAY loop runs end-to-end within the A100/H100 budget at the
   chosen scale.
