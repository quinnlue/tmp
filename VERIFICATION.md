# Verification & Testing Strategy - Metagradients (PyTorch REPLAY, optimized SDPA optional)

> Companion to `README.md`.
> **Purpose:** establish, with high confidence and *before any scientific result is trusted*, that:
> 1. the functional PyTorch inner loop computes the correct metagradient `∇_θ φ`;
> 2. lazy k-ary REPLAY matches plain unrolled autograd; and
> 3. any optimized attention backend adopted later matches PyTorch math attention through second
>    order and end-to-end.

---

## 0. Guiding principle - bootstrap trust from the simplest reference

The mathematical reference is a tiny, fully unrolled PyTorch training trajectory using:

- explicit parameter and optimizer-state tensors;
- `torch.func.functional_call` for model evaluation;
- `torch.autograd.grad(..., create_graph=True)` for differentiable inner-loop gradients; and
- PyTorch **math scaled-dot-product attention (SDPA)**.

The authoritative inner optimizer is **Smooth AdamW**, not `torch.optim.AdamW`. Its parameter
recurrence is:

```text
p_next = (1 - learning_rate * weight_decay) * p
         - learning_rate * m_hat / sqrt(v_hat + epsilon^2)
```

Putting epsilon inside the square root avoids a singular derivative at zero second moment during
backward-over-backward. This choice is part of the training algorithm and is verified directly; it
is not intended to numerically match PyTorch AdamW.

Math SDPA is the intended attention oracle because it follows the unfused reference formulation and
is expected to support higher-order autograd; L1 explicitly verifies that assumption. Optimized SDPA
backends such as FlashAttention, memory-efficient attention, and cuDNN attention are optional. They
are adopted only if profiling justifies them and they match the math backend in our exact first- and
second-order configuration.

The inner training step must be a functional state transition:

```text
(params_t, adam_state_t, step_t, θ) -> (params_{t+1}, adam_state_{t+1}, step_{t+1})
```

No in-place mutation of model parameters or optimizer state is allowed in the differentiable path.
Do not rely on ordinary `optimizer.step()` as the reference implementation; implement Smooth AdamW
as tensor operations so every dependency is explicit and testable.

> **REPLAY rule:** retrieve optimizer states through a lazy balanced k-ary tree and propagate the
> reverse adjoint one training step at a time. REPLAY is intended to change only the memory schedule,
> but that equivalence depends on pure deterministic state transitions. Therefore recursive REPLAY
> must numerically match the unrolled reference before it is trusted.

---

## 1. Execution environments

| Role | Machine | Used for | Precision |
|---|---|---|---|
| **Math oracle** | laptop CPU | L1-L3 + GATE ground truth | **float64** |
| **GPU dev / smoke** | laptop RTX 4050 (6 GB) | small PyTorch CUDA runs and attention-backend checks | fp32 / tf32 |
| **Authoritative GPU** | remote Linux VM, A100/H100 | full training, GATE-on-GPU, scale checks | fp32 / tf32 |

Notes:

- Keep the strict numerical oracle on CPU in float64. Consumer-GPU fp64 is too slow for this role.
- Use Python 3.11 or 3.12 in a dedicated micromamba environment.
- PyTorch CUDA works natively on Windows; WSL2 is optional for local GPU development.
- Start without custom CUDA/Triton kernels. Add an optimized attention backend only after the pure
  PyTorch path is correct and profiling shows attention double-backward is a bottleneck.

---

## 2. Architecture invariants that make verification possible

### Functional training state

The differentiable inner loop receives and returns all mutable training state explicitly:

- named model parameters;
- model buffers, if any;
- Adam first moments, second moments, and step count;
- data/mask step index; and
- cluster logits `θ`.

Use `torch.func.functional_call` to evaluate the module with explicit parameters and buffers. Avoid
BatchNorm-style mutable running statistics in the initial model.

### Swappable attention

All model code calls attention through one interface. The backend selector changes only attention:

- `math`: PyTorch SDPA forced to `SDPBackend.MATH`; the correctness oracle.
- `optimized`: a selected PyTorch SDPA backend, enabled only after verification.
- `sigmoid`: optional custom fallback if softmax double-backward is unusable.

Nothing else in the model, training step, or replay loop may depend on the selected backend.

---

## 3. Where backward-over-backward enters

A training step is:

```text
g_t = ∇_{params_t} L_t(params_t; w)
s_{t+1} = smooth_adamw_update(s_t, g_t)
```

The inner gradient `g_t` is created with `create_graph=True`. The metagradient `∇_θ φ` differentiates
through all `T` inner gradients and optimizer updates, producing a backward-over-backward
computation.

PyTorch autograd performs each step-wise VJP. REPLAY is the memory schedule that regenerates
optimizer states instead of storing every intermediate state. Attention backends are a separate systems
choice and are tested independently.

## 3a. How REPLAY is implemented - lazy k-ary reverse traversal

The repository keeps two trajectory implementations:

| Scheme | Memory | Extra compute | Use |
|---|---|---|---|
| Store-all unrolled autograd | O(T) | none | L2 mathematical reference |
| **Lazy k-ary REPLAY** | O(k log_k(T)) optimizer states | O(log_k(T)) trajectory reruns | production replay implementation |

The recursive engine:

1. runs the trajectory once to obtain the final state;
2. recursively materializes balanced child-root states only when reverse traversal needs them;
3. yields states from `s_T` through `s_0`, deleting each child set after traversal; and
4. applies one differentiable inner step per yielded state to propagate the state adjoint and
   accumulate the metagradient.

Randomness should not come from global RNG state: the MAE mask is a pure function of
`(image_id, step)`. This makes recomputation directly testable and avoids silently replaying a
different training function.

---

## 4. Verification ladder

Each rung depends only on already-passed earlier rungs.

### L1 - one weighted MAE step is differentiable with respect to `θ` *(CPU, fp64)*

- **Goal:** verify the weighting, temperature, inner gradient, and functional Smooth AdamW update.
- **Oracle:** `torch.autograd.gradcheck`, `torch.autograd.gradgradcheck`, and central finite
  differences on a tiny double-precision configuration.
- **Catches:** weighting on the wrong axis, detached weights, in-place mutation, incorrect optimizer
  state, and unsupported second-order operations.

### L2 - unrolled T-step metagradient is the ground truth *(CPU, fp64)*

- **Goal:** compute `g_ref = ∇_θ (φ ∘ train_unrolled)(θ)` using plain unrolled autograd.
- **Independent optimizer oracle:** manually propagate a scalar parameter, Smooth AdamW moments, and
  the sensitivities of all three with respect to a scalar metaparameter; compare final state,
  objective, and metagradient to autograd at `rtol <= 1e-10`, `atol <= 1e-12`.
- **Oracle for `g_ref`:** full-coordinate central differences with Richardson extrapolation, plus
  deterministic zero-sum directional derivatives.
- **Authoritative config:** CPU float64; `T = 8`; 8x8 images with 4x4 patches; encoder/decoder
  dimension 8; math SDPA; batch size 4; separate two-image objective batch; Smooth AdamW with
  `learning_rate = 2e-3`, `betas = (0.8, 0.9)`, `epsilon = 1e-4`, and `weight_decay = 0.03`;
  softmax temperature `0.7`; nontrivial logits.
- **Granularities:** run the complete check for both two equal-sized clusters and four stable
  per-sample groups. The two representations must also agree when per-sample logits are tied by
  cluster.
- **Tolerance:** reverse-mode metagradient versus Richardson finite difference at `rtol <= 1e-4`,
  `atol <= 1e-7`, with matching signs for coordinates whose gradient magnitude exceeds `1e-6`.
- **Structural invariants:** softmax shift invariance, zero-sum metagradient, cluster-gradient equals
  the sum of corresponding tied sample gradients, graph-mode value equivalence, determinism, and
  no mutation of the initial state.

### L3 - recursive REPLAY equals unrolled autograd *(CPU fp64, then GPU/fp32)*

- **Goal:** confirm lazy k-ary state retrieval and the custom step-wise adjoint leave the metagradient unchanged.
- **Config:** exercise branching factors 2, 3, and larger than `T`, functional Smooth AdamW,
  deterministic per-step masks, at least two clusters, and nontrivial `θ`.
- **Metric/tolerance:** mean-relative error approximately `<= 1e-9` in fp64 and `<= 1e-4` in fp32,
  plus a small maximum absolute error.
- **Failure means:** tree boundaries, explicit state, recomputation, the custom adjoint, or a higher-order operator
  is wrong. Do not continue to scientific experiments.

The custom k-ary autograd function is verified with the adjoint dot-product test
`⟨v, J u⟩ = ⟨Jᵀ v, u⟩`. Differentiating through its backward again is explicitly unsupported.

### GATE - deterministic replay *(CPU exact; also GPU/VM)*

This is a correctness requirement, not general hygiene. REPLAY regenerates optimizer states during
backward; a different regeneration yields the gradient of a different training trajectory.

Checks:

1. Two same-seed training runs produce identical parameter trajectories and `φ`.
2. Every state regenerated by lazy-tree traversal reproduces the originally recorded state.
3. The MAE mask is a pure function of `(image_id, step)` and reproduces exactly.
4. The data order and batch composition reproduce exactly.

GPU configuration is centralized in `determinism.configure_replay_determinism(seed, tf32=...)`,
which sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, seeds Python/NumPy/PyTorch,
and applies:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

Set all Python, NumPy, CPU, and CUDA seeds. Treat any deterministic-algorithm exception as an
unsupported operation to replace or explicitly investigate.

### L0 - optimized attention backend matches math SDPA *(GPU; only if adopting one)*

- **Goal:** verify that the selected optimized backend supports and matches double-backward.
- **Cases:** head dimension 64, `causal=False`, actual encoder/decoder sequence lengths, fp32 and
  tf32 settings, and the exact masks used by MAE.
- **Oracle:** math SDPA, comparing forward values, first-order gradients, and second-order
  vector-Jacobian products.
- **Metric/tolerance:** mean-relative and max-absolute error; start with `<= 1e-3` for fp32
  second-order results and loosen only with evidence.
- **Failure policy:** keep math SDPA. An optimized backend that lacks double-backward is not usable in
  the differentiable inner loop.

### L4 - REPLAY with optimized attention equals REPLAY with math SDPA *(GPU; optional)*

- **Goal:** swapping only the attention backend leaves the end-to-end metagradient usable.
- **Metric:** per-coordinate sign agreement for non-near-zero coordinates and cosine similarity
  greater than `0.99`; also report magnitude error.
- **Reason:** the meta-update consumes `sign(g)`, so directional agreement is the load-bearing
  property.

### L6 - no scale-only bugs *(authoritative VM, real scale)*

- **Goal:** catch memory, precision, masking, and backend failures that appear only at realistic
  sequence lengths, model size, or trajectory length.
- **Setup:** central-difference a handful of `θ` coordinates or two to three random directions
  against the production engine.
- **Metric:** finite-difference agreement for math SDPA; sign and cosine agreement if an optimized
  backend is in the loop.

---

## 5. Optimized-attention policy

Math SDPA is the default until profiling demonstrates that attention double-backward materially
limits runtime or memory. Before selecting an optimized backend:

- prove it supports backward-over-backward for bidirectional attention;
- test head dimension 64 and actual sequence lengths;
- test the intended fp32/tf32 settings;
- run L0, L4, GATE, and L6; and
- retain a configuration switch that forces math SDPA.

Do not assume that a backend supporting ordinary training backward also supports second-order
autograd.

## 6. Determinism contract

Fixed data order and batch composition · MAE mask is a pure function of `(image_id, step)` · dropout
off · no stochastic augmentation · explicit optimizer state · no in-place mutation in the
differentiable path · deterministic PyTorch algorithms enabled · deterministic cuDNN settings · no
uninvestigated nondeterministic reductions/scatters.

## 7. Tolerances at a glance

| Rung | Compares | Precision | Primary metric | Initial bar |
|---|---|---|---|---|
| L1 | step vs gradcheck/finite difference | fp64 | relative | approximately 1e-6 |
| L2 | unrolled AD vs Richardson finite difference on `θ` | fp64 | relative + absolute | 1e-4 + 1e-7 |
| L3 | recursive REPLAY vs unrolled AD | fp64 / fp32 | relative + max-abs | 1e-9 / 1e-4 |
| GATE | replayed vs recorded states | exact / fp32 | equality / tight tolerance | identical trajectory |
| L0 | optimized vs math SDPA | fp32 / tf32 | relative + max-abs | approximately 1e-3 at order 2 |
| L4 | optimized vs math SDPA in REPLAY | fp32 / tf32 | sign + cosine | 100% sign / cosine > 0.99 |
| L6 | production engine vs finite difference | fp32 / tf32 | sign + cosine | as L4 |

## 8. Test organization

- Use `pytest`.
- Mark CPU oracle tests with `@pytest.mark.cpu`: L1-L3 and GATE-CPU.
- Mark CUDA tests with `@pytest.mark.gpu`: GATE-GPU, L0, L4, and L6.
- Keep the paper-faithful count baseline isolated in `tests/test_paper_mgd.py`, including
  zero-surrogate equivalence, finite differences, replay equivalence, count-update invariants, and
  a tiny end-to-end curation sanity check.
- Gate every merge on the small CPU oracle suite.
- Run the GPU suite on the local RTX 4050 where it fits and on the authoritative A100/H100 before
  scientific runs.

## 9. Definition of done before trusting scientific results

1. L1 and L2 analytic, gradcheck, finite-difference, and structural-invariant tests pass.
2. L3 recursive replay matches unrolled autograd with functional Smooth AdamW and deterministic
   masks.
3. GATE passes on CPU and on the authoritative GPU configuration.
4. Metasmoothness is measured and acceptable, or documented interventions are applied.
5. If optimized attention is adopted, L0, L4, GATE, and L6 pass.
6. Only then are H1-H4 from `README.md` measured.

## 10. Open risks

- PyTorch operators or optimized attention backends may lack second-order autograd support.
- GPU deterministic modes may reject an operation or reduce performance.
- Functional Smooth AdamW, lazy tree traversal, and the step-wise adjoint are easy to wire incorrectly.
- Store-all unrolled autograd may only fit tiny reference configurations.
- The 6 GB laptop GPU will require a small model and short `T`; real runs belong on the A100/H100.
- Softmax attention double-backward may remain the dominant runtime even with recursive REPLAY.
