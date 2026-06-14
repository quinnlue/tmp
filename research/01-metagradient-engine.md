# The Metagradient Engine

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

## Question/Goal

Provide a from-scratch, functional PyTorch implementation of metagradient descent (`∇_z φ(A(z))`, Engstrom et al. 2025, arXiv:2503.13751): a differentiable inner training loop `A`, a low-dimensional data-weight reparameterization `z`, and a memory-efficient exact metagradient backend (REPLAY). This engine is the shared foundation reused by every experiment in the project (ResNet-9/CIFAR-10 metasmoothness work, ViT metasmoothness work, per-cluster curation).

## Setup

- **Inner training is functional**: model parameters, optimizer state, and step counters are explicit tensors. Model evaluation uses `torch.func.functional_call`; inner gradients use `torch.autograd.grad(..., create_graph=True)`. No in-place mutation of parameters or optimizer state in the differentiable path.
- **Math SDPA is the attention correctness oracle** for any transformer model in this engine, since it follows the unfused reference formulation and supports the required higher-order autograd; optimized SDPA backends (FlashAttention, memory-efficient, cuDNN) are optional and must independently verify first- and second-order agreement before adoption.
- Bit-determinism is required end-to-end (fixed data order, fixed batch composition, seeded masks/augmentation as a pure function of `(image_id, step)`, dropout off, fp32/tf32).

## Method

### Functional inner optimizer: Smooth AdamW

`functional_train.py` defines `SmoothAdamWConfig` (`learning_rate`, `betas`, `eps`, `weight_decay`) and `TrainState` (`parameters`, `buffers`, `first_moments`, `second_moments`, `step`), all explicit tensors/dataclasses.

`functional_smooth_adamw(state, gradients, config)` implements the parameter recurrence:

```text
p_next = (1 - lr * weight_decay) * p
         - lr * m_hat / sqrt(v_hat + eps**2)
```

This is **intentionally different from `torch.optim.AdamW`**: epsilon is **inside** the square root. This avoids a singular derivative at zero second moment during backward-over-backward — it is "Smooth AdamW," chosen specifically to keep the inner training step smooth for higher-order differentiation, and is verified directly rather than benchmarked against ordinary AdamW.

`weighted_inner_step(model, state, images, labels, group_ids, logits, base_group_masses, optimizer_config, temperature, create_graph)`:
1. Evaluates the model functionally on `images` to get predictions.
2. Computes `per_example_cross_entropy_loss(predictions, labels)`.
3. Reweights it via `weighted_example_loss` (see below) using `logits`, `group_ids`, `base_group_masses`, `temperature`.
4. Takes `torch.autograd.grad(loss, params, create_graph=create_graph)` and applies `functional_smooth_adamw` to produce the next `TrainState`.

The full inner step is a pure function `(params_t, adam_state_t, step_t, θ) -> (params_{t+1}, adam_state_{t+1}, step_{t+1})`.

### Persistent-softmax / group-softmax data-weight reparameterization

`weighting.py` provides the loss-reweighting primitives:

- `group_masses(logits, temperature)`: `softmax(logits / temperature, dim=0)` over a 1D tensor of group logits — one entry per group (cluster/class/example).
- `example_multipliers(logits, group_ids, base_group_masses, temperature)`:
  ```python
  target_group_masses = group_masses(logits, temperature)
  return target_group_masses[group_ids] / base_group_masses[group_ids]
  ```
  i.e. example `i` (belonging to `group_ids[i]`) is multiplied by `softmax(logits)[group_i] / base[group_i]`. `base_group_masses` must be positive and sum to one.
- `weighted_example_loss(per_example_loss, logits, group_ids, base_group_masses, temperature)`: returns `(per_example_loss * multipliers).mean()`.

**`z = 0 ⇒ uniform identity`**: when `logits = 0` and `base_group_masses` is itself the uniform/base distribution (`softmax(0) = uniform`), `target_group_masses == base_group_masses` for every group, so every multiplier is 1 and the weighted loss reduces to the ordinary mean — exactly standard training.

More generally, the engine supports `effective_group_logits = z + T·log(base)` (`T` = temperature) as a way to generalize "z=0 ⇒ ordinary training" to **any** base distribution `base`: substituting this into `softmax(effective_group_logits / T)` recovers `base` at `z=0` (since `softmax(log(base)) = base` up to normalization), and `z` then perturbs away from that base — so the reparameterization can express curation relative to an arbitrary starting distribution, not only the uniform one.

### Cluster-basis / granularity reparameterization

A single `group_ids` vector (one entry per training example, `torch.long`, values in `[0, num_groups)`) maps each example to a group. The same `weighted_example_loss` machinery handles:
- **per-cluster / per-class** weighting: `group_ids` assigns each example to its class, `logits ∈ ℝ^k` (k classes/clusters).
- **per-example** weighting: `group_ids = arange(n)` (each example its own group), `logits ∈ ℝ^n`.

Per-cluster is the restriction of per-example to the subspace where weights are constant within a group (`z_example = z_cluster[group(example)]`) — one parameterization at different dimensionalities, letting the same loss-weighting and REPLAY code serve per-class, per-cluster, or per-example curation.

### Metagradient backends

| Backend | File | Memory | Behavior |
|---|---|---|---|
| Lazy k-ary REPLAY | `recursive_replay.py` (`recursive_replay_state`) | `O(k · log_k T)` optimizer states | Exact first-order metagradients via a lazy balanced k-ary tree reverse traversal; production backend. |
| Store-all unrolled autograd | `metagrad.py` (`train_unrolled`) | `O(T)` | Mathematical/ground-truth reference; OOMs past roughly 50 inner steps. |
| Paper-faithful count relaxation | `paper_mgd.py` | uses REPLAY internally | Algorithm 1 / Appendix C.2 baseline, isolated from the persistent cluster/sample softmax-weighting path. |

**`recursive_replay.recursive_replay_state`** (`recursive_replay.py`):
- `_lazy_reverse_states` performs the paper's lazy balanced k-ary tree reverse traversal: it runs the trajectory once to get the final state, recursively materializes balanced child-root checkpoints only when the reverse traversal needs them, yields states `s_T` through `s_0`, and deletes each child-checkpoint set after traversal.
- `_RecursiveReplayEngine.metagradient` then applies one differentiable inner step (`create_graph=True`) per yielded state, propagating the state adjoint backward and accumulating the metagradient via `torch.autograd.grad` with `materialize_grads=True`.
- Implemented as a custom `torch.autograd.Function` (`_RecursiveReplayState`) whose `forward` runs the trajectory (detached) and whose `backward` calls `engine.metagradient`.
- **Raises on higher-order differentiation**: `metagradient()` raises `RuntimeError("higher-order differentiation through recursive REPLAY is unsupported")` if `torch.is_grad_enabled()` is true when called — i.e. REPLAY gives exact first-order metagradients only; a second metagradient-of-metagradient pass through REPLAY is not supported.
- `ReplayCheckpointConfig` (`backend="memory"|"disk"`, `directory`, `interval_steps`) controls checkpoint storage: `"memory"` (default) keeps the fastest behavior; `"disk"` writes detached checkpoint states to ephemeral run directories, reloads on demand, and removes them after traversal — only lazy-tree child roots spill to disk; active state, initial/final states, adjoints, and the current differentiable step stay in memory. `interval_steps=N` writes coarse disk checkpoints every N steps and recursively replays each block with in-memory lazy-tree checkpoints (lower disk I/O at the cost of higher peak disk space and block-local memory).
- `branching_factor` (≥2) is the explicit memory/compute tradeoff knob.

**`metagrad.train_unrolled`** (`metagrad.py`):
- `InnerBatch` (images, labels, group_ids) / `ObjectiveBatch` (images, labels).
- `train_unrolled` loops `weighted_inner_step` over a fixed `trajectory: Sequence[InnerBatch]`, storing every intermediate `TrainState` for differentiation — the `O(T)` mathematical reference (L2 in the verification ladder).
- `classification_objective` evaluates unweighted held-out cross-entropy on an `ObjectiveBatch`.
- `unrolled_objective` composes the two: store-all inner training → held-out objective, differentiable end-to-end.

**`paper_mgd.py`** — the paper-faithful baseline, intentionally separate from the persistent cluster/sample softmax path:
- Non-negative integer `counts: Tensor` per candidate (`initialize_counts`), expanded into a deterministic fixed-budget trajectory (`count_schedule_indices`, `build_count_trajectory`) via deterministic shuffling.
- `paper_inner_step` takes one ordinary Smooth-AdamW step and, at a single chosen `perturbation_step`, injects a zero-initialized continuous `perturbations` tensor as `Σ perturbations[i] · ℓ(probe_i; θ)` added to the batch loss (scaled by `1/batch_size`) — the paper's single-step differentiable surrogate at `z=0`.
- `paper_recursive_replay_state` / `paper_replay_objective` / `paper_replay_metagradient` route this trajectory through `recursive_replay.recursive_replay_state` to get `∇_perturbations objective` (the metagradient w.r.t. the injected perturbations) via `torch.autograd.grad`.
- `paper_mgd_outer_step` ties it together: builds the count-induced trajectory, selects candidate coordinates (`bernoulli_coordinates` for `"projected_sign"`, `shard_coordinates` for `"fixed_budget_ranked"`), computes the metagradient, and applies one of two outer updates:
  - `projected_sign_update`: Algorithm 1's `counts ← max(0, counts - sign(g))`.
  - `fixed_budget_ranked_update`: Appendix C.2's fixed-budget ranked exchange (increment the lowest-gradient selected coordinates, decrement the highest-gradient ones with positive count, preserving total count).

## Determinism requirements

`determinism.py` provides `configure_replay_determinism(seed, *, tf32)`:
- Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before CUDA initialization** (raises if CUDA is already initialized with a different value).
- Seeds `random`, `numpy`, `torch`, and `torch.cuda.manual_seed_all`.
- Calls `torch.use_deterministic_algorithms(True, warn_only=False)`, sets `cudnn.benchmark = False`, `cudnn.deterministic = True`, and configures TF32 via `_set_tf32(tf32)` (using `torch.backends.fp32_precision`/`cuda.matmul.fp32_precision`/`cudnn.fp32_precision` where available, else the legacy `allow_tf32` flags).

`assert_replay_determinism(*, tf32, require_cuda=True)` raises `RuntimeError` listing any of: CUDA unavailable when required, deterministic algorithms disabled or warn-only, cuDNN benchmarking enabled, cuDNN deterministic mode disabled, wrong `CUBLAS_WORKSPACE_CONFIG`, or mismatched TF32 configuration.

**Why this matters**: REPLAY regenerates optimizer states during the backward pass by re-running the (deterministic) forward trajectory — a different regeneration computes the gradient of a *different* training trajectory, silently corrupting the metagradient. fp16/bf16/TF32 are faster and can be bit-deterministic, but they **flip the sign of the tiny Definition-1 metasmoothness second difference** and are imprecise enough to break exact metagradient agreement; the engine therefore requires fp32 with TF32 off for any finite-difference or REPLAY-correctness work.

## Reproduction

The verification ladder (`VERIFICATION.md`) establishes correctness before any scientific result is trusted, each rung depending on the previous:

- **L1** (CPU, fp64): one weighted-loss inner step is differentiable w.r.t. `θ` — `gradcheck`/`gradgradcheck`/finite differences on a tiny config. Catches wrong-axis weighting, detached weights, in-place mutation, bad optimizer state, unsupported second-order ops.
- **L2** (CPU, fp64): `train_unrolled`'s metagradient matches an independent manual-propagation oracle and full-coordinate Richardson-extrapolated central differences, at `T=8`, tiny 8×8/4×4-patch images, math SDPA, `lr=2e-3, betas=(0.8,0.9), eps=1e-4, weight_decay=0.03`, temperature `0.7`, for both two equal clusters and four per-sample groups (and their agreement when tied). Tolerances: `rtol<=1e-4, atol<=1e-7`.
- **L3** (CPU fp64 → GPU fp32): `recursive_replay_state` matches `train_unrolled` across branching factors 2, 3, and >T, ≥2 clusters, nontrivial `θ`. Tolerance: mean-relative error `<=1e-9` (fp64) / `<=1e-4` (fp32).
- **GATE** (CPU exact, GPU/VM): two same-seed runs bit-match; every lazy-tree-regenerated state reproduces the original; MAE/data masks are pure functions of `(image_id, step)`.
- **L0/L4/L6** (GPU, only if adopting optimized attention): optimized SDPA backend matches math SDPA through second order, then in full REPLAY, then in the production engine vs finite differences at scale.

Run the CPU oracle suite (`pytest`, `@pytest.mark.cpu`) on every merge; run the GPU suite (`@pytest.mark.gpu`) on the RTX 4050 where it fits and on the authoritative A100/H100 before scientific runs. Definition of done before trusting H1–H4-style scientific results: L1+L2 pass, L3 matches with functional Smooth AdamW and deterministic masks, GATE passes on CPU and the authoritative GPU, metasmoothness is measured and acceptable (or interventions applied), and L0/L4/L6 pass if optimized attention is in use.

## Gotchas

- Smooth AdamW's `eps` is **inside** the square root — do not compare it numerically to `torch.optim.AdamW`; it is a deliberately different, smoother recurrence for backward-over-backward.
- `recursive_replay_state` raises if called with `torch.is_grad_enabled() == True` — it provides exact first-order metagradients only, not differentiable-through-REPLAY higher-order metagradients.
- REPLAY's cost is dominated by trajectory length `T` and model size, **not** by the dimensionality of `z` — clustering does not make REPLAY cheaper; it only improves the statistical conditioning of the meta-optimization.
- `base_group_masses` must be positive and sum to one, and must match `logits` in shape/device; `group_ids` must be `torch.long` and index validly into `logits`.
- `CUBLAS_WORKSPACE_CONFIG` must be set before CUDA initializes — `configure_replay_determinism` raises if CUDA is already initialized with a different value.
- fp16/bf16/TF32 are faster but invalidate both metasmoothness finite differences and exact REPLAY metagradients — fp32 with TF32 off only.

## Source documents

- `README.md`
- `VERIFICATION.md`
- `HANDOFF_metasmoothness.md`
- `functional_train.py`
- `weighting.py`
- `recursive_replay.py`
- `metagrad.py`
- `paper_mgd.py`
- `determinism.py`
