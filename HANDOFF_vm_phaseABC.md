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
