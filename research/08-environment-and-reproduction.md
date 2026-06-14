# Environment & Reproduction

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

Practical guide to the environments, data caches, run commands, and the
engineering gotchas that recur across every experiment. The per-experiment docs
(02–07) each have their own `## Reproduction`; this collects the shared parts.

## Environments

**Laptop (dev + CPU tests + smoke runs)**
- Repo root: `C:\ml\dataset-curation\meta-grad-descent-w-clustering`
- Interpreter: `C:/Users/luequ/micromamba/envs/torch311/python.exe` (torch 2.8.0+cu128, RTX 4050 Laptop GPU)
- Run the unit tests here before any GPU run: `… -m pytest -q` (the `tests/` suite is CPU-only / fast).

**GPU VMs (heavy runs)** — all ephemeral vast.ai A40 (46 GB) boxes,
`torch 2.12.0+cu126`, interpreter `/venv/main/bin/python` (a.k.a. activate
`/venv/main`). They differ per workstream because each was a fresh box:

| Workstream | VM (at time of run) | Repo path on VM |
|---|---|---|
| ResNet-9 Phases A/B/C ([02](02-metasmoothness-resnet9.md)) | `ssh -p 4462 root@160.250.70.29` (B/C), `ssh -p 4146 root@160.250.70.25` (deep A) | `/workspace/tmp` (from `github.com/quinnlue/tmp.git`) |
| CIFAR-10 granularity ([05](05-mgd-granularity-cifar10.md)) | `ssh -p 4462 root@160.250.70.29` | `/workspace/autoresearch-mgd` |
| ImageNet-LT granularity ([06](06-mgd-granularity-imagenetlt.md)) | `ssh root@160.250.70.25` | `/workspace/imagenet-lt-autoresearch` |
| CIFAR100-LT ViT ([07](07-mgd-granularity-cifar100lt-vit.md)) | `ssh -p 4191 root@160.250.70.25` | `/workspace/meta-grad-descent-w-clustering` |

Fresh-box setup (~5 min): clone/scp the repo, `pip install -q pytest pyarrow
pillow matplotlib`, build the data cache (below), then `pytest -q` to validate.
The VM repos are not always git checkouts — `scp` the current code over before
running, and confirm the engine files (`recursive_replay.py`, `determinism.py`)
are present.

## Data caches

- **CIFAR-10:** `python -m tools.build_cifar_cache ./data` → `data/cifar10_{train,test}.npz`
  (pulls the HuggingFace parquet mirror; `load_cifar_subset` fast-paths the npz).
  The torchvision mirror (cs.toronto.edu) is throttled to ~30 KB/s on the VMs —
  do **not** let torchvision download.
- **CIFAR100-LT:** `_cifar100_lt.load_cifar100_lt("./data/cifar100-lt", config="r-100")`
  caches `data/cifar100-lt/cifar100_lt_r-100_{train,test}.npz` from
  `tomas-gajarsky/cifar100-lt` on first use.
- **ImageNet-LT:** `inria-chile/imagenet-lt-v2` parquet + cached DINOv2 feature
  artifacts (`dinov2_{train,val,test}.pt`) and the hierarchical cluster basis
  (`dinov2_hkmeans_k1000.*`); see [06](06-mgd-granularity-imagenetlt.md).

## Run commands by experiment

- **Metasmoothness, ResNet-9** ([02](02-metasmoothness-resnet9.md)): `experiments.cifar10.metasmooth_vm`
  (Phase A, env-var config), `experiments.cifar10.metasmooth_train` (Phase B), `experiments.cifar10.metasmooth_mgd`
  (Phase C per-cluster MGD; env `METHOD/N_POOL/N_OBJ/INNER_EPOCHS/META_STEPS/TARGET_CLASSES/…`),
  render with `experiments.cifar10.render_metasmooth_vm`.
- **Metasmoothness, ViT** ([03](03-metasmoothness-vit.md)): `experiments.cifar10.vit_metasmooth`
  (Phase A), `experiments.cifar10.vit_train` (Phase B), `experiments.cifar10.vit_per_example`,
  render with `experiments.cifar10.render_vit`. Menu in `experiments.cifar10.vit_menu`.
- **CIFAR100-LT scale study** ([04](04-cifar100lt-vit-scale-study.md)):
  `experiments.cifar100_lt.vit_scale_sweep`.
- **CIFAR-10 granularity** ([05](05-mgd-granularity-cifar10.md)):
  `python -m experiments.cifar10.compare_mgd_granularity …` + `experiments.cifar10.summarize_mgd_granularity`.
- **ImageNet-LT granularity** ([06](06-mgd-granularity-imagenetlt.md)):
  `python -m experiments.imagenet_lt.compare_mgd …` + `experiments.imagenet_lt.summarize_mgd`;
  feature/cluster prep via `experiments.imagenet_lt.extract_embeddings` / `experiments.imagenet_lt.build_hkmeans`.
- **CIFAR100-LT ViT granularity** ([07](07-mgd-granularity-cifar100lt-vit.md)):
  `experiments.cifar100_lt.vit_mgd --phase {baseline,search,reeval,all}` + `experiments.cifar100_lt.render_vit_mgd`.

## Engineering gotchas (don't relearn these)

**Numerics / determinism**
- **fp32 only** for metasmoothness finite differences and for exact metagradients —
  TF32/fp16/bf16 flip the sign of the tiny Definition-1 second difference and break
  REPLAY bit-determinism. bf16 is fine for ordinary training (scale sweep, reevals).
- Configure determinism *before* CUDA initializes: `configure_replay_determinism(seed, tf32=False)`
  sets `use_deterministic_algorithms(True)` and requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  to already be in the environment (set it at module import).
- Under `use_deterministic_algorithms(True)`, **`scatter_add` and `bincount` raise on
  CUDA** — do those reductions (class/cluster mass summaries, label counts, k-means)
  on CPU. Switch back with `torch.use_deterministic_algorithms(False)` for ordinary
  (bf16, augmented, fused-AdamW) training.

**Normalization & the functional engine**
- The functional engine holds **buffers fixed**, so BatchNorm running stats don't
  update during REPLAY. Use **GroupNorm** for ResNet curation (`SMOOTH_GN_ROUTINE`,
  0 buffers) — re-validated metasmooth in Phase A. ViT is LayerNorm-only, so there
  are no running-stat buffers and **no BN recalibration** is needed.

**REPLAY cost & memory**
- Store-all unrolling OOMs past ~50 inner steps on an A40 → use REPLAY
  (`recursive_replay_state`, `branching_factor=4`+), O(k·log T) memory. REPLAY does
  **not** support higher-order diff (it raises if grad is enabled); take the
  first-order metagradient with `autograd.grad(obj, z)`.
- The REPLAY **backward dominates** wall time and is largely branching-independent
  (per-step create-graph autograd); the practical speed lever is the **inner horizon**,
  not the branching factor.

**SSH / process management on the VMs**
- Clean detach: `setsid <cmd> > log 2>&1 < /dev/null &` (a bare `nohup … &` can keep
  the ssh channel open until the child's fds close).
- `pgrep -f <pattern>` **self-matches** the polling command's own cmdline → poll a
  log-file marker (e.g. `ALL_PHASES_DONE`, `Traceback`) instead.
- Piping a long-running command through `tail`/`head` buffers until exit — read the
  log file directly.

**Windows**
- The temperature knob env var is `TEMPERATURE`, **not** `TEMP` (Windows reserves `TEMP`).

## Source documents

- `HANDOFF_metasmoothness.md`, `HANDOFF_vm_phaseABC.md`, `HANDOFF_vit_metasmoothness.md` (all verbatim in [archive/original-handoffs.md](archive/original-handoffs.md))
- `.worktrees/autoresearch-mgd/artifacts/AUTORESEARCH_REPORT.md`
- `.worktrees/imagenet-lt-autoresearch/artifacts/imagenet_lt/IMAGENET_LT_MGD_RESULTS.md`
- the runner scripts named above
