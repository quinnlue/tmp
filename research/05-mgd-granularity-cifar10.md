# MGD Granularity on CIFAR-10: Per-example vs Per-class (Persistent-Softmax)

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

## Question/Goal

Under a known skewed target-label distribution, how does **persistent-softmax** per-example metagradient descent (MGD) compare with the same mechanism reparameterized onto one weight per class?

This is a comparison of two parameterizations of the repository's **persistent-softmax MGD** relaxation (a group-softmax reparameterization of training-set weights). It is explicitly **not** the paper's discrete count-based MGD algorithm implemented separately in `paper_mgd.py`.

The intended tradeoff under test:
- Per-class has lower metaparameter dimension and should be statistically more stable, but cannot distinguish examples within a class.
- Per-example has much more expressive selection capacity, but may overfit a small objective set.

Grounding paper: [Optimizing ML Training with Metagradient Descent](https://arxiv.org/abs/2503.13751) (Engstrom et al.), which frames training configuration as a differentiable metaparameter optimization problem via REPLAY and notes that high-dimensional per-example selection can overfit a small output/objective set.

## Setup

### Fair-comparison design

- CIFAR-10 training pool: exactly **4,000 examples, 400 per class**, balanced.
- Pool examples have stable global IDs; every pool example appears exactly once per inner epoch.
- Objective and meta-validation splits are disjoint from each other and from the pool, and share the same explicit all-class skew.
- The official CIFAR-10 test split is untouched during MGD and evaluated only after the best meta-step is selected by meta-validation CE.
- Both skew-matched target metrics and balanced-test metrics are reported.

Primary vehicle-heavy target probabilities (CIFAR-10 class order):

```text
airplane 0.400, automobile 0.200, bird 0.100, cat 0.100,
deer 0.050, dog 0.050, frog 0.025, horse 0.025,
ship 0.025, truck 0.025
```

Hard-animal control target probabilities:

```text
airplane 0.025, automobile 0.025, bird 0.100, cat 0.400,
deer 0.100, dog 0.200, frog 0.050, horse 0.050,
ship 0.025, truck 0.025
```

### Shared model and meta-optimization

- Model: smooth GroupNorm ResNet-9 (`ms.SMOOTH_GN_ROUTINE`).
- Inner optimizer: differentiable smooth AdamW.
- Inner horizon: 12 epochs x 16 batches = **192 steps**.
- Batch size: 250. Inner learning rate: 0.02.
- REPLAY branching factor: 4.
- Meta-steps: 12 for paired comparison runs.
- Outer update: signed metagradient update normalized to a fixed **KL change of 0.0025** in the induced training distribution per step (fixed-KL is necessary because the same raw logit step on 10 class coordinates vs. 4,000 example coordinates would not produce comparable movement in the induced sample distribution).
- Strict REPLAY determinism configured before CUDA initialization.

### The only experimental variable

| Arm | Group IDs | Base group masses | Metaparameter dimension |
|---|---|---|---:|
| Per-class | True class labels | 0.1 per class | 10 |
| Per-example | Stable pool IDs | 1/4000 per example | 4,000 |

### Fairness invariants

At uniform logits, both parameterizations are mathematically and empirically identical:
- Uniform per-class/per-example objective difference: zero.
- Uniform meta-validation difference: zero.
- Uniform trained-state checksum difference: zero.
- Initial class-aggregated metagradient relative L2 difference: approximately `1e-7` to `2e-7`.
- Fixed-KL updates achieved the requested KL movement in every paired run.

### Controls

- **Uniform:** balanced pool with uniform training mass.
- **Class-ratio oracle:** a static class weighting given the target-label probabilities directly. Answers how well class weighting can perform if it already knows the desired label distribution. A strong reference, not a guaranteed upper bound.

## Method

### Autoresearch Q0-Q5 queue

The GPU was used serially; only one experiment ran at a time.

| Queue | Purpose | Key change | Result |
|---|---|---|---|
| Q0 | Tiny paired smoke | 400 pool, 8 inner steps, 1 meta-step | Fairness checks passed; skew was learnable |
| Q1 | Primary comparison | Vehicle skew, 2,000 objective, 192 inner steps | Large per-example win |
| Q2 | Diagnose per-class instability | Per-class only, 10x smaller KL step, 24 meta-steps | More stable, still far behind |
| Q3 | Robustness to target difficulty | Hard animal skew | Per-example nearly matched oracle |
| Q4 | Test small-objective overfitting | 200 objective examples, seed 0 | Gap narrowed; per-example showed positive test gap |
| Q5 | Confirm noisy small-objective result | Same as Q4, seed 1 | Per-example still won CE; result remained noisy |

The queue was stopped after Q5 because the primary result, target-difficulty control, small-objective behavior, and one justified confirmation seed had all been established.

## Results

All MGD results below use the meta-step selected by minimum skew-matched meta-validation CE. The official test set was not used for selection.

| Run | Method | Test CE | Gain vs uniform | Target acc | Balanced CE | Obj-to-test gap | Selected step |
|---|---|---:|---:|---:|---:|---:|---:|
| Large objective / vehicle skew | uniform | 1.691 | - | 0.355 | 1.751 | -0.047 | - |
| Large objective / vehicle skew | per-class | 1.681 | +0.010 (+0.6%) | 0.353 | 1.766 | -0.052 | 1 |
| Large objective / vehicle skew | **per-example** | **1.455** | **+0.236 (+14.0%)** | **0.512** | 1.837 | -0.043 | 12 |
| Large objective / vehicle skew | class-ratio oracle | 1.364 | +0.327 (+19.3%) | 0.537 | 2.241 | -0.025 | - |
| Large objective / animal skew | uniform | 1.790 | - | 0.328 | 1.751 | -0.006 | - |
| Large objective / animal skew | per-class | 1.783 | +0.007 (+0.4%) | 0.321 | **1.698** | +0.027 | 12 |
| Large objective / animal skew | **per-example** | **1.580** | **+0.211 (+11.8%)** | **0.429** | 1.810 | +0.006 | 10 |
| Large objective / animal skew | class-ratio oracle | 1.575 | +0.215 (+12.0%) | 0.443 | 2.362 | +0.003 | - |
| Small objective / seed 0 | uniform | 1.691 | - | 0.355 | 1.751 | +0.041 | - |
| Small objective / seed 0 | per-class | 1.503 | +0.189 (+11.1%) | 0.432 | **1.710** | -0.026 | 12 |
| Small objective / seed 0 | **per-example** | **1.496** | **+0.195 (+11.6%)** | **0.496** | 1.811 | +0.020 | 9 |
| Small objective / seed 0 | class-ratio oracle | 1.364 | +0.327 (+19.3%) | 0.537 | 2.241 | +0.010 | - |
| Small objective / seed 1 | uniform | 1.720 | - | 0.424 | 1.791 | -0.023 | - |
| Small objective / seed 1 | per-class | 1.646 | +0.075 (+4.3%) | **0.447** | 1.769 | -0.027 | 6 |
| Small objective / seed 1 | **per-example** | **1.581** | **+0.139 (+8.1%)** | 0.444 | **1.768** | +0.011 | 12 |
| Small objective / seed 1 | class-ratio oracle | 1.382 | +0.338 (+19.6%) | 0.530 | 2.285 | +0.075 | - |

Q2 (smaller per-class KL step) improved per-class's primary-skew test CE from `1.681` to `1.659`, but it remained far behind per-example at `1.455`.

## Findings

1. **Per-example MGD dominates with a large objective.** On the primary vehicle skew, it reduced target-test CE by 14.0% relative to uniform (1.691 -> 1.455), versus 0.6% for per-class (1.691 -> 1.681).
2. **The result is not specific to easy vehicle classes.** On the hard-animal skew, per-example reaches 1.580, nearly matching the class-ratio oracle at 1.575 (+11.8% vs +12.0%); per-class reaches 1.783 (+0.4%).
3. **Per-example gains come primarily from within-class selection.** In Q1, selected per-example class masses ranged only from about `0.094` to `0.115`, far from the target's `0.025`-`0.400` range. Its overall effective sample size (ESS) fell from 4,000 to about 3,731, with within-class ESS around 371-378 examples/class. Aggregate learned class masses stayed near uniform while within-class ESS fell -- per-example selects useful examples rather than recovering label prevalence.
4. **Per-class MGD is update-sensitive and noisy.** A 10x smaller KL step improves stability and primary-skew test CE (1.681 -> 1.659) but the low-dimensional trajectory still oscillates and does not recover the class-ratio oracle.
5. **Small-objective overfitting appears for per-example, as expected.** At seed 0, test CE is 1.503 (per-class) vs 1.496 (per-example); at seed 1, 1.646 vs 1.581. Per-example develops a positive objective-to-test gap in both small-objective runs (`+0.020`, `+0.011`), while per-class gaps remain negative (`-0.026`, `-0.027`). Per-example still wins target-test CE in both seeds, but the margin narrows substantially versus the large-objective condition.
6. **Target-versus-balanced tradeoff is real.** Per-example usually optimizes the skewed target more aggressively; per-class often retains stronger balanced-test performance (e.g., animal skew: per-class balanced CE 1.698 vs per-example 1.810; small-objective seed 0: 1.710 vs 1.811).

### Interpretation

The class basis is too restrictive for this supervised CIFAR-10 setting: examples sharing a label vary materially in usefulness for the skewed target. Per-example MGD exploits that signal and can match a class-ratio oracle without moving aggregate class masses much. The statistical advantage of the low-dimensional class basis appears only as reduced overfitting and better balanced-task retention when the objective set is small -- it does not translate into superior target-test performance in this setup.

## Reproduction

Primary paired comparison (Q1), run from `/workspace/autoresearch-mgd` on the VM:

```bash
/venv/main/bin/python -m experiments.compare_mgd_granularity \
  --data-dir /workspace/tmp/data \
  --output artifacts/q1_primary.json \
  --pool-per-class 400 \
  --objective-size 2000 \
  --validation-size 2000 \
  --inner-epochs 12 \
  --batch-size 250 \
  --inner-lr 0.02 \
  --meta-steps 12 \
  --step-kl 0.0025 \
  --meta-optimizer sign_kl \
  --method replay \
  --device cuda 2>&1 | tee logs/q1_primary.log
```

Hard-animal skew (Q3) adds `--target-probs 0.025,0.025,0.1,0.4,0.1,0.2,0.05,0.05,0.025,0.025`.

Small-objective comparison (Q4/Q5) uses `--objective-size 200 --seed 0` (Q4) or `--seed 1` (Q5), with `--validation-size 2000`, `--meta-steps 12`, `--step-kl 0.0025`.

Verification:

```text
Local: 73 passed, 6 deselected
VM:    79 passed in 397.26s
```

```powershell
# Local isolated worktree
C:\Users\luequ\micromamba\envs\torch311\python.exe -m pytest -m "not gpu" -q
C:\Users\luequ\micromamba\envs\torch311\python.exe -m experiments.summarize_mgd_granularity --artifacts-dir artifacts
```

## Gotchas

- This compares two **persistent-softmax MGD** parameterizations, not the paper's paper-faithful count-based MGD in `paper_mgd.py`. Do not conflate the two when citing "MGD" results from this experiment.
- Fixed-KL signed updates are fair in induced-distribution movement, but other outer optimizers (e.g., Adam) may alter the ranking -- per-class performance in particular was update-sensitive.
- Large-objective conditions used one seed; the small-objective condition was confirmed with two seeds (0 and 1) and remained noisy.
- The model is a smooth GroupNorm ResNet-9 with a 192-step inner horizon, not full convergence and not the original paper's large-scale application.
- The class-ratio oracle knows only target label proportions; it is not a true per-example upper bound.

## Source documents

- `.worktrees/autoresearch-mgd/artifacts/AUTORESEARCH_REPORT.md`
- `.worktrees/autoresearch-mgd/artifacts/AUTORESEARCH_RESULTS.md`
