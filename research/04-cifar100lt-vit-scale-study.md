# CIFAR100-LT corrected-smooth ViT compute-scale study

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

## Goal

Use the corrected-smooth ViT architecture from the ViT metasmoothness study
(mean pooling, pre-norm, GELU, `final_scale=10`) to run an ordinary-training
compute-scale study on CIFAR100-LT, in order to choose the backbone size,
learning rate, and training horizon for the full-model MGD comparison.

## Setup

- Dataset: `tomas-gajarsky/cifar100-lt`, `r-100`.
- Long-tailed training pool: 10,847 images.
- Balanced official test split: 50 images/class for validation and 50
  images/class for the untouched final test.
- Architecture correction from the ViT metasmoothness study: mean pooling,
  pre-norm, GELU, and `final_scale=10`.
- Optimizer: AdamW, bf16, random crop/flip augmentation, 10% warmup, cosine
  decay.
- Profiles:
  - **Tiny**: 5,367,460 parameters, dim 192, depth 12, 3 heads
  - **Small**: 21,351,652 parameters, dim 384, depth 12, 6 heads
  - **Base**: 85,170,532 parameters, dim 768, depth 12, 12 heads

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

The independent 150-epoch finalists were run concurrently, so their
wall-clock times are not directly comparable. Use the 75-epoch matrix for
clean relative runtime comparisons.

### 150-epoch final-test tiers

| Profile | Balanced | Many | Medium | Few |
|---|---:|---:|---:|---:|
| Tiny | 25.96% | 47.49% | 22.80% | **4.53%** |
| Small | 27.58% | 50.97% | 24.00% | 4.47% |
| Base | **27.64%** | **51.09%** | **24.40%** | 4.07% |

## Findings / Decisions

- **Ordinary-training efficiency knee: ViT-Small at `5e-4`.** It reaches
  99.8% of Base accuracy with 25.1% of Base parameters.
- **Primary full-model MGD backbone: ViT-Tiny at `1e-3`.** It retains 93.9%
  of Base accuracy with 6.3% of Base parameters. Exact REPLAY cost is
  dominated by model size and trajectory length, so Tiny permits meaningfully
  deeper and better-converged MGD instead of spending the budget on model
  width.
- **Capacity confirmation: ViT-Small at `5e-4`.** Run it after the Tiny MGD
  comparison has stable settings.
- **Do not use Base for the initial MGD matrix.** At 150 epochs it adds only
  0.06 balanced-accuracy points over Small and 1.68 points over Tiny.

Tiny@1e-3, 75-epoch baseline numbers (the figures used as the MGD comparison
point): **23.42% balanced test accuracy**, best val epoch **58**,
many/medium/few = **45.77% / 19.49% / 1.93%**.

## Full-model MGD plan

Use the corrected-smooth ViT-Tiny with a 100-epoch inner horizon. The
150-epoch curve peaks at epoch 90, so 100 epochs captures convergence while
avoiding the flat tail. The inner training must use the repo's functional
SmoothAdamW and exact REPLAY path; the ordinary scale sweep used PyTorch
AdamW only to choose the backbone and horizon.

Compare all methods under the same initialization, deterministic batch order,
inner horizon, balanced validation-CE objective, and untouched final test:

1. Uniform unweighted baseline.
2. Label groups: one learned weight per CIFAR-100 class.
3. Each learned clustering technique: one learned weight per cluster.
4. Per-example MGD: one learned weight per training example.

Use enough outer MGD steps to demonstrate convergence rather than a single
metagradient update. Track objective/test balanced CE and accuracy, tier
accuracy, weight entropy/effective sample size, and generalization gap at
every outer step. Confirm the final MGD setting once on ViT-Small.

This plan was executed in
[07-mgd-granularity-cifar100lt-vit.md](07-mgd-granularity-cifar100lt-vit.md).

## Source documents

- `CIFAR100_LT_VIT_SCALE_STUDY.md` (verbatim in [archive/original-handoffs.md](archive/original-handoffs.md))
