# MGD Granularity on ImageNet-LT: Per-class vs Per-cluster vs Per-example

> Part of the consolidated metagradient data-curation research. See [README.md](README.md) for the overview and cross-experiment synthesis.

## Question/Goal

On ImageNet-LT, for a balanced (held-out class-balanced) cross-entropy objective, which weighting basis -- per-class label, global label-agnostic visual cluster, or per-example -- produces the best metagradient-selected training distribution? This workstream tests three grouping techniques under a fixed signed-KL outer update and reports both a frozen-backbone linear-head routine and a full-ResNet-9 fine-tuning variant.

## Setup

### Dataset and frozen-backbone (DINOv2 linear head) routine

- Dataset: `inria-chile/imagenet-lt-v2` Hugging Face parquet ([Vo et al. 2024](https://arxiv.org/abs/2405.15613)).
- Backbone: frozen self-supervised DINOv2 ViT-B/14, 768-dimensional features.
- Routine `A(z)`: buffer-free linear head with final-logit scale 10.
- Split: 115,846 source train examples; 1 balanced objective example and 1 balanced meta-validation example per class; 113,846 exposed inner-train examples.
- Inner training: one epoch, batch size 4,096, 28 smooth-AdamW steps, learning rate 0.01.
- Outer training: REPLAY, fixed signed KL step 0.001, 100 meta-steps.
- Selection: minimum balanced meta-validation CE.
- Group counts: **1,000 per-class**, **1,000 global label-agnostic hierarchical k-means clusters**, and **113,846 per-example** groups.

### Full ResNet-9 variant

- Model: the trained smooth ResNet-9 baseline, warm-started from `resnet9_smooth_baseline_10ep.pt`.
- Full-model loop: every one of the 7.08M trainable parameters is updated inside the differentiable inner loop -- not a frozen-backbone or projection-head run.
- Architecture: average pooling, BatchNorm-before-ReLU, logits divided by 10. BatchNorm running statistics held fixed during REPLAY for exact deterministic differentiation; BatchNorm affine parameters remain trainable.
- Shared exposed pool: 8,192 images selected to cover all 1,000 labels and all 1,000 global DINOv2 hierarchical clusters.
- Grouping arms: `per_class` (one persistent learned weight per label) and `per_cluster` (one persistent learned weight per global label-mixing cluster). No `per_example` arm in this variant.
- Inner loop: 64 full-model momentum-SGD steps, batch size 128, 64px images, peak LR 0.0001 with warmup/cosine, weight decay 0.0005.
- Outer loop: exact REPLAY metagradients, 20 fixed-KL signed updates of 0.001.
- Selection: minimum balanced CE on a disjoint 1-example-per-class meta-validation set.
- Final evaluation: complete official 20,000-image validation and 50,000-image test splits.

## Method

Both variants use a fixed signed-KL outer step and select the reported meta-step by minimum balanced meta-validation CE (frozen-backbone: over 100 meta-steps; full-ResNet9: over 20 meta-steps). The 1,000 global hierarchical k-means clusters are label-agnostic (constructed without label information) and mix labels by construction -- they range from 1 to 4,882 examples per cluster.

## Results

### Frozen DINOv2 linear-head routine -- Selected Results (100 meta-steps, primary)

| Technique | Selected step | Test balanced CE | CE delta vs uniform | Relative reduction | Balanced test accuracy | Fraction of oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| Uniform control | - | 1.616359 | - | - | - | - |
| Per-class MGD | 43 | 1.392773 | +0.223586 | 13.83% | 70.912% | 72.24% |
| Per-cluster MGD | 28 | 1.573897 | +0.042462 | 2.63% | 67.840% | 13.72% |
| Per-example MGD | 21 | 1.447500 | +0.168859 | 10.45% | 69.682% | 54.56% |
| Class-prior oracle | - | 1.306855 | +0.309504 | 19.15% | - | 100.00% |

Positive loss delta means lower held-out CE than the uniform-logit baseline. All three MGD grouping techniques satisfy the success criterion.

### Frozen DINOv2 -- steps20 vs steps100 trajectories

`mgd_primary_k1000_steps20.md` (20 meta-steps):

| Technique | Selected step | Objective balanced CE delta | Meta-val balanced CE delta | Val balanced CE delta | Test balanced CE delta |
|---|---:|---:|---:|---:|---:|
| per_class | 20 | +0.332984 | +0.222416 | +0.203644 | +0.206977 |
| per_cluster | 20 | +0.119909 | +0.050524 | +0.041575 | +0.042098 |
| per_example | 20 | +0.707778 | +0.183314 | +0.158902 | +0.168651 |

`mgd_primary_k1000_steps100.md` (100 meta-steps, matches the Selected Results table above):

| Technique | Selected step | Objective balanced CE delta | Meta-val balanced CE delta | Val balanced CE delta | Test balanced CE delta |
|---|---:|---:|---:|---:|---:|
| per_class | 43 | +0.460098 | +0.256436 | +0.226989 | +0.223586 |
| per_cluster | 28 | +0.141962 | +0.051775 | +0.041384 | +0.042462 |
| per_example | 21 | +0.723966 | +0.183505 | +0.158872 | +0.168859 |

Positive deltas mean the selected MGD step reduced cross-entropy relative to that technique's uniform-logit step 0. At 20 meta-steps, `per_class` and `per_cluster` are both still at their final available step (20) and have not yet reached the selected steps (43 and 28 respectively) found with the longer 100-step run; `per_example` is at step 20 in both runs, close to its eventual selected step of 21, and its test delta is nearly identical (+0.168651 vs +0.168859).

### Full ResNet-9 variant -- Official Split Results

| Treatment | Selected step | Val balanced CE | Val balanced acc | Test balanced CE | Test balanced acc | Test CE delta vs uniform | Oracle gain recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform weights | - | 6.141503 | 4.400% | 6.172461 | 4.408% | - | - |
| Balanced class-prior oracle | - | 5.968172 | 4.445% | 5.989790 | 4.464% | +0.182671 | 100.0% |
| Per-label MGD | 20 | 5.980006 | 4.515% | 6.003799 | 4.546% | **+0.168663** | **92.3%** |
| Per-cluster MGD | 20 | 5.996985 | 4.500% | 6.021418 | 4.536% | **+0.151044** | **82.7%** |

Positive CE delta means lower held-out loss than uniform full-model fine-tuning.

#### Outer-loop trajectory (full ResNet-9)

| Meta-step | Per-label meta-val CE delta | Per-cluster meta-val CE delta |
|---:|---:|---:|
| 0 | 0.000000 | 0.000000 |
| 5 | +0.065489 | +0.058014 |
| 10 | +0.111260 | +0.098705 |
| 15 | +0.140751 | +0.125586 |
| 20 | **+0.157434** | **+0.142474** |

Both grouping treatments improve monotonically through step 20. Per-label weighting is strongest; label-agnostic cluster weighting captures about 89.6% of the per-label test-CE improvement.

## Findings

1. **Per-class MGD is strongest on balanced held-out ImageNet-LT loss** in both backbones tested. Frozen DINOv2: 1.392773 test balanced CE (+13.83% relative reduction vs uniform, 70.912% balanced test accuracy, recovering 72.24% of the class-prior oracle's gain) at selected step 43. Full ResNet-9: per-label MGD recovers 92.3% of the oracle gain (test CE delta +0.168663 vs oracle's +0.182671).
2. **Per-example MGD fits the small balanced objective fastest and most aggressively, but degrades after an early peak.** It peaks at meta-step 21 (frozen-backbone routine) with test balanced CE 1.447500 (+10.45%, 54.56% of oracle gain). Its objective delta (+0.723966) is much larger than its held-out test delta (+0.168859) -- evidence of objective overfitting. (Not tested in the full-ResNet9 variant.)
3. **Per-cluster MGD (label-agnostic) is the weakest basis for this objective in the frozen-backbone routine.** Test balanced CE 1.573897, only +2.63% relative reduction and 13.72% of the oracle gain at selected step 28 -- substantially behind both per-class and per-example.
4. **In the full ResNet-9 variant, per-cluster MGD is much more competitive with per-class** (82.7% vs 92.3% of oracle gain, monotonically improving trajectory through step 20), though still second to per-label.
5. **Class-prior oracle reference:** test balanced CE 1.306855 (+0.309504 / +19.15% vs uniform 1.616359) in the frozen-backbone routine, defining 100% of the oracle gain; full-ResNet9 oracle test CE delta is +0.182671 (test CE 5.989790 vs uniform 6.172461).

### Interpretation

For this balanced-loss supervised target, the class prior is the strongest curation basis in both backbones tested. Per-example MGD is competitive in the frozen-backbone routine but exhibits clear objective overfitting (large objective delta, much smaller and non-monotonic held-out delta, peaking early at step 21 then degrading). The label-agnostic global hierarchical visual clusters (1,000 clusters, ranging 1-4,882 examples, mixing labels by construction) form a weaker basis than the class prior in the frozen-backbone routine (13.72% of oracle gain) but recover a much larger share (82.7%) when used to weight a full-model fine-tuning run -- still behind per-class (92.3%) but a meaningfully useful label-agnostic signal in that setting.

## Reproduction

Artifacts (frozen-backbone routine):
- `mgd_primary_k1000_steps100.compact.json`: compact controls, selected results, and trajectories.
- `mgd_primary_k1000_steps100.json`: full raw run.
- `mgd_primary_k1000_steps100.png`: loss-delta trajectory plot.
- `mgd_primary_k1000_steps100.summary.json`: generated selected-step summary.
- `dinov2_hkmeans_k1000.*`: hierarchical cluster basis and diagnostics.
- `dinov2_{train,val,test}.pt`: cached DINOv2 feature artifacts.

Artifacts (full ResNet-9 variant):
- `full_resnet_mgd_primary_64steps_20meta.json`: complete outer-loop trajectories.
- `full_resnet_mgd_primary_64steps_20meta.external.json`: selected-step official validation/test evaluation.
- `full_resnet_mgd_pool_8192.pt`: exact shared pool covering every label and cluster.
- `full_resnet_mgd_coverage_preflight.json`: alignment and fairness preflight.

## Gotchas

- The frozen-backbone (DINOv2 linear-head) and full-ResNet-9 results are **separate experiments with different backbones, inner loops, pools, and meta-step budgets** -- do not directly compare absolute CE values across the two tables.
- The full ResNet-9 variant does not include a `per_example` arm; only `per_class` and `per_cluster` were run.
- The 1,000 global hierarchical k-means clusters are label-agnostic and mix labels by construction (cluster sizes range from 1 to 4,882 examples).
- At 20 meta-steps, the steps20 run for `per_class` and `per_cluster` has not yet reached the selected steps found at 100 meta-steps (43 and 28 respectively); only `per_example`'s 20-step and 100-step results are close, since its selected step (21) is near both budgets.

## Source documents

- `.worktrees/imagenet-lt-autoresearch/artifacts/imagenet_lt/IMAGENET_LT_MGD_RESULTS.md`
- `.worktrees/imagenet-lt-autoresearch/artifacts/imagenet_lt/FULL_RESNET9_MGD_RESULTS.md`
- `.worktrees/imagenet-lt-autoresearch/artifacts/imagenet_lt/mgd_primary_k1000_steps20.md`
- `.worktrees/imagenet-lt-autoresearch/artifacts/imagenet_lt/mgd_primary_k1000_steps100.md`
