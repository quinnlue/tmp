# Metagradient Data Curation — Consolidated Research

This folder is the organized, refined record of a multi-phase investigation into
**metagradient descent (MGD) for dataset curation** — learning per-example /
per-group training-data weights by differentiating a held-out objective through
the entire inner training run. All of it grounds in Engstrom et al. 2025,
*Optimizing ML Training with Metagradient Descent*
([arXiv:2503.13751](https://arxiv.org/abs/2503.13751)), and all of it runs on one
shared PyTorch engine (functional Smooth AdamW + REPLAY + a group-softmax
data-weight reparameterization).

The work splits into three threads: (1) building and validating the differentiable
engine; (2) measuring and engineering **metasmoothness** — whether a training
routine is smooth enough in its data weights for metagradients to be useful — on
ResNet-9 and then ViT; and (3) the headline question, **at what granularity should
data weights live** — one per class, one per visual cluster, or one per example?
Thread (3) was run on three different problems and produced three different
winners, which is the most interesting result here (see the synthesis below).

## Document index

| # | Document | What it covers |
|---|---|---|
| 01 | [The metagradient engine](01-metagradient-engine.md) | Functional Smooth AdamW (eps inside sqrt), the group-softmax data-weight reparameterization (`z=0 ⇒ uniform`), the per-class/cluster/example unification via `group_ids`, REPLAY vs store-all vs paper count-MGD, and the fp32/determinism contract. |
| 02 | [Metasmoothness — ResNet-9 / CIFAR-10](02-metasmoothness-resnet9.md) | The paper's `S`/`Ŝ` diagnostics, the smooth-model menu (÷10 final scale is dominant), GroupNorm as the buffer-free curation model, and Phases A (scaled ranking), B (smooth beats baseline 93.1/93.7 vs 91.2), C (per-cluster MGD −8.1% held-out CE under distribution shift). |
| 03 | [Metasmoothness — ViT](03-metasmoothness-vit.md) | The ViT menu; final scale dominant; post-norm is a *negative* lever for ViT (corrected `SMOOTH_VIT` = mean + pre-norm + GELU + ÷10); per-example `Ŝ` is unreliable; the AdamW LR reversal. |
| 04 | [CIFAR100-LT ViT scale study](04-cifar100lt-vit-scale-study.md) | Tiny/Small/Base ViT compute-scale sweep on CIFAR100-LT `r-100`; selects ViT-Tiny @ 1e-3 as the MGD backbone and the validated 75-epoch recipe (23.42% balanced test). |
| 05 | [MGD granularity — CIFAR-10](05-mgd-granularity-cifar10.md) | Per-example vs per-class persistent-softmax MGD on a smooth GN ResNet-9, skewed-target objective. **Per-example wins.** |
| 06 | [MGD granularity — ImageNet-LT](06-mgd-granularity-imagenetlt.md) | Per-class / per-cluster / per-example on a frozen DINOv2 linear head and a full ResNet-9, balanced objective. **Per-class wins** (clusters far more competitive when the backbone trains). |
| 07 | [MGD granularity — CIFAR100-LT / ViT-Tiny](07-mgd-granularity-cifar100lt-vit.md) | The capstone: all three granularities on a trainable ViT-Tiny, two-stage (fp32 REPLAY search → full-recipe reeval). **Per-cluster wins (+1.32).** |
| 08 | [Environment & reproduction](08-environment-and-reproduction.md) | Laptop/VM environments, data caches, per-experiment run commands, and the consolidated engineering gotchas. |

Suggested reading order: 01 → 02 → 03 → 04 → 05 → 06 → 07. The metasmoothness work
(02–03) establishes *which* routines are curatable; the MGD comparisons (05–07)
then ask *at what granularity*.

## Cross-experiment synthesis: which granularity wins?

The same group-softmax MGD mechanism, applied at three granularities, was run on
three problems. The winners disagree — and that disagreement is the finding.

| Experiment | Backbone (trainable?) | Objective | Search budget | Winner | Per-example | Per-cluster |
|---|---|---|---|---|---|---|
| CIFAR-10 ([05](05-mgd-granularity-cifar10.md)) | smooth GN ResNet-9 (yes) | skewed target | 192 inner steps, ≤12 meta-steps | **per-example** (+14.0% vs uniform) | best | not tested |
| ImageNet-LT, frozen ([06](06-mgd-granularity-imagenetlt.md)) | DINOv2 linear head (no) | balanced | 100 meta-steps | **per-class** (72% of oracle) | 2nd, overfits (peaks step 21) | weakest (14% of oracle) |
| ImageNet-LT, full ([06](06-mgd-granularity-imagenetlt.md)) | full ResNet-9 (yes) | balanced | 20 meta-steps | **per-class** | not tested | strong (82.7% of oracle) |
| CIFAR100-LT ([07](07-mgd-granularity-cifar100lt-vit.md)) | ViT-Tiny (yes) | balanced long-tail | 15 inner-ep × 15 meta-steps | **per-cluster** (+1.32) | collapses to baseline | best |

Four factors explain the pattern:

1. **Objective shape.** A sharply *skewed-target* objective rewards fine-grained
   selection: per-example can pick the within-class examples that match the
   target, and the class basis is too coarse to express it (CIFAR-10). A
   *balanced* objective gives the class prior a natural advantage and removes much
   of per-example's edge (ImageNet-LT, CIFAR100-LT).
2. **Objective-set size and search budget gate per-example.** Per-example has the
   most expressive capacity but the highest variance. With a tiny balanced
   objective (ImageNet-LT, 1 example/class) it overfits; with a short search
   horizon (CIFAR100-LT) its 10,847-dim metaparameter never leaves uniform. It
   needs both a large objective and enough meta-steps to pay off (CIFAR-10).
3. **Backbone trainability decides whether clusters matter.** Label-agnostic
   visual clusters are weak on a *frozen* linear head (ImageNet-LT: 14% of oracle)
   but strong once the *whole model trains* (ImageNet-LT full ResNet-9: 82.7%;
   CIFAR100-LT ViT: outright winner). Clusters encode representation structure
   that only becomes exploitable when the representation is being shaped.
4. **The cluster basis is the middle path.** It is finer than the class basis (so
   it can express within-class structure) yet coherent enough to receive a clean
   metagradient (unlike per-example, which spreads each tiny weight nudge over too
   many images). It wins outright when all of (trainable backbone) + (balanced
   objective) + (moderate search budget) hold — exactly the CIFAR100-LT ViT
   regime — and is the runner-up that nearly matches per-class when the backbone
   trains on ImageNet-LT.

**Takeaway.** There is no universally best granularity; the right basis is set by
the objective (skewed vs balanced), the objective/search budget (which bounds how
much per-example capacity can be used safely), and whether the backbone is frozen
or trainable. The cluster-basis reparameterization — the project's core idea — is
the robust default: it captures most of per-example's expressive benefit without
its overfitting/under-signal failure modes, and it is the clear winner in the most
realistic regime tested (a trainable transformer on a long-tailed balanced target).

## Provenance

These documents refine and consolidate the original per-workstream handoffs and
result files that were scattered across four git branches/worktrees
(`vm-metasmooth-phaseABC`, `claude/optimistic-goldberg-0ac9e8`,
`codex/autoresearch-mgd`, `codex/imagenet-lt-autoresearch`). Each document's
`## Source documents` section lists the originals it draws from. All quantitative
results are reproduced from those sources; nothing here is re-measured.
