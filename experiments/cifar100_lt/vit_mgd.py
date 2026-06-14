"""Per-example vs per-cluster vs per-class metagradient data curation on
CIFAR100-LT with the corrected-smooth ViT-Tiny 75-epoch recipe.

Two-stage design (see plan):

  Phase "baseline"  -- train the uniform recipe once (PyTorch AdamW, bf16,
                       augmented, 75 epochs).  It is the uniform comparison point
                       AND the source of the penultimate ViT-Tiny embeddings used
                       to build the label-agnostic per-cluster basis.
  Phase "search"    -- for each granularity {per_class, per_cluster, per_example}
                       run exact-REPLAY metagradient descent on the per-group data
                       weights z over a cheaper inner horizon (fp32, deterministic,
                       no augmentation), selecting the meta-step with the lowest
                       balanced meta-validation CE.  Saves the learned per-example
                       multipliers.
  Phase "reeval"    -- re-train the FULL 75-epoch recipe under each method's learned
                       multipliers and report balanced val/test + many/medium/few.

z is the deviation from the base distribution: example i is weighted by
softmax(z + T log base)[g_i] / base[g_i], so z = 0 is ordinary training for ANY
base (per-class long-tail counts, per-cluster sizes, or per-example uniform).  This
matches the ImageNet-LT comparison driver's `effective_group_logits` parameterization.

Reuses the scale-sweep recipe (`_run_cifar100_lt_vit_scale_sweep`) for the ordinary
training loop and the REPLAY engine (`recursive_replay` + `functional_train`) for the
metagradients.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# REPLAY needs strict cuBLAS determinism; this must be set before CUDA initializes
# (the search phase calls configure_replay_determinism, which refuses to enable
# determinism if CUDA was initialized without it).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor
from torch.func import functional_call
from torch.nn import functional as F

from experiments.cifar100_lt.cluster_basis import hierarchical_kmeans_cluster_basis
from experiments.cifar100_lt.data import CIFAR100LTData, load_cifar100_lt
from experiments.cifar100_lt.vit_scale_sweep import (
    PROFILES,
    augment,
    build_model,
    configure_training,
    evaluate,
    learning_rate_at_step,
    normalize_images,
    resolve_device,
    save_json,
)
from determinism import configure_replay_determinism
from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from model import VisionTransformerClassifier, per_example_cross_entropy_loss
from recursive_replay import recursive_replay_state
from weighting import group_masses

Granularity = Literal["per_class", "per_cluster", "per_example"]
GRANULARITIES: tuple[Granularity, ...] = ("per_class", "per_cluster", "per_example")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    cache_dir: str = "./data/cifar100-lt"
    artifact_dir: str = "artifacts/cifar100_lt_vit_mgd"
    dataset_config: str = "r-100"
    device: str = "auto"
    seed: int = 0
    # Full 75-epoch recipe (Phase baseline / reeval)
    epochs: int = 75
    batch_size: int = 256
    eval_batch_size: int = 512
    peak_lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_fraction: float = 0.1
    amp: str = "bf16"
    validation_per_class: int = 50
    # MGD search (Phase search, REPLAY, fp32)
    granularities: tuple[Granularity, ...] = GRANULARITIES
    search_epochs: int = 25
    meta_steps: int = 20
    inner_lr: float = 1e-3
    temperature: float = 1.0
    outer_step_kl: float = 1e-3
    branching_factor: int = 4
    num_clusters: int = 128
    objective_per_class: int = 25  # of the 50/class balanced val: objective vs meta-val


# --------------------------------------------------------------------------- #
# Granularity spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MgdSpec:
    name: Granularity
    group_ids: Tensor  # [n_pool] group index per pool example
    base_group_masses: Tensor  # [num_groups] sums to 1
    group_class_ids: Tensor  # [num_groups] representative class (summary only)

    @property
    def num_groups(self) -> int:
        return int(self.base_group_masses.numel())


def class_base_masses(labels: Tensor, *, num_classes: int) -> Tensor:
    # bincount on CPU: it is nondeterministic on CUDA and the search loop may have
    # deterministic-algorithms mode enabled from a previous granularity.
    counts = torch.bincount(labels.cpu(), minlength=num_classes).to(torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("every class must appear in the training pool")
    return (counts / counts.sum()).to(labels.device)


def build_spec(
    granularity: Granularity,
    pool_labels: Tensor,
    *,
    num_classes: int,
    cluster_group_ids: Tensor | None = None,
    cluster_base_masses: Tensor | None = None,
) -> MgdSpec:
    device = pool_labels.device
    if granularity == "per_class":
        group_ids = pool_labels.clone()
        base = class_base_masses(pool_labels, num_classes=num_classes)
        group_class_ids = torch.arange(num_classes, dtype=torch.long, device=device)
    elif granularity == "per_example":
        n = pool_labels.numel()
        group_ids = torch.arange(n, dtype=torch.long, device=device)
        base = torch.full((n,), 1.0 / float(n), dtype=torch.float32, device=device)
        group_class_ids = pool_labels.clone()
    elif granularity == "per_cluster":
        if cluster_group_ids is None or cluster_base_masses is None:
            raise ValueError("per_cluster requires cluster assignments")
        group_ids = cluster_group_ids.to(device=device, dtype=torch.long)
        base = cluster_base_masses.to(device=device, dtype=torch.float32)
        num_groups = int(base.numel())
        # Representative class per cluster = majority pool label (summary only).
        # Computed on CPU to stay deterministic-mode safe (bincount on CUDA raises).
        labels_cpu = pool_labels.cpu()
        group_ids_cpu = group_ids.cpu()
        group_class_ids = torch.zeros(num_groups, dtype=torch.long)
        for cluster in range(num_groups):
            members = labels_cpu[group_ids_cpu == cluster]
            if members.numel():
                group_class_ids[cluster] = torch.bincount(
                    members, minlength=num_classes
                ).argmax()
        group_class_ids = group_class_ids.to(device)
    else:
        raise ValueError(f"unknown granularity: {granularity}")
    base = base / base.sum()
    return MgdSpec(
        name=granularity,
        group_ids=group_ids,
        base_group_masses=base,
        group_class_ids=group_class_ids,
    )


def effective_group_logits(z: Tensor, base_group_masses: Tensor, temperature: float) -> Tensor:
    """z is the deviation from base; z=0 -> target distribution == base."""
    return z + temperature * torch.log(base_group_masses.to(z.device))


def effective_group_masses(z: Tensor, base_group_masses: Tensor, temperature: float) -> Tensor:
    return group_masses(effective_group_logits(z, base_group_masses, temperature), temperature)


def mass_summary(z: Tensor, spec: MgdSpec, *, num_classes: int, temperature: float) -> dict[str, Any]:
    masses = effective_group_masses(z.detach(), spec.base_group_masses, temperature)
    # Aggregate to per-class on CPU: scatter_add on CUDA is nondeterministic and
    # raises under the search phase's deterministic-algorithms mode.
    class_masses = torch.zeros(num_classes, dtype=masses.dtype)
    class_masses.scatter_add_(0, spec.group_class_ids.cpu(), masses.cpu())
    class_masses = class_masses.to(masses.device)
    entropy = float(-(masses * masses.log()).sum().item())
    ess = float((1.0 / masses.square().sum()).item())
    return {
        "entropy": entropy,
        "effective_sample_size": ess,
        "max_class_mass": float(class_masses.max().item()),
        "min_class_mass": float(class_masses.min().item()),
    }


# --------------------------------------------------------------------------- #
# Data / embeddings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Materialized:
    train_images: Tensor
    train_labels: Tensor
    val_images: Tensor
    val_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    train_counts: Tensor


def materialize(data: CIFAR100LTData, device: torch.device) -> Materialized:
    return Materialized(
        train_images=normalize_images(data.train_images, device),
        train_labels=data.train_labels.to(device),
        val_images=normalize_images(data.val_images, device),
        val_labels=data.val_labels.to(device),
        test_images=normalize_images(data.test_images, device),
        test_labels=data.test_labels.to(device),
        train_counts=data.train_class_counts.to(device),
    )


def per_class_first_k(labels: Tensor, k: int, *, num_classes: int) -> tuple[Tensor, Tensor]:
    """Split a balanced index set into the first k per class and the remainder."""
    first: list[Tensor] = []
    rest: list[Tensor] = []
    for class_id in range(num_classes):
        idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        first.append(idx[:k])
        rest.append(idx[k:])
    return torch.cat(first), torch.cat(rest)


@torch.inference_mode()
def pooled_features(model: VisionTransformerClassifier, images: Tensor, *, batch_size: int) -> Tensor:
    """Penultimate mean-pooled token features (pre-head), for clustering."""
    from model import patchify, _sincos_2d

    model.eval()
    config = model.config
    chunks: list[Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        inputs = images[start : start + batch_size]
        patches = patchify(inputs, config.patch_size)
        positions = _sincos_2d(
            config.grid_size, config.encoder_dim, device=inputs.device, dtype=inputs.dtype
        )
        encoded = model.patch_embedding(patches) + positions.unsqueeze(0)
        for block in model.encoder_blocks:
            encoded = block(encoded)
        encoded = model.encoder_norm(encoded)
        pooled = encoded.mean(dim=1) if config.pool == "mean" else encoded[:, 0]
        chunks.append(pooled.float())
    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------- #
# Full-recipe training (Phase baseline + reeval)
# --------------------------------------------------------------------------- #
def train_full_recipe(
    config: Config,
    data: Materialized,
    device: torch.device,
    *,
    multipliers: Tensor | None,
    tag: str,
    output_path: Path,
) -> dict[str, Any]:
    """The user's ViT-Tiny 75-epoch recipe, optionally with per-example loss weights.

    multipliers=None reproduces the ordinary unweighted baseline.
    """
    configure_training(config.seed)
    torch.use_deterministic_algorithms(False)
    model = build_model(PROFILES["tiny"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    n_train = data.train_labels.numel()
    steps_per_epoch = math.ceil(n_train / config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    if multipliers is not None and multipliers.numel() != n_train:
        raise ValueError("multipliers must have one entry per training example")
    weights = None if multipliers is None else multipliers.to(device=device, dtype=torch.float32)

    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    best_val = -math.inf
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=generator, device=device)
        for batch_start in range(0, n_train, config.batch_size):
            indices = permutation[batch_start : batch_start + config.batch_size]
            inputs = augment(data.train_images.index_select(0, indices))
            targets = data.train_labels.index_select(0, indices)
            lr = learning_rate_at_step(
                global_step,
                total_steps=total_steps,
                warmup_fraction=config.warmup_fraction,
                peak_learning_rate=config.peak_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.amp):
                logits = model(inputs)
                if weights is None:
                    loss = F.cross_entropy(logits, targets)
                else:
                    per_example = F.cross_entropy(logits, targets, reduction="none")
                    loss = (per_example * weights.index_select(0, indices)).mean()
            loss.backward()
            optimizer.step()
            global_step += 1
        val_metrics = evaluate(
            model, data.val_images, data.val_labels,
            batch_size=config.eval_batch_size, amp=config.amp,
            train_class_counts=data.train_counts,
        )
        history.append({"epoch": epoch, "val": val_metrics})
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"[{tag}] epoch {epoch:3d}/{config.epochs} "
            f"val_bal={val_metrics['balanced_accuracy']:.4f} few={val_metrics['few_accuracy']:.4f}",
            flush=True,
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, data.test_images, data.test_labels,
        batch_size=config.eval_batch_size, amp=config.amp,
        train_class_counts=data.train_counts,
    )
    result = {
        "tag": tag,
        "best_epoch": best_epoch,
        "val": history[best_epoch - 1]["val"],
        "test": test_metrics,
        "history": history,
        "minutes": (time.perf_counter() - started) / 60.0,
        "weighted": multipliers is not None,
    }
    save_json(output_path, result)
    print(
        f"[{tag}] DONE best_epoch={best_epoch} "
        f"val_bal={result['val']['balanced_accuracy']:.4f} "
        f"test_bal={test_metrics['balanced_accuracy']:.4f} "
        f"(many={test_metrics['many_accuracy']:.4f} "
        f"med={test_metrics['medium_accuracy']:.4f} few={test_metrics['few_accuracy']:.4f})",
        flush=True,
    )
    return {"result": result, "model": model}


def _autocast(device: torch.device, amp: str):
    from contextlib import nullcontext

    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


# --------------------------------------------------------------------------- #
# Phase 0: baseline + embeddings
# --------------------------------------------------------------------------- #
def phase_baseline(config: Config, data: Materialized, device: torch.device) -> None:
    artifact = Path(config.artifact_dir)
    outcome = train_full_recipe(
        config, data, device,
        multipliers=None, tag="baseline",
        output_path=artifact / "baseline.json",
    )
    model = outcome["model"]
    embeddings = pooled_features(model, data.train_images, batch_size=config.eval_batch_size)
    torch.save(
        {"model": model.state_dict(), "best_epoch": outcome["result"]["best_epoch"]},
        artifact / "baseline_checkpoint.pt",
    )
    torch.save(embeddings.cpu(), artifact / "pool_embeddings.pt")
    print(f"[baseline] saved embeddings {tuple(embeddings.shape)}", flush=True)


# --------------------------------------------------------------------------- #
# Phase 1: MGD search via REPLAY
# --------------------------------------------------------------------------- #
def build_inner_schedule(n_pool: int, *, epochs: int, batch_size: int, seed: int, device: torch.device) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed + 1)
    schedule: list[Tensor] = []
    for _ in range(epochs):
        permutation = torch.randperm(n_pool, generator=generator)
        for start in range(0, n_pool, batch_size):
            schedule.append(permutation[start : start + batch_size].to(device))
    return schedule


@torch.no_grad()
def eval_pooled(model: VisionTransformerClassifier, state, images: Tensor, labels: Tensor, *, batch_size: int) -> tuple[float, float]:
    """Mean CE and accuracy over a balanced set (== balanced CE / balanced acc)."""
    total_loss = 0.0
    correct = 0
    n = labels.numel()
    for start in range(0, n, batch_size):
        chunk_images = images[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size]
        logits = functional_call(model, (state.parameters, state.buffers), (chunk_images,))
        total_loss += float(per_example_cross_entropy_loss(logits, chunk_labels).sum().item())
        correct += int(logits.argmax(1).eq(chunk_labels).sum().item())
    return total_loss / n, correct / n


def objective_ce(model: VisionTransformerClassifier, state, images: Tensor, labels: Tensor, *, batch_size: int) -> Tensor:
    """Differentiable mean CE over the (balanced) objective set."""
    total = torch.zeros((), dtype=torch.float32, device=images.device)
    n = labels.numel()
    for start in range(0, n, batch_size):
        chunk_images = images[start : start + batch_size]
        chunk_labels = labels[start : start + batch_size]
        logits = functional_call(model, (state.parameters, state.buffers), (chunk_images,))
        total = total + per_example_cross_entropy_loss(logits, chunk_labels).sum()
    return total / float(n)


def fixed_kl_signed_update(
    z: Tensor, gradient: Tensor, base_group_masses: Tensor, *, temperature: float, target_kl: float
) -> dict[str, float]:
    """Signed metagradient step scaled to a fixed KL on the group distribution.

    Ported from the ImageNet-LT comparison driver: descend on sign(-grad), bisect
    the step scale so the distribution moves by ~target_kl, then re-center z.
    """
    old = effective_group_masses(z.detach(), base_group_masses, temperature)
    direction = -gradient.sign()
    if not torch.any(direction):
        return {"distribution_kl": 0.0, "logit_scale": 0.0}

    def kl_at(scale: float) -> float:
        proposal = effective_group_masses(z.detach() + direction * scale, base_group_masses, temperature)
        return float((proposal * (proposal.log() - old.log())).sum().item())

    lower, upper = 0.0, 1.0
    while kl_at(upper) < target_kl and upper < 256.0:
        upper *= 2.0
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        if kl_at(middle) < target_kl:
            lower = middle
        else:
            upper = middle
    scale = 0.5 * (lower + upper)
    with torch.no_grad():
        z.add_(direction, alpha=scale)
        z.sub_(z.mean())
        new = effective_group_masses(z, base_group_masses, temperature)
        actual_kl = float((new * (new.log() - old.log())).sum().item())
    return {"distribution_kl": actual_kl, "logit_scale": float(scale)}


def run_search(
    config: Config,
    spec: MgdSpec,
    data: Materialized,
    *,
    objective_images: Tensor,
    objective_labels: Tensor,
    meta_images: Tensor,
    meta_labels: Tensor,
    device: torch.device,
    num_classes: int,
    model: VisionTransformerClassifier | None = None,
) -> dict[str, Any]:
    configure_replay_determinism(config.seed, tf32=False)
    if model is None:
        model = build_model(PROFILES["tiny"])
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    initial_state = initialize_train_state(model)
    schedule = build_inner_schedule(
        data.train_labels.numel(),
        epochs=config.search_epochs,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
    )
    total_steps = len(schedule)
    base = spec.base_group_masses
    betas = (0.9, 0.999)

    def replay_step(state, step_index: int, replay_z: Tensor, create_graph: bool):
        positions = schedule[step_index]
        images = data.train_images.index_select(0, positions)
        labels = data.train_labels.index_select(0, positions)
        group_ids = spec.group_ids.index_select(0, positions)
        effective = effective_group_logits(replay_z, base, config.temperature)
        lr = learning_rate_at_step(
            step_index,
            total_steps=total_steps,
            warmup_fraction=config.warmup_fraction,
            peak_learning_rate=config.inner_lr,
        )
        opt_cfg = SmoothAdamWConfig(
            learning_rate=lr, betas=betas, eps=1e-8, weight_decay=config.weight_decay
        )
        next_state, _ = weighted_inner_step(
            model, state, images, labels, group_ids, effective, base, opt_cfg,
            temperature=config.temperature, create_graph=create_graph,
        )
        return next_state

    z = torch.zeros(spec.num_groups, dtype=torch.float32, device=device, requires_grad=True)
    history: list[dict[str, Any]] = []
    z_snapshots: list[Tensor] = []
    for step in range(config.meta_steps + 1):
        step_started = time.perf_counter()
        # Snapshot the z that produces this step's trained model (z is mutated by
        # the signed-KL update at the END of the iteration).
        z_snapshots.append(z.detach().clone())
        final_state = recursive_replay_state(
            initial_state, z, total_steps, replay_step, branching_factor=config.branching_factor
        )
        objective = objective_ce(
            model, final_state, objective_images, objective_labels, batch_size=config.eval_batch_size
        )
        meta_ce, meta_acc = eval_pooled(
            model, final_state, meta_images, meta_labels, batch_size=config.eval_batch_size
        )
        obj_acc = eval_pooled(
            model, final_state, objective_images, objective_labels, batch_size=config.eval_batch_size
        )[1]
        record = {
            "step": step,
            "objective_ce": float(objective.detach().item()),
            "objective_balanced_acc": obj_acc,
            "meta_validation_ce": meta_ce,
            "meta_validation_balanced_acc": meta_acc,
            "seconds": time.perf_counter() - step_started,
            **mass_summary(z, spec, num_classes=num_classes, temperature=config.temperature),
        }
        if step < config.meta_steps:
            (gradient,) = torch.autograd.grad(objective, z)
            record["gradient_norm"] = float(gradient.norm().item())
            record["update"] = fixed_kl_signed_update(
                z, gradient, base, temperature=config.temperature, target_kl=config.outer_step_kl
            )
        history.append(record)
        print(
            f"[search/{spec.name}] step {step:2d}/{config.meta_steps} "
            f"obj_CE={record['objective_ce']:.4f} meta_CE={meta_ce:.4f} "
            f"meta_acc={meta_acc:.4f} ess={record['effective_sample_size']:.1f} "
            f"({record['seconds']:.1f}s)",
            flush=True,
        )

    best_index = min(range(len(history)), key=lambda i: history[i]["meta_validation_ce"])
    # Per-example multipliers at the selected step (for Phase reeval).
    z_best = z_snapshots[best_index]
    effective = effective_group_logits(z_best, base, config.temperature)
    target = group_masses(effective, config.temperature)
    multipliers = (target[spec.group_ids] / base[spec.group_ids]).detach()
    return {
        "granularity": spec.name,
        "num_groups": spec.num_groups,
        "selection_rule": "minimum balanced meta-validation CE",
        "selected_step": best_index,
        "history": history,
        "selected_multipliers": multipliers.cpu().tolist(),
        "group_ids": spec.group_ids.cpu().tolist(),
        "cluster_base_masses": base.cpu().tolist() if spec.name == "per_cluster" else None,
    }


# --------------------------------------------------------------------------- #
# Phase 2: re-eval under learned weights
# --------------------------------------------------------------------------- #
def phase_reeval(config: Config, data: Materialized, device: torch.device, granularity: Granularity) -> None:
    artifact = Path(config.artifact_dir)
    search_path = artifact / f"search_{granularity}.json"
    payload = json.loads(search_path.read_text(encoding="utf-8"))
    multipliers = torch.tensor(payload["selected_multipliers"], dtype=torch.float32, device=device)
    train_full_recipe(
        config, data, device,
        multipliers=multipliers, tag=f"reeval_{granularity}",
        output_path=artifact / f"reeval_{granularity}.json",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_or_build_clusters(config: Config, data: Materialized, device: torch.device) -> tuple[Tensor, Tensor]:
    artifact = Path(config.artifact_dir)
    embeddings_path = artifact / "pool_embeddings.pt"
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"{embeddings_path} not found -- run the 'baseline' phase first to produce embeddings"
        )
    # Cluster on CPU: k-means uses bincount/cdist that are nondeterministic on CUDA
    # and would raise under the search phase's deterministic-algorithms mode.
    embeddings = torch.load(embeddings_path, map_location="cpu")
    basis = hierarchical_kmeans_cluster_basis(
        embeddings, num_clusters=config.num_clusters, levels=2, seed=config.seed
    )
    return basis.group_ids.to(device), basis.base_group_masses.to(device)


def phase_search(config: Config, data: Materialized, device: torch.device, granularity: Granularity) -> None:
    artifact = Path(config.artifact_dir)
    num_classes = int(data.train_counts.numel())
    objective_indices, meta_indices = per_class_first_k(
        data.val_labels, config.objective_per_class, num_classes=num_classes
    )
    objective_images = data.val_images.index_select(0, objective_indices)
    objective_labels = data.val_labels.index_select(0, objective_indices)
    meta_images = data.val_images.index_select(0, meta_indices)
    meta_labels = data.val_labels.index_select(0, meta_indices)

    cluster_group_ids = cluster_base = None
    if granularity == "per_cluster":
        cluster_group_ids, cluster_base = load_or_build_clusters(config, data, device)
    spec = build_spec(
        granularity, data.train_labels, num_classes=num_classes,
        cluster_group_ids=cluster_group_ids, cluster_base_masses=cluster_base,
    )
    result = run_search(
        config, spec, data,
        objective_images=objective_images, objective_labels=objective_labels,
        meta_images=meta_images, meta_labels=meta_labels,
        device=device, num_classes=num_classes,
    )
    save_json(artifact / f"search_{granularity}.json", result)


def parse_args() -> tuple[Config, str, tuple[Granularity, ...]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "search", "reeval", "all"), default="all")
    parser.add_argument("--granularities", default=",".join(GRANULARITIES))
    parser.add_argument("--cache-dir", default=Config.cache_dir)
    parser.add_argument("--artifact-dir", default=Config.artifact_dir)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--peak-lr", type=float, default=Config.peak_lr)
    parser.add_argument("--search-epochs", type=int, default=Config.search_epochs)
    parser.add_argument("--meta-steps", type=int, default=Config.meta_steps)
    parser.add_argument("--inner-lr", type=float, default=Config.inner_lr)
    parser.add_argument("--outer-step-kl", type=float, default=Config.outer_step_kl)
    parser.add_argument("--branching-factor", type=int, default=Config.branching_factor)
    parser.add_argument("--num-clusters", type=int, default=Config.num_clusters)
    parser.add_argument("--objective-per-class", type=int, default=Config.objective_per_class)
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default=Config.amp)
    args = parser.parse_args()
    granularities = tuple(g.strip() for g in args.granularities.split(",") if g.strip())  # type: ignore[assignment]
    config = Config(
        cache_dir=args.cache_dir, artifact_dir=args.artifact_dir, device=args.device,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, peak_lr=args.peak_lr,
        search_epochs=args.search_epochs, meta_steps=args.meta_steps, inner_lr=args.inner_lr,
        outer_step_kl=args.outer_step_kl, branching_factor=args.branching_factor,
        num_clusters=args.num_clusters, objective_per_class=args.objective_per_class, amp=args.amp,
    )
    return config, args.phase, granularities  # type: ignore[return-value]


def main() -> None:
    config, phase, granularities = parse_args()
    device = resolve_device(config.device)
    Path(config.artifact_dir).mkdir(parents=True, exist_ok=True)
    data = materialize(
        load_cifar100_lt(
            config.cache_dir, config=config.dataset_config,
            validation_per_class=config.validation_per_class, seed=config.seed,
        ),
        device,
    )
    if phase in ("baseline", "all"):
        phase_baseline(config, data, device)
    if phase in ("search", "all"):
        for granularity in granularities:
            phase_search(config, data, device, granularity)
    if phase in ("reeval", "all"):
        for granularity in granularities:
            phase_reeval(config, data, device, granularity)


if __name__ == "__main__":
    main()
