"""Fair per-example versus per-class persistent-weight MGD on CIFAR-10.

The training pool is exactly class-balanced. Both MGD parameterizations use the
same pool, skewed held-out objective, model initialization, inner trajectory,
optimizer, outer optimizer, and evaluation sets. The only experimental variable
is whether the persistent softmax weight logits are tied by class or are free
per training example.

This is intentionally not the paper's discrete count-MGD algorithm. It compares
two granularities of the repository's persistent-softmax MGD relaxation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.func import functional_call

import metasmooth as ms
from determinism import configure_replay_determinism
from functional_train import (
    SmoothAdamWConfig,
    TrainState,
    initialize_train_state,
    weighted_inner_step,
)
from metagrad import InnerBatch, ObjectiveBatch, train_unrolled
from recursive_replay import recursive_replay_state
from weighting import group_masses


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
DEFAULT_TARGET = (0.40, 0.20, 0.10, 0.10, 0.05, 0.05, 0.025, 0.025, 0.025, 0.025)
Granularity = Literal["per_class", "per_example"]


@dataclass(frozen=True)
class Config:
    data_dir: str
    output: str
    seed: int
    pool_per_class: int
    objective_size: int
    validation_size: int
    target_probs: tuple[float, ...]
    inner_epochs: int
    batch_size: int
    inner_lr: float
    meta_steps: int
    meta_lr: float
    temperature: float
    meta_optimizer: Literal["adam", "sign_kl"]
    step_kl: float
    method: Literal["replay", "store_all"]
    branching_factor: int
    device: str
    methods: tuple[Granularity, ...]


@dataclass
class Split:
    pool_images: Tensor
    pool_labels: Tensor
    objective_images: Tensor
    objective_labels: Tensor
    validation_images: Tensor
    validation_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    pool_source_indices: Tensor
    objective_source_indices: Tensor
    validation_source_indices: Tensor


def parse_probabilities(value: str) -> tuple[float, ...]:
    probabilities = tuple(float(item) for item in value.split(","))
    if len(probabilities) != len(CLASSES):
        raise argparse.ArgumentTypeError(f"expected {len(CLASSES)} probabilities")
    if any(probability <= 0 for probability in probabilities):
        raise argparse.ArgumentTypeError("target probabilities must be positive")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise argparse.ArgumentTypeError(f"target probabilities sum to {total}, not 1")
    return probabilities


def exact_counts(total: int, probabilities: tuple[float, ...]) -> tuple[int, ...]:
    """Allocate ``total`` items using deterministic largest-remainder rounding."""
    expected = np.asarray(probabilities, dtype=np.float64) * total
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(counts.sum())
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:remainder]] += 1
    return tuple(int(count) for count in counts)


def select_split_indices(
    labels: np.ndarray,
    *,
    pool_per_class: int,
    objective_counts: tuple[int, ...],
    validation_counts: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose balanced-pool and disjoint exactly-skewed objective/validation splits."""
    generator = np.random.default_rng(seed)
    pool_parts: list[np.ndarray] = []
    objective_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_id, (objective_count, validation_count) in enumerate(
        zip(objective_counts, validation_counts, strict=True)
    ):
        indices = np.flatnonzero(labels == class_id)
        generator.shuffle(indices)
        needed = pool_per_class + objective_count + validation_count
        if indices.size < needed:
            raise ValueError(
                f"class {class_id} has {indices.size} examples but needs {needed}"
            )
        pool_parts.append(indices[:pool_per_class])
        objective_stop = pool_per_class + objective_count
        objective_parts.append(indices[pool_per_class:objective_stop])
        validation_parts.append(indices[objective_stop:needed])
    pool = np.concatenate(pool_parts)
    objective = np.concatenate(objective_parts)
    validation = np.concatenate(validation_parts)
    generator.shuffle(pool)
    generator.shuffle(objective)
    generator.shuffle(validation)
    return pool, objective, validation


def normalize_images(images: np.ndarray) -> Tensor:
    result = torch.from_numpy(images).float().div_(255.0)
    result = result.permute(0, 3, 1, 2).contiguous()
    mean = result.new_tensor(ms._CIFAR_MEAN).view(1, 3, 1, 1)
    std = result.new_tensor(ms._CIFAR_STD).view(1, 3, 1, 1)
    return (result - mean) / std


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing CIFAR cache: {path}")
    blob = np.load(path)
    return blob["images"], blob["labels"]


def load_split(config: Config, device: torch.device) -> Split:
    data_dir = Path(config.data_dir)
    train_images, train_labels = load_cache(data_dir / "cifar10_train.npz")
    test_images, test_labels = load_cache(data_dir / "cifar10_test.npz")
    objective_counts = exact_counts(config.objective_size, config.target_probs)
    validation_counts = exact_counts(config.validation_size, config.target_probs)
    pool_idx, objective_idx, validation_idx = select_split_indices(
        train_labels,
        pool_per_class=config.pool_per_class,
        objective_counts=objective_counts,
        validation_counts=validation_counts,
        seed=config.seed,
    )
    return Split(
        pool_images=normalize_images(train_images[pool_idx]).to(device),
        pool_labels=torch.from_numpy(train_labels[pool_idx]).long().to(device),
        objective_images=normalize_images(train_images[objective_idx]).to(device),
        objective_labels=torch.from_numpy(train_labels[objective_idx]).long().to(device),
        validation_images=normalize_images(train_images[validation_idx]).to(device),
        validation_labels=torch.from_numpy(train_labels[validation_idx]).long().to(device),
        test_images=normalize_images(test_images).to(device),
        test_labels=torch.from_numpy(test_labels).long().to(device),
        pool_source_indices=torch.from_numpy(pool_idx),
        objective_source_indices=torch.from_numpy(objective_idx),
        validation_source_indices=torch.from_numpy(validation_idx),
    )


def trajectory_indices(
    pool_size: int, *, epochs: int, batch_size: int, seed: int, device: torch.device
) -> tuple[Tensor, ...]:
    if pool_size % batch_size:
        raise ValueError("pool size must be divisible by batch size")
    generator = torch.Generator().manual_seed(seed)
    batches: list[Tensor] = []
    for _ in range(epochs):
        permutation = torch.randperm(pool_size, generator=generator)
        for start in range(0, pool_size, batch_size):
            batches.append(permutation[start:start + batch_size].to(device))
    return tuple(batches)


def make_trajectory(
    split: Split, indices: tuple[Tensor, ...], granularity: Granularity
) -> tuple[InnerBatch, ...]:
    if granularity == "per_class":
        group_ids = split.pool_labels
    else:
        group_ids = torch.arange(split.pool_labels.numel(), device=split.pool_labels.device)
    return tuple(
        InnerBatch(split.pool_images[index], split.pool_labels[index], group_ids[index])
        for index in indices
    )


def parameterization(
    split: Split, granularity: Granularity
) -> tuple[Tensor, Tensor]:
    device = split.pool_labels.device
    if granularity == "per_class":
        logits = torch.zeros(len(CLASSES), device=device, requires_grad=True)
        base_masses = torch.full((len(CLASSES),), 1.0 / len(CLASSES), device=device)
    else:
        size = split.pool_labels.numel()
        logits = torch.zeros(size, device=device, requires_grad=True)
        base_masses = torch.full((size,), 1.0 / size, device=device)
    return logits, base_masses


def class_mass_summary(
    logits: Tensor, split: Split, granularity: Granularity, temperature: float = 1.0
) -> dict:
    masses = group_masses(logits.detach(), temperature)
    if granularity == "per_class":
        class_masses = masses
        example_masses = masses[split.pool_labels] / torch.bincount(
            split.pool_labels, minlength=len(CLASSES)
        )[split.pool_labels]
    else:
        example_masses = masses
        class_masses = torch.zeros(len(CLASSES), device=masses.device)
        class_masses.scatter_add_(0, split.pool_labels, masses)

    class_ess = []
    for class_id in range(len(CLASSES)):
        values = example_masses[split.pool_labels == class_id]
        class_ess.append(float(values.sum().square() / values.square().sum()))
    return {
        "class_masses": class_masses.cpu().tolist(),
        "class_multipliers": (len(CLASSES) * class_masses).cpu().tolist(),
        "overall_ess": float(1.0 / example_masses.square().sum()),
        "entropy": float(-(example_masses * example_masses.log()).sum()),
        "class_ess": class_ess,
    }


def oracle_logits(
    split: Split, granularity: Granularity, target_probs: Tensor, temperature: float = 1.0
) -> Tensor:
    if granularity == "per_class":
        return temperature * target_probs.log()
    return temperature * target_probs[split.pool_labels].log()


def distribution_kl(new_masses: Tensor, old_masses: Tensor) -> Tensor:
    return (new_masses * (new_masses.log() - old_masses.log())).sum()


@torch.no_grad()
def apply_meta_update(
    logits: Tensor,
    gradient: Tensor,
    config: Config,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    old_masses = group_masses(logits, config.temperature)
    if optimizer is not None:
        logits.grad = gradient
        optimizer.step()
        optimizer.zero_grad()
        scale = config.meta_lr
    else:
        direction = -gradient.sign()
        if not torch.any(direction):
            return {"distribution_kl": 0.0, "logit_scale": 0.0}

        def kl_at(scale_value: float) -> float:
            candidate = group_masses(logits + scale_value * direction, config.temperature)
            return float(distribution_kl(candidate, old_masses))

        lower, upper = 0.0, 1.0
        while kl_at(upper) < config.step_kl and upper < 128.0:
            upper *= 2.0
        for _ in range(40):
            middle = 0.5 * (lower + upper)
            if kl_at(middle) < config.step_kl:
                lower = middle
            else:
                upper = middle
        scale = 0.5 * (lower + upper)
        logits.add_(direction, alpha=scale)

    logits.sub_(logits.mean())
    new_masses = group_masses(logits, config.temperature)
    return {
        "distribution_kl": float(distribution_kl(new_masses, old_masses)),
        "logit_scale": float(scale),
    }


def objective_value(model: ms.ResNet9, state: TrainState, batch: ObjectiveBatch) -> Tensor:
    predictions = functional_call(model, (state.parameters, state.buffers), (batch.images,))
    return F.cross_entropy(predictions, batch.labels)


@torch.no_grad()
def evaluate(
    model: ms.ResNet9,
    state: TrainState,
    images: Tensor,
    labels: Tensor,
    target_probs: Tensor,
    *,
    batch_size: int = 1000,
) -> dict:
    loss_sums = torch.zeros(len(CLASSES), device=images.device)
    correct = torch.zeros(len(CLASSES), device=images.device)
    counts = torch.zeros(len(CLASSES), device=images.device)
    for start in range(0, images.shape[0], batch_size):
        x = images[start:start + batch_size]
        y = labels[start:start + batch_size]
        predictions = functional_call(model, (state.parameters, state.buffers), (x,))
        losses = F.cross_entropy(predictions, y, reduction="none")
        ones = torch.ones_like(losses)
        loss_sums.scatter_add_(0, y, losses)
        correct.scatter_add_(0, y, (predictions.argmax(1) == y).to(losses.dtype))
        counts.scatter_add_(0, y, ones)
    per_class_ce = loss_sums / counts
    per_class_acc = correct / counts
    return {
        "target_ce": float((target_probs * per_class_ce).sum()),
        "target_acc": float((target_probs * per_class_acc).sum()),
        "balanced_ce": float(per_class_ce.mean()),
        "balanced_acc": float(per_class_acc.mean()),
        "per_class_ce": per_class_ce.cpu().tolist(),
        "per_class_acc": per_class_acc.cpu().tolist(),
        "counts": counts.long().cpu().tolist(),
    }


def gradient_summary(gradient: Tensor, split: Split, granularity: Granularity) -> dict:
    if granularity == "per_class":
        class_sum = gradient
    else:
        class_sum = torch.zeros(len(CLASSES), device=gradient.device)
        class_sum.scatter_add_(0, split.pool_labels, gradient)
    return {
        "norm": float(gradient.norm()),
        "abs_mean": float(gradient.abs().mean()),
        "positive_fraction": float((gradient > 0).float().mean()),
        "class_sum": class_sum.detach().cpu().tolist(),
    }


def state_checksum(state: TrainState) -> float:
    return float(sum(value.detach().double().sum() for value in state.parameters.values()))


def save_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def train_state(
    model: ms.ResNet9,
    trajectory: tuple[InnerBatch, ...],
    logits: Tensor,
    base_masses: Tensor,
    optimizer_config: SmoothAdamWConfig,
    config: Config,
) -> TrainState:
    def replay_step(state: TrainState, index: int, replay_logits: Tensor, create_graph: bool):
        batch = trajectory[index]
        next_state, _ = weighted_inner_step(
            model,
            state,
            batch.images,
            batch.labels,
            batch.group_ids,
            replay_logits,
            base_masses,
            optimizer_config,
            temperature=config.temperature,
            create_graph=create_graph,
        )
        return next_state

    initial_state = initialize_train_state(model)
    if config.method == "replay":
        return recursive_replay_state(
            initial_state,
            logits,
            len(trajectory),
            replay_step,
            branching_factor=config.branching_factor,
        )
    return train_unrolled(
        model,
        initial_state,
        trajectory,
        logits,
        base_masses,
        optimizer_config,
        temperature=config.temperature,
        create_graph=True,
    )


def run_method(
    config: Config,
    split: Split,
    batch_indices: tuple[Tensor, ...],
    target_probs: Tensor,
    granularity: Granularity,
    payload: dict,
) -> None:
    print(f"\n=== {granularity} ===", flush=True)
    configure_replay_determinism(config.seed, tf32=False)
    model = ms.ResNet9(ms.SMOOTH_GN_ROUTINE, num_classes=len(CLASSES)).to(split.pool_images.device)
    model.train()
    trajectory = make_trajectory(split, batch_indices, granularity)
    logits, base_masses = parameterization(split, granularity)
    optimizer_config = SmoothAdamWConfig(learning_rate=config.inner_lr, weight_decay=0.0)
    objective_batch = ObjectiveBatch(split.objective_images, split.objective_labels)
    outer_optimizer = (
        torch.optim.Adam([logits], lr=config.meta_lr)
        if config.meta_optimizer == "adam"
        else None
    )

    result = {"granularity": granularity, "history": []}
    payload["methods"][granularity] = result
    output_path = Path(config.output)
    start_time = time.perf_counter()
    snapshots: list[Tensor] = []

    for step in range(config.meta_steps + 1):
        step_start = time.perf_counter()
        snapshots.append(logits.detach().cpu().clone())
        final_state = train_state(
            model, trajectory, logits, base_masses, optimizer_config, config
        )
        objective = objective_value(model, final_state, objective_batch)
        objective_metrics = evaluate(
            model,
            final_state,
            split.objective_images,
            split.objective_labels,
            target_probs,
        )
        validation_metrics = evaluate(
            model,
            final_state,
            split.validation_images,
            split.validation_labels,
            target_probs,
        )
        weight_metrics = class_mass_summary(
            logits, split, granularity, config.temperature
        )
        target_l1 = float(
            torch.tensor(weight_metrics["class_masses"]).sub(
                target_probs.detach().cpu()
            ).abs().sum()
        )
        record = {
            "step": step,
            "objective_ce": float(objective.detach()),
            "objective": objective_metrics,
            "validation": validation_metrics,
            "validation_gap": validation_metrics["target_ce"] - float(objective.detach()),
            "weights": weight_metrics,
            "target_mass_l1": target_l1,
            "state_checksum": state_checksum(final_state),
            "minutes": (time.perf_counter() - step_start) / 60.0,
        }

        if step < config.meta_steps:
            (gradient,) = torch.autograd.grad(objective, logits)
            record["gradient"] = gradient_summary(gradient, split, granularity)
            record["update"] = apply_meta_update(
                logits, gradient, config, outer_optimizer
            )

        result["history"].append(record)
        save_result(output_path, payload)
        masses = weight_metrics["class_masses"]
        print(
            f"step {step:2d}: obj={record['objective_ce']:.4f} "
            f"val={validation_metrics['target_ce']:.4f}/{validation_metrics['target_acc']:.3f} "
            f"balanced-val={validation_metrics['balanced_ce']:.4f}/"
            f"{validation_metrics['balanced_acc']:.3f} "
            f"L1={target_l1:.3f} ESS={weight_metrics['overall_ess']:.0f} "
            f"[{record['minutes']:.1f}m] "
            f"class_mass={[round(value, 3) for value in masses]}",
            flush=True,
        )

    selected_step = min(
        range(len(result["history"])),
        key=lambda index: result["history"][index]["validation"]["target_ce"],
    )
    selected_logits = snapshots[selected_step].to(split.pool_images.device).requires_grad_(True)
    selected_state = train_state(
        model, trajectory, selected_logits, base_masses, optimizer_config, config
    )
    selected_history = result["history"][selected_step]
    selected_test = evaluate(
        model, selected_state, split.test_images, split.test_labels, target_probs
    )
    result["selected"] = {
        "selection_rule": "minimum skew-matched meta-validation CE",
        "step": selected_step,
        "objective_ce": selected_history["objective_ce"],
        "validation": selected_history["validation"],
        "test": selected_test,
        "test_gap": selected_test["target_ce"] - selected_history["objective_ce"],
        "weights": class_mass_summary(
            selected_logits, split, granularity, config.temperature
        ),
        "state_checksum": state_checksum(selected_state),
    }
    result["total_minutes"] = (time.perf_counter() - start_time) / 60.0
    save_result(output_path, payload)


def run_static_control(
    config: Config,
    split: Split,
    batch_indices: tuple[Tensor, ...],
    target_probs: Tensor,
    class_masses: Tensor,
) -> dict:
    """Evaluate a fixed class-mass distribution as a control."""
    configure_replay_determinism(config.seed, tf32=False)
    model = ms.ResNet9(ms.SMOOTH_GN_ROUTINE, num_classes=len(CLASSES)).to(split.pool_images.device)
    model.train()
    trajectory = make_trajectory(split, batch_indices, "per_class")
    logits = oracle_logits(
        split, "per_class", class_masses, config.temperature
    ).detach().requires_grad_(True)
    base_masses = torch.full_like(logits, 1.0 / len(CLASSES))
    state = train_state(
        model,
        trajectory,
        logits,
        base_masses,
        SmoothAdamWConfig(learning_rate=config.inner_lr, weight_decay=0.0),
        config,
    )
    objective = objective_value(
        model, state, ObjectiveBatch(split.objective_images, split.objective_labels)
    )
    objective_metrics = evaluate(
        model,
        state,
        split.objective_images,
        split.objective_labels,
        target_probs,
    )
    validation_metrics = evaluate(
        model,
        state,
        split.validation_images,
        split.validation_labels,
        target_probs,
    )
    test_metrics = evaluate(model, state, split.test_images, split.test_labels, target_probs)
    return {
        "objective_ce": float(objective.detach()),
        "objective": objective_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "validation_gap": validation_metrics["target_ce"] - float(objective.detach()),
        "test_gap": test_metrics["target_ce"] - float(objective.detach()),
        "weights": class_mass_summary(logits, split, "per_class", config.temperature),
        "state_checksum": state_checksum(state),
    }


def validate_config(config: Config) -> None:
    if config.pool_per_class <= 0 or config.objective_size <= 0 or config.validation_size <= 0:
        raise ValueError("pool_per_class, objective_size, and validation_size must be positive")
    if config.inner_epochs <= 0 or config.batch_size <= 0:
        raise ValueError("inner_epochs and batch_size must be positive")
    if config.meta_steps < 0 or config.meta_lr <= 0 or config.inner_lr <= 0:
        raise ValueError("learning rates must be positive and meta_steps non-negative")
    if config.step_kl <= 0:
        raise ValueError("step_kl must be positive")
    if config.pool_per_class * len(CLASSES) % config.batch_size:
        raise ValueError("balanced pool size must be divisible by batch_size")


def main(config: Config) -> None:
    validate_config(config)
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if config.device == "auto" else config.device
    device = torch.device(device_name)
    configure_replay_determinism(config.seed, tf32=False)
    split = load_split(config, device)
    target_probs = torch.tensor(config.target_probs, device=device)
    batch_indices = trajectory_indices(
        split.pool_labels.numel(),
        epochs=config.inner_epochs,
        batch_size=config.batch_size,
        seed=config.seed + 1,
        device=device,
    )
    exposure = torch.bincount(
        torch.cat(batch_indices), minlength=split.pool_labels.numel()
    )
    if not torch.all(exposure == config.inner_epochs):
        raise RuntimeError("inner trajectory does not expose every pool example once per epoch")
    objective_counts = torch.bincount(
        split.objective_labels, minlength=len(CLASSES)
    ).cpu().tolist()
    pool_counts = torch.bincount(split.pool_labels, minlength=len(CLASSES)).cpu().tolist()
    validation_counts = torch.bincount(
        split.validation_labels, minlength=len(CLASSES)
    ).cpu().tolist()
    payload = {
        "config": asdict(config),
        "design": {
            "classes": CLASSES,
            "pool_counts": pool_counts,
            "objective_counts": objective_counts,
            "validation_counts": validation_counts,
            "target_probs": config.target_probs,
            "test_counts": torch.bincount(
                split.test_labels, minlength=len(CLASSES)
            ).cpu().tolist(),
            "inner_steps": len(batch_indices),
            "comparison": (
                "persistent-softmax MGD with identical data, initialization, trajectory, "
                "inner optimizer, outer optimizer, and evaluation; only granularity differs"
            ),
        },
        "uniform": None,
        "oracle": None,
        "methods": {},
        "fairness_checks": {},
    }
    print(json.dumps(payload["design"], indent=2), flush=True)
    uniform_masses = torch.full_like(target_probs, 1.0 / len(CLASSES))
    payload["uniform"] = run_static_control(
        config, split, batch_indices, target_probs, uniform_masses
    )
    print(
        f"uniform: obj={payload['uniform']['objective_ce']:.4f} "
        f"val={payload['uniform']['validation']['target_ce']:.4f} "
        f"test={payload['uniform']['test']['target_ce']:.4f}/"
        f"{payload['uniform']['test']['target_acc']:.3f}",
        flush=True,
    )
    payload["oracle"] = run_static_control(
        config, split, batch_indices, target_probs, target_probs
    )
    print(
        f"oracle: obj={payload['oracle']['objective_ce']:.4f} "
        f"test={payload['oracle']['test']['target_ce']:.4f}/"
        f"{payload['oracle']['test']['target_acc']:.3f}",
        flush=True,
    )
    save_result(Path(config.output), payload)
    for granularity in config.methods:
        run_method(config, split, batch_indices, target_probs, granularity, payload)
    if set(config.methods) == {"per_class", "per_example"} and config.meta_steps > 0:
        class_zero = payload["methods"]["per_class"]["history"][0]
        example_zero = payload["methods"]["per_example"]["history"][0]
        class_gradient = torch.tensor(class_zero["gradient"]["class_sum"])
        example_gradient = torch.tensor(example_zero["gradient"]["class_sum"])
        fairness_checks = {
            "uniform_objective_abs_diff": abs(
                class_zero["objective_ce"] - example_zero["objective_ce"]
            ),
            "uniform_validation_abs_diff": abs(
                class_zero["validation"]["target_ce"]
                - example_zero["validation"]["target_ce"]
            ),
            "uniform_state_checksum_abs_diff": abs(
                class_zero["state_checksum"] - example_zero["state_checksum"]
            ),
            "initial_aggregated_gradient_max_abs_diff": float(
                (class_gradient - example_gradient).abs().max()
            ),
            "initial_aggregated_gradient_relative_l2": float(
                (class_gradient - example_gradient).norm()
                / class_gradient.norm().clamp_min(1e-12)
            ),
        }
        fairness_checks["passed"] = (
            fairness_checks["uniform_objective_abs_diff"] <= 1e-6
            and fairness_checks["uniform_validation_abs_diff"] <= 1e-6
            and fairness_checks["uniform_state_checksum_abs_diff"] <= 1e-5
            and fairness_checks["initial_aggregated_gradient_relative_l2"] <= 1e-4
        )
        payload["fairness_checks"] = fairness_checks
        print("fairness checks:", payload["fairness_checks"], flush=True)
        save_result(Path(config.output), payload)
        if not fairness_checks["passed"]:
            raise RuntimeError("per-class/per-example fairness preflight failed")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/workspace/tmp/data")
    parser.add_argument("--output", default="artifacts/mgd_granularity_primary.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-per-class", type=int, default=400)
    parser.add_argument("--objective-size", type=int, default=2000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument(
        "--target-probs",
        type=parse_probabilities,
        default=DEFAULT_TARGET,
        help="comma-separated probabilities in CIFAR-10 class order",
    )
    parser.add_argument("--inner-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--inner-lr", type=float, default=0.02)
    parser.add_argument("--meta-steps", type=int, default=12)
    parser.add_argument("--meta-lr", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--meta-optimizer", choices=("adam", "sign_kl"), default="sign_kl")
    parser.add_argument("--step-kl", type=float, default=0.0025)
    parser.add_argument("--method", choices=("replay", "store_all"), default="replay")
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--methods",
        default="per_class,per_example",
        help="comma-separated subset of per_class,per_example",
    )
    args = parser.parse_args()
    methods = tuple(args.methods.split(","))
    if any(method not in ("per_class", "per_example") for method in methods):
        parser.error("--methods must contain only per_class and per_example")
    return Config(
        data_dir=args.data_dir,
        output=args.output,
        seed=args.seed,
        pool_per_class=args.pool_per_class,
        objective_size=args.objective_size,
        validation_size=args.validation_size,
        target_probs=tuple(args.target_probs),
        inner_epochs=args.inner_epochs,
        batch_size=args.batch_size,
        inner_lr=args.inner_lr,
        meta_steps=args.meta_steps,
        meta_lr=args.meta_lr,
        temperature=args.temperature,
        meta_optimizer=args.meta_optimizer,
        step_kl=args.step_kl,
        method=args.method,
        branching_factor=args.branching_factor,
        device=args.device,
        methods=methods,
    )


if __name__ == "__main__":
    main(parse_args())
