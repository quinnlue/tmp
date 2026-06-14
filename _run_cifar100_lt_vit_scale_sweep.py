from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from _cifar100_lt import CIFAR100LTData, load_cifar100_lt
from _vit_menu import SMOOTH_VIT, ViTGeometry
from model import VisionTransformerClassifier

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class ViTProfile:
    name: str
    dim: int
    depth: int
    heads: int
    mlp_ratio: float = 4.0
    patch: int = 4


PROFILES = {
    "tiny": ViTProfile("tiny", dim=192, depth=12, heads=3),
    "small": ViTProfile("small", dim=384, depth=12, heads=6),
    "base": ViTProfile("base", dim=768, depth=12, heads=12),
}


@dataclass(frozen=True)
class SweepConfig:
    cache_dir: str
    output: str
    checkpoint_dir: str
    dataset_config: str = "r-100"
    profiles: tuple[str, ...] = ("tiny", "small", "base")
    epoch_ranges: tuple[int, ...] = (25,)
    learning_rates: tuple[float, ...] = (1e-3, 5e-4, 2e-4)
    batch_size: int = 256
    eval_batch_size: int = 512
    weight_decay: float = 0.05
    warmup_fraction: float = 0.1
    seed: int = 0
    validation_per_class: int = 50
    device: str = "auto"
    amp: str = "bf16"


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    return result


def parse_args() -> SweepConfig:
    parser = argparse.ArgumentParser(
        description="Sweep corrected smooth ViT compute scales on Hugging Face CIFAR100-LT."
    )
    parser.add_argument("--cache-dir", default="./data/cifar100-lt")
    parser.add_argument("--output", default="cifar100_lt_vit_scale_sweep.json")
    parser.add_argument("--checkpoint-dir", default="checkpoints/cifar100-lt-vit-sweep")
    parser.add_argument("--dataset-config", choices=("r-10", "r-20", "r-50", "r-100"), default="r-100")
    parser.add_argument("--profiles", type=_csv_strings, default=("tiny", "small", "base"))
    parser.add_argument("--epoch-ranges", type=_csv_ints, default=(25,))
    parser.add_argument("--learning-rates", type=_csv_floats, default=(1e-3, 5e-4, 2e-4))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default="bf16")
    return SweepConfig(**vars(parser.parse_args()))


def validate_config(config: SweepConfig) -> None:
    unknown = set(config.profiles).difference(PROFILES)
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    if any(epoch <= 0 for epoch in config.epoch_ranges):
        raise ValueError("epoch_ranges must be positive")
    if any(lr <= 0 for lr in config.learning_rates):
        raise ValueError("learning_rates must be positive")
    if config.batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= config.warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def configure_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def normalize_images(images: Tensor, device: torch.device) -> Tensor:
    result = images.to(device=device, dtype=torch.float32).div(255.0).permute(0, 3, 1, 2).contiguous()
    mean = result.new_tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = result.new_tensor(CIFAR100_STD).view(1, 3, 1, 1)
    return (result - mean) / std


def augment(images: Tensor) -> Tensor:
    batch = images.shape[0]
    flipped = torch.rand(batch, device=images.device) < 0.5
    images = torch.where(flipped[:, None, None, None], images.flip(-1), images)
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    row_offsets = torch.randint(0, 9, (batch,), device=images.device)
    column_offsets = torch.randint(0, 9, (batch,), device=images.device)
    coordinates = torch.arange(32, device=images.device)
    rows = row_offsets[:, None] + coordinates[None, :]
    columns = column_offsets[:, None] + coordinates[None, :]
    padded = padded.gather(2, rows[:, None, :, None].expand(batch, 3, 32, 40))
    return padded.gather(3, columns[:, None, None, :].expand(batch, 3, 32, 32))


def learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
    peak_learning_rate: float,
) -> float:
    warmup_steps = round(total_steps * warmup_fraction)
    if warmup_steps > 0 and step < warmup_steps:
        return peak_learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def frequency_tiers(train_class_counts: Tensor) -> dict[str, Tensor]:
    return {
        "many": train_class_counts > 100,
        "medium": (train_class_counts >= 20) & (train_class_counts <= 100),
        "few": train_class_counts < 20,
    }


@torch.inference_mode()
def evaluate(
    model: VisionTransformerClassifier,
    images: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    amp: str,
    train_class_counts: Tensor,
) -> dict[str, Any]:
    model.eval()
    num_classes = int(train_class_counts.numel())
    loss_sums = torch.zeros(num_classes, dtype=torch.float64, device=images.device)
    correct_sums = torch.zeros_like(loss_sums)
    counts = torch.zeros_like(loss_sums)
    for start in range(0, images.shape[0], batch_size):
        inputs = images[start : start + batch_size]
        targets = labels[start : start + batch_size]
        with _autocast(images.device, amp):
            logits = model(inputs)
            losses = F.cross_entropy(logits.float(), targets, reduction="none")
        loss_sums.scatter_add_(0, targets, losses.to(torch.float64))
        correct_sums.scatter_add_(0, targets, logits.argmax(1).eq(targets).to(torch.float64))
        counts.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.float64))
    present = counts > 0
    per_class_loss = loss_sums[present] / counts[present]
    per_class_accuracy = correct_sums[present] / counts[present]
    result: dict[str, Any] = {
        "loss": float(loss_sums.sum().div(counts.sum()).item()),
        "accuracy": float(correct_sums.sum().div(counts.sum()).item()),
        "balanced_loss": float(per_class_loss.mean().item()),
        "balanced_accuracy": float(per_class_accuracy.mean().item()),
    }
    for name, tier in frequency_tiers(train_class_counts).items():
        tier = tier.to(images.device) & present
        result[f"{name}_accuracy"] = (
            float((correct_sums[tier] / counts[tier]).mean().item())
            if bool(tier.any())
            else float("nan")
        )
    return result


def _autocast(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "off":
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def build_model(profile: ViTProfile) -> VisionTransformerClassifier:
    geometry = ViTGeometry(
        image_size=32,
        patch=profile.patch,
        dim=profile.dim,
        depth=profile.depth,
        heads=profile.heads,
        mlp_ratio=profile.mlp_ratio,
    )
    return VisionTransformerClassifier(SMOOTH_VIT.build_config(100, geometry))


def run_id(profile: str, epochs: int, learning_rate: float) -> str:
    lr_text = f"{learning_rate:.0e}".replace("+", "")
    return f"{profile}-e{epochs}-lr{lr_text}"


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_one(
    config: SweepConfig,
    profile: ViTProfile,
    *,
    epochs: int,
    learning_rate: float,
    data: CIFAR100LTData,
    device: torch.device,
    output_payload: dict[str, Any],
) -> dict[str, Any]:
    identifier = run_id(profile.name, epochs, learning_rate)
    # Keep initialization, minibatch order, and augmentation randomness matched
    # across learning-rate candidates within each model profile.
    configure_training(config.seed)
    checkpoint_root = Path(config.checkpoint_dir)
    last_path = checkpoint_root / f"{identifier}.last.pt"
    best_path = checkpoint_root / f"{identifier}.best.pt"
    model = build_model(profile).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    train_images = normalize_images(data.train_images, device)
    train_labels = data.train_labels.to(device)
    val_images = normalize_images(data.val_images, device)
    val_labels = data.val_labels.to(device)
    test_images = normalize_images(data.test_images, device)
    test_labels = data.test_labels.to(device)
    train_counts = data.train_class_counts.to(device)
    steps_per_epoch = math.ceil(train_labels.numel() / config.batch_size)
    total_steps = epochs * steps_per_epoch
    history: list[dict[str, Any]] = []
    start_epoch = 0
    global_step = 0
    best_val = -math.inf
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint["best_val"])

    generator = torch.Generator(device=device).manual_seed(config.seed + start_epoch + 1)
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_labels.numel(), generator=generator, device=device)
        loss_sum = 0.0
        correct = 0
        epoch_started = time.perf_counter()
        for batch_start in range(0, train_labels.numel(), config.batch_size):
            indices = permutation[batch_start : batch_start + config.batch_size]
            inputs = augment(train_images.index_select(0, indices))
            targets = train_labels.index_select(0, indices)
            lr = learning_rate_at_step(
                global_step,
                total_steps=total_steps,
                warmup_fraction=config.warmup_fraction,
                peak_learning_rate=learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * targets.numel()
            correct += int(logits.detach().argmax(1).eq(targets).sum().item())
            global_step += 1
        val_metrics = evaluate(
            model,
            val_images,
            val_labels,
            batch_size=config.eval_batch_size,
            amp=config.amp,
            train_class_counts=train_counts,
        )
        elapsed = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / train_labels.numel(),
            "train_accuracy": correct / train_labels.numel(),
            "examples_per_second": train_labels.numel() / elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val": val_metrics,
        }
        history.append(record)
        if val_metrics["balanced_accuracy"] > best_val:
            best_val = val_metrics["balanced_accuracy"]
            save_checkpoint(
                best_path,
                {"model": model.state_dict(), "epoch": epoch, "val": val_metrics},
            )
        save_checkpoint(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "history": history,
            },
        )
        output_payload["runs"][identifier] = {
            "status": "running",
            "profile": asdict(profile),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
        }
        save_json(config.output, output_payload)
        print(
            f"{identifier} epoch {epoch:3d}/{epochs}: "
            f"train={record['train_accuracy']:.4f} "
            f"val_bal={val_metrics['balanced_accuracy']:.4f} "
            f"few={val_metrics['few_accuracy']:.4f} "
            f"{record['examples_per_second']:.0f} img/s",
            flush=True,
        )

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(
        model,
        test_images,
        test_labels,
        batch_size=config.eval_batch_size,
        amp=config.amp,
        train_class_counts=train_counts,
    )
    return {
        "status": "completed",
        "profile": asdict(profile),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(best["epoch"]),
        "best_val": best["val"],
        "test": test_metrics,
        "history": history,
        "minutes": (time.perf_counter() - started) / 60.0,
        "best_checkpoint": str(best_path.resolve()),
    }


def main() -> None:
    config = parse_args()
    validate_config(config)
    configure_training(config.seed)
    device = resolve_device(config.device)
    data = load_cifar100_lt(
        config.cache_dir,
        config=config.dataset_config,
        validation_per_class=config.validation_per_class,
        seed=config.seed,
    )
    payload: dict[str, Any]
    if Path(config.output).exists():
        payload = json.loads(Path(config.output).read_text(encoding="utf-8"))
    else:
        payload = {
            "config": asdict(config),
            "dataset": {
                "train_examples": int(data.train_labels.numel()),
                "val_examples": int(data.val_labels.numel()),
                "test_examples": int(data.test_labels.numel()),
                "train_class_counts": data.train_class_counts.tolist(),
            },
            "smooth_routine": asdict(SMOOTH_VIT),
            "runs": {},
        }
    for profile_name in config.profiles:
        for epochs in config.epoch_ranges:
            for learning_rate in config.learning_rates:
                identifier = run_id(profile_name, epochs, learning_rate)
                if payload["runs"].get(identifier, {}).get("status") == "completed":
                    print(f"skip completed {identifier}", flush=True)
                    continue
                result = train_one(
                    config,
                    PROFILES[profile_name],
                    epochs=epochs,
                    learning_rate=learning_rate,
                    data=data,
                    device=device,
                    output_payload=payload,
                )
                payload["runs"][identifier] = result
                save_json(config.output, payload)
                print(
                    f"completed {identifier}: test_bal={result['test']['balanced_accuracy']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
