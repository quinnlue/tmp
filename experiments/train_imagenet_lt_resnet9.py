from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

import metasmooth as ms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: str
    output: str
    checkpoint: str
    epochs: int = 10
    batch_size: int = 256
    eval_batch_size: int = 512
    workers: int = 8
    image_size: int = 128
    learning_rate: float = 1.0
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_fraction: float = 0.1
    amp: str = "bf16"
    device: str = "auto"
    seed: int = 0
    max_train_examples: int | None = None
    max_val_examples: int | None = None
    max_test_examples: int | None = None
    shuffle_chunk_size: int = 1
    many_threshold: int = 100
    few_threshold: int = 20


class PreprocessedDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        dataset: Dataset[tuple[Tensor, int]],
        *,
        image_size: int,
        training: bool,
    ) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return preprocess_image(image, image_size=self.image_size, training=self.training), int(label)


class StorageBlockShuffleSampler(Sampler[int]):
    """Shuffle parquet row groups while preserving cache-friendly reads within each group."""

    def __init__(
        self,
        blocks: Sequence[Sequence[int]],
        *,
        seed: int,
        chunk_size: int | None = None,
    ) -> None:
        chunks: list[tuple[int, ...]] = []
        for block in blocks:
            values = tuple(int(index) for index in block)
            size = len(values) if chunk_size is None else chunk_size
            chunks.extend(values[start : start + size] for start in range(0, len(values), size))
        self.blocks = tuple(chunks)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return sum(len(block) for block in self.blocks)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        block_order = torch.randperm(len(self.blocks), generator=generator).tolist()
        for block_index in block_order:
            block = self.blocks[block_index]
            within_block = torch.randperm(len(block), generator=generator).tolist()
            yield from (block[offset] for offset in within_block)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train the best prior accuracy/metasmooth ResNet-9 on ImageNet-LT parquet."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--shuffle-chunk-size", type=int, default=1)
    parser.add_argument("--many-threshold", type=int, default=100)
    parser.add_argument("--few-threshold", type=int, default=20)
    return TrainingConfig(**vars(parser.parse_args()))


def validate_config(config: TrainingConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.workers < 0:
        raise ValueError("workers must be non-negative")
    if config.image_size <= 0 or config.image_size % 8:
        raise ValueError("image_size must be positive and divisible by 8")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= config.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= config.warmup_fraction < 1:
        raise ValueError("warmup_fraction must be in [0, 1)")
    for name in ("max_train_examples", "max_val_examples", "max_test_examples"):
        value = getattr(config, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided")
    if config.shuffle_chunk_size <= 0:
        raise ValueError("shuffle_chunk_size must be positive")
    if config.few_threshold <= 0 or config.many_threshold < config.few_threshold:
        raise ValueError("class-frequency thresholds are invalid")


def resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
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


def preprocess_image(
    image: Tensor,
    *,
    image_size: int,
    training: bool,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected RGB image tensor with shape [3, height, width]")
    image = image.to(dtype=torch.float32).div(255.0).unsqueeze(0)
    if training:
        resize_size = max(image_size + 16, round(image_size * 1.125))
        image = F.interpolate(image, size=(resize_size, resize_size), mode="bilinear", align_corners=False)
        max_offset = resize_size - image_size
        top = int(torch.randint(max_offset + 1, ()).item())
        left = int(torch.randint(max_offset + 1, ()).item())
        image = image[:, :, top : top + image_size, left : left + image_size]
        if bool(torch.rand(()) < 0.5):
            image = image.flip(-1)
    elif image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean_tensor = image.new_tensor(tuple(mean)).view(1, 3, 1, 1)
    std_tensor = image.new_tensor(tuple(std)).view(1, 3, 1, 1)
    return ((image - mean_tensor) / std_tensor).squeeze(0)


def _initialize_worker(worker_id: int) -> None:
    del worker_id
    torch.set_num_threads(1)


def _limit_dataset(dataset: Dataset[tuple[Tensor, int]], maximum: int | None) -> Dataset[tuple[Tensor, int]]:
    if maximum is None or maximum >= len(dataset):
        return dataset
    return Subset(dataset, range(maximum))


def storage_blocks(dataset: Dataset[tuple[Tensor, int]], *, fallback_size: int = 100) -> list[range]:
    shards = getattr(dataset, "shards", None)
    if shards is not None:
        return [
            range(
                int(shard.start_index + row_group.start_index),
                int(shard.start_index + row_group.stop_index),
            )
            for shard in shards
            for row_group in shard.row_groups
        ]
    return [range(start, min(start + fallback_size, len(dataset))) for start in range(0, len(dataset), fallback_size)]


def build_dataloader(
    dataset: Dataset[tuple[Tensor, int]],
    *,
    batch_size: int,
    workers: int,
    image_size: int,
    training: bool,
    seed: int = 0,
    shuffle_chunk_size: int = 1,
) -> DataLoader[tuple[Tensor, Tensor]]:
    if workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    processed = PreprocessedDataset(dataset, image_size=image_size, training=training)
    sampler = (
        StorageBlockShuffleSampler(
            storage_blocks(dataset),
            seed=seed,
            chunk_size=shuffle_chunk_size,
        )
        if training
        else None
    )
    return DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_initialize_worker if workers > 0 else None,
    )


def learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_learning_rate * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _autocast_context(device: torch.device, amp: str) -> ContextManager[Any]:
    if device.type != "cuda" or amp == "off":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _class_metric(values: Tensor, counts: Tensor, mask: Tensor | None = None) -> float:
    present = counts > 0
    if mask is not None:
        present &= mask
    if not bool(present.any()):
        return float("nan")
    return float((values[present] / counts[present]).mean().item())


def classification_metrics(
    *,
    loss_sums: Tensor,
    correct_sums: Tensor,
    top5_sums: Tensor,
    counts: Tensor,
    train_class_counts: Tensor,
    many_threshold: int = 100,
    few_threshold: int = 20,
) -> dict[str, float]:
    total = counts.sum()
    many = train_class_counts > many_threshold
    few = train_class_counts < few_threshold
    medium = ~(many | few)
    return {
        "loss": float(loss_sums.sum().div(total).item()),
        "accuracy": float(correct_sums.sum().div(total).item()),
        "top5_accuracy": float(top5_sums.sum().div(total).item()),
        "balanced_loss": _class_metric(loss_sums, counts),
        "balanced_accuracy": _class_metric(correct_sums, counts),
        "many_accuracy": _class_metric(correct_sums, counts, many),
        "medium_accuracy": _class_metric(correct_sums, counts, medium),
        "few_accuracy": _class_metric(correct_sums, counts, few),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
    amp: str,
    num_classes: int,
    train_class_counts: Tensor,
    many_threshold: int = 100,
    few_threshold: int = 20,
) -> dict[str, float]:
    model.eval()
    loss_sums = torch.zeros(num_classes, dtype=torch.float64)
    correct_sums = torch.zeros(num_classes, dtype=torch.float64)
    top5_sums = torch.zeros(num_classes, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast_context(device, amp):
            logits = model(images)
            losses = F.cross_entropy(logits, labels, reduction="none")
        predictions = logits.argmax(dim=1)
        top5 = logits.topk(min(5, num_classes), dim=1).indices.eq(labels[:, None]).any(dim=1)
        cpu_labels = labels.cpu()
        counts += torch.bincount(cpu_labels, minlength=num_classes)
        loss_sums.scatter_add_(0, cpu_labels, losses.float().cpu().to(torch.float64))
        correct_sums.scatter_add_(0, cpu_labels, predictions.eq(labels).cpu().to(torch.float64))
        top5_sums.scatter_add_(0, cpu_labels, top5.cpu().to(torch.float64))
    return classification_metrics(
        loss_sums=loss_sums,
        correct_sums=correct_sums,
        top5_sums=top5_sums,
        counts=counts,
        train_class_counts=train_class_counts.to(torch.float64),
        many_threshold=many_threshold,
        few_threshold=few_threshold,
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: str,
    scaler: torch.amp.GradScaler | None,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_sum = 0.0
    correct = 0
    examples = 0
    started = time.perf_counter()
    for images, labels in dataloader:
        learning_rate = learning_rate_at_step(
            global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_learning_rate=peak_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        batch_size = labels.numel()
        loss_sum += float(loss.detach().item()) * batch_size
        correct += int(logits.detach().argmax(dim=1).eq(labels).sum().item())
        examples += batch_size
        global_step += 1
    elapsed = time.perf_counter() - started
    return {
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
        "examples_per_second": examples / elapsed,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }, global_step


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "routine": asdict(ms.SMOOTH_ROUTINE),
            "config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(destination)


def train(
    config: TrainingConfig,
    *,
    dataset_factory: Callable[..., Dataset[tuple[Tensor, int]]] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    configure_training(config.seed)
    device = resolve_device(config.device)
    if config.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 AMP was requested but is unsupported on this CUDA device")

    if dataset_factory is None:
        from imagenet_lt_parquet import ImageNetLTParquetDataset

        dataset_factory = ImageNetLTParquetDataset

    raw_train = dataset_factory(config.dataset_root, split="train")
    raw_val = dataset_factory(config.dataset_root, split="val")
    raw_test = dataset_factory(config.dataset_root, split="test")
    num_classes = int(max(int(raw_train.labels.max().item()) + 1, getattr(raw_train, "num_classes", 0)))
    train_class_counts = torch.bincount(raw_train.labels.to(torch.long), minlength=num_classes)

    train_dataset = _limit_dataset(raw_train, config.max_train_examples)
    val_dataset = _limit_dataset(raw_val, config.max_val_examples)
    test_dataset = _limit_dataset(raw_test, config.max_test_examples)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        workers=config.workers,
        image_size=config.image_size,
        training=True,
        seed=config.seed,
        shuffle_chunk_size=config.shuffle_chunk_size,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.eval_batch_size,
        workers=config.workers,
        image_size=config.image_size,
        training=False,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config.eval_batch_size,
        workers=config.workers,
        image_size=config.image_size,
        training=False,
    )

    model = ms.ResNet9(ms.SMOOTH_ROUTINE, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and config.amp == "fp16" else None
    total_steps = config.epochs * len(train_loader)
    warmup_steps = round(config.warmup_fraction * total_steps)
    result: dict[str, Any] = {
        "config": asdict(config),
        "routine": asdict(ms.SMOOTH_ROUTINE),
        "device": str(device),
        "num_classes": num_classes,
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "epochs": [],
    }
    best_balanced_accuracy = -math.inf
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            amp=config.amp,
            scaler=scaler,
            global_step=global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_learning_rate=config.learning_rate,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            amp=config.amp,
            num_classes=num_classes,
            train_class_counts=train_class_counts,
            many_threshold=config.many_threshold,
            few_threshold=config.few_threshold,
        )
        epoch_result = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        result["epochs"].append(epoch_result)
        if val_metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics["balanced_accuracy"]
            result["best_epoch"] = epoch
            result["best_val"] = val_metrics
            _save_checkpoint(config.checkpoint, model=model, config=config, epoch=epoch, metrics=val_metrics)
        _save_json(config.output, result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)

    checkpoint = torch.load(config.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result["test"] = evaluate(
        model,
        test_loader,
        device=device,
        amp=config.amp,
        num_classes=num_classes,
        train_class_counts=train_class_counts,
        many_threshold=config.many_threshold,
        few_threshold=config.few_threshold,
    )
    _save_json(config.output, result)
    print(json.dumps({"best_epoch": result["best_epoch"], "test": result["test"]}, sort_keys=True), flush=True)
    return result


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
