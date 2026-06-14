from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

HF_DATASET_NAME = "tomas-gajarsky/cifar100-lt"
SUPPORTED_CONFIGS = ("r-10", "r-20", "r-50", "r-100")


@dataclass(frozen=True)
class CIFAR100LTData:
    train_images: Tensor
    train_labels: Tensor
    val_images: Tensor
    val_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    train_class_counts: Tensor
    config: str


def load_cifar100_lt(
    cache_dir: str | Path,
    *,
    config: str = "r-100",
    validation_per_class: int = 50,
    seed: int = 0,
) -> CIFAR100LTData:
    if config not in SUPPORTED_CONFIGS:
        raise ValueError(f"config must be one of {SUPPORTED_CONFIGS}")
    if validation_per_class <= 0 or validation_per_class >= 100:
        raise ValueError("validation_per_class must be in [1, 99]")
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_images, train_labels = _load_or_build_split(root, config=config, split="train")
    balanced_images, balanced_labels = _load_or_build_split(root, config=config, split="test")
    val_indices, test_indices = balanced_eval_split(
        balanced_labels,
        validation_per_class=validation_per_class,
        seed=seed,
    )
    train_images_tensor = torch.from_numpy(train_images)
    train_labels_tensor = torch.from_numpy(train_labels).to(torch.long)
    balanced_images_tensor = torch.from_numpy(balanced_images)
    balanced_labels_tensor = torch.from_numpy(balanced_labels).to(torch.long)
    return CIFAR100LTData(
        train_images=train_images_tensor,
        train_labels=train_labels_tensor,
        val_images=balanced_images_tensor.index_select(0, val_indices),
        val_labels=balanced_labels_tensor.index_select(0, val_indices),
        test_images=balanced_images_tensor.index_select(0, test_indices),
        test_labels=balanced_labels_tensor.index_select(0, test_indices),
        train_class_counts=torch.bincount(train_labels_tensor, minlength=100),
        config=config,
    )


def balanced_eval_split(
    labels: np.ndarray | Tensor,
    *,
    validation_per_class: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    num_classes = int(labels_tensor.max().item()) + 1
    generator = torch.Generator().manual_seed(seed)
    validation: list[Tensor] = []
    test: list[Tensor] = []
    for class_id in range(num_classes):
        class_indices = torch.nonzero(labels_tensor == class_id, as_tuple=False).flatten()
        if class_indices.numel() <= validation_per_class:
            raise ValueError(
                f"class {class_id} has {class_indices.numel()} examples, "
                f"not enough for validation_per_class={validation_per_class}"
            )
        order = torch.randperm(class_indices.numel(), generator=generator)
        validation.append(class_indices[order[:validation_per_class]])
        test.append(class_indices[order[validation_per_class:]])
    return torch.cat(validation), torch.cat(test)


def _load_or_build_split(root: Path, *, config: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    cache = root / f"cifar100_lt_{config}_{split}.npz"
    if cache.exists():
        payload = np.load(cache)
        return payload["images"], payload["labels"]

    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET_NAME, config, split=split, cache_dir=str(root / "hf"))
    images = np.empty((len(dataset), 32, 32, 3), dtype=np.uint8)
    labels = np.empty(len(dataset), dtype=np.int64)
    for index, row in enumerate(dataset):
        image = np.asarray(row["img"].convert("RGB"), dtype=np.uint8)
        if image.shape != (32, 32, 3):
            raise ValueError(f"unexpected image shape at row {index}: {image.shape}")
        images[index] = image
        labels[index] = int(row["fine_label"])
    np.savez(cache, images=images, labels=labels)
    return images, labels
