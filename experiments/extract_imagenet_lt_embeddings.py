from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_BACKBONES = ("dinov2_vitb14", "vit_b_16", "resnet50")


@dataclass(frozen=True)
class ExtractionConfig:
    dataset_root: str
    output: str
    split: str = "train"
    backbone: str = "vit_b_16"
    batch_size: int = 32
    workers: int = 0
    device: str = "auto"
    max_examples: int | None = None
    index_artifact: str | None = None
    index_key: str = "auto"
    random_weights: bool = False
    seed: int = 0
    save_every_batches: int = 20


@dataclass(frozen=True)
class FrozenBackbone:
    name: str
    image_size: int
    embedding_dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    module: nn.Module
    encoder: Callable[[nn.Module, Tensor], Tensor]

    def encode(self, images: Tensor) -> Tensor:
        embeddings = self.encoder(self.module, images)
        if embeddings.ndim != 2:
            raise ValueError(f"expected rank-2 embeddings, got shape {tuple(embeddings.shape)}")
        return embeddings


@dataclass(frozen=True)
class ExtractionSummary:
    output: str
    split: str
    backbone: str
    completed: int
    requested_examples: int
    total_dataset_examples: int
    embedding_dim: int
    random_weights: bool
    device: str


class IndexedDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(
        self,
        dataset: Dataset[tuple[Tensor, int]],
        *,
        image_size: int | None = None,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        image, label = self.dataset[index]
        if self.image_size is not None:
            image = preprocess_image(
                image,
                image_size=self.image_size,
                mean=self.mean,
                std=self.std,
            )
        return image, int(label), int(index)


def _default_dataset_factory(root: str, *, split: str) -> Dataset[tuple[Tensor, int]]:
    from imagenet_lt_parquet import ImageNetLTParquetDataset

    return ImageNetLTParquetDataset(root, split=split)


def parse_args() -> ExtractionConfig:
    parser = argparse.ArgumentParser(
        description="Extract frozen embeddings from the lazy ImageNet-LT parquet adapter."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default="vit_b_16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--index-artifact", default=None)
    parser.add_argument("--index-key", default="auto")
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every-batches", type=int, default=20)
    args = parser.parse_args()
    return ExtractionConfig(
        dataset_root=args.dataset_root,
        output=args.output,
        split=args.split,
        backbone=args.backbone,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        max_examples=args.max_examples,
        index_artifact=args.index_artifact,
        index_key=args.index_key,
        random_weights=args.random_weights,
        seed=args.seed,
        save_every_batches=args.save_every_batches,
    )


def validate_config(config: ExtractionConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.workers < 0:
        raise ValueError("workers must be non-negative")
    if config.max_examples is not None and config.max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    if config.save_every_batches <= 0:
        raise ValueError("save_every_batches must be positive")


def resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def configure_extraction_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def preprocess_image(
    image: Tensor,
    *,
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("expected image tensor with shape [channels, height, width]")
    if image.shape[0] != 3:
        raise ValueError("expected RGB image tensor with three channels")
    normalized = image.to(dtype=torch.float32).div(255.0).unsqueeze(0)
    if normalized.shape[-2:] != (image_size, image_size):
        normalized = F.interpolate(
            normalized,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    mean_tensor = normalized.new_tensor(tuple(mean)).view(1, 3, 1, 1)
    std_tensor = normalized.new_tensor(tuple(std)).view(1, 3, 1, 1)
    return ((normalized - mean_tensor) / std_tensor).squeeze(0)


def collate_indexed_batch(
    batch: Sequence[tuple[Tensor, int, int]],
) -> tuple[list[Tensor], Tensor, Tensor]:
    # ImageNet images have variable spatial sizes. Keep them as a list until
    # deterministic resize/normalization immediately before the forward pass.
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    global_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, labels, global_indices


def _ensure_long_vector(name: str, value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer indices")
    return value.to(dtype=torch.long, device="cpu")


def _extract_indices_from_payload(payload: Any, *, index_key: str) -> tuple[Tensor, str | None]:
    if isinstance(payload, Tensor):
        return _ensure_long_vector("index_artifact", payload), None
    if not isinstance(payload, dict):
        raise ValueError("index artifact must be a tensor or dictionary")

    if index_key != "auto":
        if index_key not in payload:
            raise ValueError(f"index artifact does not contain key {index_key!r}")
        return _ensure_long_vector(f"index_artifact[{index_key!r}]", payload[index_key]), index_key

    preferred_keys = (
        "indices",
        "train_indices",
        "split_train_indices",
        "objective_indices",
        "meta_validation_indices",
    )
    for candidate in preferred_keys:
        if candidate in payload and isinstance(payload[candidate], Tensor):
            return _ensure_long_vector(f"index_artifact[{candidate!r}]", payload[candidate]), candidate
    tensor_keys = [key for key, value in payload.items() if isinstance(value, Tensor) and value.ndim == 1]
    if len(tensor_keys) == 1:
        key = tensor_keys[0]
        return _ensure_long_vector(f"index_artifact[{key!r}]", payload[key]), key
    raise ValueError("index artifact auto-discovery expected a single one-dimensional tensor key")


def load_requested_global_indices(
    *,
    index_artifact: str | None,
    index_key: str,
    total_examples: int,
    max_examples: int | None,
) -> tuple[Tensor, str | None]:
    if index_artifact is None:
        indices = torch.arange(total_examples, dtype=torch.long)
        if max_examples is not None:
            indices = indices[:max_examples]
        return indices, None

    payload = torch.load(index_artifact, map_location="cpu")
    indices, resolved_key = _extract_indices_from_payload(payload, index_key=index_key)
    if indices.numel() == 0:
        raise ValueError("index artifact must select at least one example")
    if torch.any(indices < 0) or torch.any(indices >= total_examples):
        raise ValueError("index artifact contains an out-of-range example index")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("index artifact must not contain duplicate example indices")
    if max_examples is not None:
        indices = indices[:max_examples]
    if indices.numel() == 0:
        raise ValueError("selection is empty after applying index_artifact/max_examples")
    return indices, resolved_key


def _encode_vit_b_16(module: nn.Module, images: Tensor) -> Tensor:
    batch_size = images.shape[0]
    vit = module
    tokens = vit._process_input(images)
    class_token = vit.class_token.expand(batch_size, -1, -1)
    encoded = vit.encoder(torch.cat((class_token, tokens), dim=1))
    return encoded[:, 0]


def _encode_resnet50(module: nn.Module, images: Tensor) -> Tensor:
    resnet = module
    x = resnet.conv1(images)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)
    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)
    x = resnet.avgpool(x)
    return torch.flatten(x, 1)


def _encode_dinov2(module: nn.Module, images: Tensor) -> Tensor:
    features = module.forward_features(images)
    if isinstance(features, dict):
        class_token = features.get("x_norm_clstoken")
        if isinstance(class_token, Tensor):
            return class_token
    if isinstance(features, Tensor):
        return features
    raise ValueError("DINOv2 forward_features did not return a class-token embedding")


def load_torchvision_backbone(name: str, *, random_weights: bool) -> FrozenBackbone:
    normalized = name.strip().lower()
    if normalized == "dinov2_vitb14":
        model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=not random_weights,
        )
        return FrozenBackbone(
            name="dinov2_vitb14",
            image_size=224,
            embedding_dim=768,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            module=model,
            encoder=_encode_dinov2,
        )
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for frozen embedding extraction."
        ) from exc

    if normalized == "vit_b_16":
        weights = None if random_weights else models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        return FrozenBackbone(
            name="vit_b_16",
            image_size=224,
            embedding_dim=768,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            module=model,
            encoder=_encode_vit_b_16,
        )
    if normalized == "resnet50":
        weights = None if random_weights else models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        return FrozenBackbone(
            name="resnet50",
            image_size=224,
            embedding_dim=2048,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            module=model,
            encoder=_encode_resnet50,
        )
    raise ValueError(f"unsupported backbone: {name!r}")


def _base_artifact_metadata(
    config: ExtractionConfig,
    *,
    backbone: FrozenBackbone,
    total_dataset_examples: int,
    requested_global_indices: Sequence[int],
    resolved_index_key: str | None,
) -> dict[str, Any]:
    return {
        "dataset_root": str(Path(config.dataset_root).resolve()),
        "split": config.split,
        "backbone": backbone.name,
        "image_size": backbone.image_size,
        "mean": list(backbone.mean),
        "std": list(backbone.std),
        "random_weights": config.random_weights,
        "total_dataset_examples": total_dataset_examples,
        "target_examples": len(requested_global_indices),
        "index_artifact": (
            None if config.index_artifact is None else str(Path(config.index_artifact).resolve())
        ),
        "index_key": resolved_index_key,
        "requested_global_indices": [int(index) for index in requested_global_indices],
    }


def initialize_artifact(
    metadata: dict[str, Any],
    *,
    target_examples: int,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "completed": 0,
        "embedding_dim": embedding_dim,
        "embeddings": torch.empty((target_examples, embedding_dim), dtype=torch.float32),
        "labels": torch.empty((target_examples,), dtype=torch.long),
        "global_indices": torch.empty((target_examples,), dtype=torch.long),
    }


def load_existing_artifact(path: str | Path) -> dict[str, Any] | None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    artifact = torch.load(artifact_path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError(f"artifact at {artifact_path} must be a dictionary")
    return artifact


def validate_or_resume_artifact(
    artifact: dict[str, Any],
    *,
    metadata: dict[str, Any],
    embedding_dim: int,
) -> int:
    existing_metadata = artifact.get("metadata")
    if existing_metadata != metadata:
        raise ValueError("existing artifact metadata does not match the requested extraction")
    if int(artifact.get("embedding_dim", -1)) != embedding_dim:
        raise ValueError("existing artifact embedding_dim does not match the requested backbone")
    embeddings = artifact.get("embeddings")
    labels = artifact.get("labels")
    global_indices = artifact.get("global_indices")
    if not isinstance(embeddings, Tensor) or not isinstance(labels, Tensor) or not isinstance(global_indices, Tensor):
        raise ValueError("existing artifact is missing required tensors")
    completed = int(artifact.get("completed", -1))
    target_examples = int(metadata["target_examples"])
    requested_global_indices = metadata.get("requested_global_indices")
    if not isinstance(requested_global_indices, list):
        requested_global_indices = list(range(target_examples))
    if len(requested_global_indices) != target_examples:
        raise ValueError("existing artifact requested-index metadata does not match target_examples")
    if embeddings.shape != (target_examples, embedding_dim):
        raise ValueError("existing artifact embeddings tensor has the wrong shape")
    if labels.shape != (target_examples,) or global_indices.shape != (target_examples,):
        raise ValueError("existing artifact label/index tensors have the wrong shape")
    if completed < 0 or completed > target_examples:
        raise ValueError("existing artifact completed count is invalid")
    if completed > 0:
        expected_indices = torch.tensor(requested_global_indices[:completed], dtype=torch.long)
        actual_indices = global_indices[:completed].to(torch.long)
        if not torch.equal(actual_indices, expected_indices):
            raise ValueError("existing artifact global indices do not match the requested index prefix")
    return completed


def save_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(destination)


def build_dataloader(
    dataset: Dataset[tuple[Tensor, int, int]],
    *,
    batch_size: int,
    workers: int,
) -> DataLoader[tuple[list[Tensor], Tensor, Tensor]]:
    if workers > 0:
        # Avoid exhausting per-process file descriptors while workers transfer
        # preprocessed tensors back to the main process.
        torch.multiprocessing.set_sharing_strategy("file_system")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_indexed_batch,
        pin_memory=False,
        worker_init_fn=_initialize_worker if workers > 0 else None,
    )


def _initialize_worker(worker_id: int) -> None:
    del worker_id
    # Each worker performs many small image resizes. Letting every worker
    # inherit the host's full CPU thread pool causes severe oversubscription.
    torch.set_num_threads(1)


def extract_embeddings(
    config: ExtractionConfig,
    *,
    dataset_factory: Callable[..., Dataset[tuple[Tensor, int]]] = _default_dataset_factory,
    backbone_loader: Callable[[str], FrozenBackbone] | None = None,
) -> ExtractionSummary:
    validate_config(config)
    configure_extraction_determinism(config.seed)
    device = resolve_device(config.device)
    dataset = dataset_factory(config.dataset_root, split=config.split)
    total_dataset_examples = len(dataset)
    requested_global_indices, resolved_index_key = load_requested_global_indices(
        index_artifact=config.index_artifact,
        index_key=config.index_key,
        total_examples=total_dataset_examples,
        max_examples=config.max_examples,
    )
    target_examples = int(requested_global_indices.numel())
    loader = backbone_loader or (lambda name: load_torchvision_backbone(name, random_weights=config.random_weights))
    backbone = loader(config.backbone)
    backbone.module.eval()
    for parameter in backbone.module.parameters():
        parameter.requires_grad_(False)
    backbone.module.to(device)

    metadata = _base_artifact_metadata(
        config,
        backbone=backbone,
        total_dataset_examples=total_dataset_examples,
        requested_global_indices=requested_global_indices.tolist(),
        resolved_index_key=resolved_index_key,
    )
    artifact = load_existing_artifact(config.output)
    completed = 0
    if artifact is not None:
        completed = validate_or_resume_artifact(
            artifact,
            metadata=metadata,
            embedding_dim=backbone.embedding_dim,
        )
    if artifact is None:
        artifact = initialize_artifact(
            metadata,
            target_examples=target_examples,
            embedding_dim=backbone.embedding_dim,
        )
    if completed >= target_examples:
        return ExtractionSummary(
            output=str(Path(config.output).resolve()),
            split=config.split,
            backbone=backbone.name,
            completed=completed,
            requested_examples=target_examples,
            total_dataset_examples=total_dataset_examples,
            embedding_dim=backbone.embedding_dim,
            random_weights=config.random_weights,
            device=str(device),
        )

    indexed_dataset = IndexedDataset(
        dataset,
        image_size=backbone.image_size,
        mean=backbone.mean,
        std=backbone.std,
    )
    pending_subset = Subset(indexed_dataset, requested_global_indices[completed:].tolist())
    dataloader = build_dataloader(
        pending_subset,
        batch_size=config.batch_size,
        workers=config.workers,
    )

    offset = completed
    with torch.inference_mode():
        for batch_index, (images, labels, global_indices) in enumerate(dataloader, start=1):
            preprocessed = torch.stack(images, dim=0).to(device)
            embeddings = backbone.encode(preprocessed).detach().to(dtype=torch.float32, device="cpu")
            batch_size = embeddings.shape[0]
            # Labels are persisted for alignment only; they are not used to drive embedding extraction.
            artifact["embeddings"][offset : offset + batch_size] = embeddings
            artifact["labels"][offset : offset + batch_size] = labels.to(dtype=torch.long)
            artifact["global_indices"][offset : offset + batch_size] = global_indices.to(dtype=torch.long)
            offset += batch_size
            artifact["completed"] = offset
            if batch_index % config.save_every_batches == 0 or offset >= target_examples:
                save_artifact(config.output, artifact)

    return ExtractionSummary(
        output=str(Path(config.output).resolve()),
        split=config.split,
        backbone=backbone.name,
        completed=offset,
        requested_examples=target_examples,
        total_dataset_examples=total_dataset_examples,
        embedding_dim=backbone.embedding_dim,
        random_weights=config.random_weights,
        device=str(device),
    )


def summary_to_json(summary: ExtractionSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)


def main() -> None:
    summary = extract_embeddings(parse_args())
    print(summary_to_json(summary))


if __name__ == "__main__":
    main()
