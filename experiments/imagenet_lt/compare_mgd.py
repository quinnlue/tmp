from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from clustering import ClusterBasis, FixedClusterBasis, load_cluster_basis_artifacts
from determinism import configure_replay_determinism
from functional_train import (
    InnerOptimizerConfig,
    SmoothAdamWConfig,
    SmoothSGDConfig,
    TrainState,
    initialize_train_state,
    weighted_inner_step,
)
from imagenet_lt import (
    ImageNetLTDataset,
    ImageNetLTEntry,
    batch_index_schedule,
    class_counts,
    load_manifest_entries,
    split_train_objective_meta,
)
from imagenet_lt_parquet import ImageNetLTParquetDataset, discover_hf_parquet_splits
from model import ViTConfig, VisionTransformerClassifier, per_example_cross_entropy_loss
from recursive_replay import recursive_replay_state
from weighting import group_masses
import metasmooth as ms


Granularity = Literal["per_class", "per_cluster", "per_example"]
RunMode = Literal["run", "preflight", "dry_run"]
InnerBackward = Literal["replay", "store_all"]
ModelProfile = Literal["vit_s", "smoke"]
ModelBackend = Literal["vit", "resnet9"]
InnerOptimizer = Literal["adamw", "sgd"]
ResNetNormalization = Literal["gn", "frozen_bn"]
PoolMode = Literal["mean", "cls"]
DataSourceMode = Literal["manifest", "parquet"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LazyImageDataset(Protocol):
    labels: Tensor
    num_classes: int

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Tensor, int]: ...

    def subset_labels(self, indices: Sequence[int] | Tensor) -> Tensor: ...


@dataclass(frozen=True)
class Config:
    train_manifest: str | None = None
    output: str = "artifacts/compare_imagenet_lt_mgd.json"
    eval_manifest: str | None = None
    val_manifest: str | None = None
    test_manifest: str | None = None
    dataset_root: str | None = None
    parquet_root: str | None = None
    train_split: str = "train"
    val_split: str | None = "val"
    test_split: str | None = "test"
    cluster_artifact_prefix: str | None = None
    exposed_train_indices_input: str | None = None
    exposed_train_indices_output: str | None = None
    features_artifact: str | None = None
    val_features_artifact: str | None = None
    test_features_artifact: str | None = None
    seed: int = 0
    objective_per_class: int = 1
    meta_validation_per_class: int = 1
    min_train_per_class: int = 1
    max_train_examples: int = 0
    inner_epochs: int = 1
    batch_size: int = 8
    max_inner_steps: int = 0
    eval_batch_size: int = 16
    inner_lr: float = 1e-3
    inner_weight_decay: float = 0.0
    inner_beta1: float = 0.9
    inner_beta2: float = 0.999
    inner_eps: float = 1e-8
    inner_optimizer: InnerOptimizer = "adamw"
    inner_momentum: float = 0.9
    inner_warmup_fraction: float = 0.1
    meta_steps: int = 0
    outer_step_kl: float = 1e-3
    temperature: float = 1.0
    branching_factor: int = 4
    inner_backward: InnerBackward = "replay"
    device: str = "auto"
    mode: RunMode = "run"
    granularities: tuple[Granularity, ...] = ("per_class", "per_cluster", "per_example")
    model_backend: ModelBackend = "vit"
    resnet_normalization: ResNetNormalization = "gn"
    initial_checkpoint: str | None = None
    model_profile: ModelProfile = "vit_s"
    image_size: int = 224
    patch_size: int = 16
    encoder_dim: int | None = None
    encoder_depth: int | None = None
    encoder_heads: int | None = None
    mlp_ratio: float | None = None
    pre_norm: bool | None = None
    pool: PoolMode | None = None
    final_logit_scale: float = ViTConfig().final_logit_scale
    materialize_train_images: bool = False
    json_indent: int = 2


@dataclass(frozen=True)
class DatasetBundle:
    source_mode: DataSourceMode
    train_dataset: LazyImageDataset
    val_dataset: LazyImageDataset | None
    test_dataset: LazyImageDataset | None
    train_alignment_labels: Tensor
    available_splits: tuple[str, ...]


@dataclass(frozen=True)
class DatasetPlan:
    source_mode: DataSourceMode
    train_labels: Tensor
    exposed_train_labels: Tensor
    split_train_indices: Tensor
    objective_indices: Tensor
    meta_validation_indices: Tensor
    inner_schedule: tuple[Tensor, ...]
    num_classes: int
    val_labels: Tensor | None
    test_labels: Tensor | None
    available_splits: tuple[str, ...]


@dataclass(frozen=True)
class FeatureArtifact:
    path: str
    features: Tensor
    labels: Tensor
    position_by_global_index: Tensor
    embedding_dim: int
    split: str | None
    backbone: str | None

    def positions_for(self, indices: Tensor) -> Tensor:
        device_indices = indices.to(device=self.position_by_global_index.device, dtype=torch.long)
        if device_indices.numel() == 0:
            return device_indices
        if torch.any(device_indices < 0) or torch.any(device_indices >= self.position_by_global_index.numel()):
            raise ValueError(f"requested indices are out of range for feature artifact {self.path}")
        positions = self.position_by_global_index.index_select(0, device_indices)
        if torch.any(positions < 0):
            missing = device_indices[positions < 0].detach().cpu().tolist()
            raise ValueError(f"feature artifact {self.path} is missing requested indices {missing[:16]}")
        return positions


@dataclass(frozen=True)
class FeatureBundle:
    train: FeatureArtifact
    val: FeatureArtifact | None
    test: FeatureArtifact | None
    embedding_dim: int
    backbone: str | None


@dataclass(frozen=True)
class MaterializedImageArtifact:
    images: Tensor
    labels: Tensor
    position_by_global_index: Tensor
    image_size: int

    def positions_for(self, indices: Tensor) -> Tensor:
        device_indices = indices.to(device=self.position_by_global_index.device, dtype=torch.long)
        if device_indices.numel() == 0:
            return device_indices
        if torch.any(device_indices < 0) or torch.any(device_indices >= self.position_by_global_index.numel()):
            raise ValueError("requested materialized-image indices are out of range")
        positions = self.position_by_global_index.index_select(0, device_indices)
        if torch.any(positions < 0):
            missing = device_indices[positions < 0].detach().cpu().tolist()
            raise ValueError(f"materialized image artifact is missing requested indices {missing[:16]}")
        return positions


@dataclass(frozen=True)
class GranularitySpec:
    name: Granularity
    group_ids: Tensor
    base_group_masses: Tensor
    group_class_ids: Tensor
    train_class_ids: Tensor
    class_tied_fairness: bool = True

    @property
    def num_groups(self) -> int:
        return int(self.base_group_masses.numel())


@dataclass(frozen=True)
class ClusterAlignment:
    subset_group_ids: Tensor
    subset_base_group_masses: Tensor
    subset_group_class_ids: Tensor
    original_group_ids: Tensor
    label_pure: bool


class ScaledLinearHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, *, final_logit_scale: float) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_classes <= 0:
            raise ValueError("embedding_dim and num_classes must be positive")
        if final_logit_scale <= 0:
            raise ValueError("final_logit_scale must be positive")
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.final_logit_scale = float(final_logit_scale)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features) / self.final_logit_scale


def parse_granularities(value: str) -> tuple[Granularity, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    valid = {"per_class", "per_cluster", "per_example"}
    if not items:
        raise argparse.ArgumentTypeError("expected at least one granularity")
    if any(item not in valid for item in items):
        raise argparse.ArgumentTypeError(
            "--granularities must be a comma-separated subset of per_class, per_cluster, per_example"
        )
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("--granularities must not repeat entries")
    return items  # type: ignore[return-value]


def resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def make_model_config(config: Config, *, num_classes: int) -> ViTConfig:
    base_defaults = ViTConfig()
    if config.model_profile == "vit_s":
        profile_defaults: dict[str, float | int] = {
            "encoder_dim": base_defaults.encoder_dim,
            "encoder_depth": base_defaults.encoder_depth,
            "encoder_heads": base_defaults.encoder_heads,
            "mlp_ratio": base_defaults.mlp_ratio,
        }
    elif config.model_profile == "smoke":
        profile_defaults = {
            "encoder_dim": 32,
            "encoder_depth": 1,
            "encoder_heads": 4,
            "mlp_ratio": 2.0,
        }
    else:
        raise ValueError(f"unknown model_profile: {config.model_profile}")
    return ViTConfig(
        image_size=config.image_size,
        patch_size=config.patch_size,
        encoder_dim=int(config.encoder_dim or profile_defaults["encoder_dim"]),
        encoder_depth=int(config.encoder_depth or profile_defaults["encoder_depth"]),
        encoder_heads=int(config.encoder_heads or profile_defaults["encoder_heads"]),
        mlp_ratio=float(config.mlp_ratio or profile_defaults["mlp_ratio"]),
        num_classes=num_classes,
        pre_norm=base_defaults.pre_norm if config.pre_norm is None else config.pre_norm,
        pool=base_defaults.pool if config.pool is None else config.pool,
        final_logit_scale=config.final_logit_scale,
    )


def validate_config(config: Config) -> None:
    if config.objective_per_class < 0 or config.meta_validation_per_class < 0:
        raise ValueError("objective_per_class and meta_validation_per_class must be non-negative")
    if config.min_train_per_class < 1:
        raise ValueError("min_train_per_class must be at least 1")
    if config.max_train_examples < 0 or config.max_inner_steps < 0:
        raise ValueError("max_train_examples and max_inner_steps must be non-negative")
    if config.exposed_train_indices_input is not None and config.max_train_examples > 0:
        raise ValueError("max_train_examples cannot be combined with exposed_train_indices_input")
    if config.inner_epochs <= 0 or config.batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("inner_epochs, batch_size, and eval_batch_size must be positive")
    if config.meta_steps < 0:
        raise ValueError("meta_steps must be non-negative")
    if config.inner_lr <= 0 or config.inner_eps <= 0:
        raise ValueError("inner_lr and inner_eps must be positive")
    if not 0 <= config.inner_momentum < 1:
        raise ValueError("inner_momentum must be in [0, 1)")
    if not 0 <= config.inner_warmup_fraction < 1:
        raise ValueError("inner_warmup_fraction must be in [0, 1)")
    if config.outer_step_kl <= 0:
        raise ValueError("outer_step_kl must be positive")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if config.branching_factor < 2:
        raise ValueError("branching_factor must be at least 2")
    if config.final_logit_scale <= 0:
        raise ValueError("final_logit_scale must be positive")
    if config.model_backend == "resnet9" and config.image_size % 8:
        raise ValueError("ResNet-9 image_size must be divisible by 8")
    if config.mode in ("run", "preflight") and "per_cluster" in config.granularities:
        if not config.cluster_artifact_prefix:
            raise ValueError("cluster_artifact_prefix is required when per_cluster is enabled")
    if config.features_artifact is None and (
        config.val_features_artifact is not None or config.test_features_artifact is not None
    ):
        raise ValueError("val/test feature artifacts require --features-artifact")
    if config.features_artifact is not None and config.model_backend != "vit":
        raise ValueError("feature artifacts use the linear-head backend and are incompatible with resnet9")
    if config.features_artifact is not None and config.materialize_train_images:
        raise ValueError("materialize_train_images is only valid for the image-training backend")
    if config.initial_checkpoint is not None and config.model_backend != "resnet9":
        raise ValueError("initial_checkpoint is currently supported only for the resnet9 backend")
    has_manifest_source = config.train_manifest is not None
    has_parquet_source = config.parquet_root is not None
    if has_manifest_source == has_parquet_source:
        raise ValueError("provide exactly one train source: train_manifest or parquet_root")
    if config.parquet_root is not None:
        if config.dataset_root is not None:
            raise ValueError("dataset_root is only valid with manifest input")
        if config.eval_manifest is not None or config.val_manifest is not None or config.test_manifest is not None:
            raise ValueError("manifest evaluation inputs are incompatible with parquet_root")
    _ = make_model_config(config, num_classes=2)


def load_manifest_dataset(
    manifest_path: str,
    *,
    root: str | None,
    validate_paths: bool,
) -> tuple[ImageNetLTDataset, tuple[ImageNetLTEntry, ...]]:
    entries = tuple(load_manifest_entries(manifest_path, root=root, validate_paths=validate_paths))
    dataset = ImageNetLTDataset(
        manifest_path,
        root=root,
        validate_paths=validate_paths,
        entries=entries,
    )
    return dataset, entries


def build_dataset_bundle(config: Config) -> DatasetBundle:
    validate_paths = config.mode != "dry_run"
    if config.parquet_root is not None:
        split_map = discover_hf_parquet_splits(config.parquet_root)
        train_dataset = ImageNetLTParquetDataset(config.parquet_root, split=config.train_split)
        val_dataset = (
            ImageNetLTParquetDataset(config.parquet_root, split=config.val_split)
            if config.val_split is not None and config.val_split in split_map
            else None
        )
        test_dataset = (
            ImageNetLTParquetDataset(config.parquet_root, split=config.test_split)
            if config.test_split is not None and config.test_split in split_map
            else None
        )
        return DatasetBundle(
            source_mode="parquet",
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_alignment_labels=train_dataset.labels.clone(),
            available_splits=tuple(sorted(split_map)),
        )

    if config.train_manifest is None:
        raise ValueError("train_manifest is required when parquet_root is omitted")
    train_dataset, train_entries = load_manifest_dataset(
        config.train_manifest,
        root=config.dataset_root,
        validate_paths=validate_paths,
    )
    resolved_val_manifest = config.val_manifest or config.eval_manifest
    val_dataset = None
    test_dataset = None
    if resolved_val_manifest is not None:
        val_dataset, _ = load_manifest_dataset(
            resolved_val_manifest,
            root=config.dataset_root,
            validate_paths=validate_paths,
        )
    if config.test_manifest is not None:
        test_dataset, _ = load_manifest_dataset(
            config.test_manifest,
            root=config.dataset_root,
            validate_paths=validate_paths,
        )
    return DatasetBundle(
        source_mode="manifest",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_alignment_labels=torch.tensor([entry.label for entry in train_entries], dtype=torch.long),
        available_splits=("train",) + (() if val_dataset is None else ("val",)) + (() if test_dataset is None else ("test",)),
    )


def load_dataset_plan(config: Config, bundle: DatasetBundle) -> DatasetPlan:
    train_labels = bundle.train_dataset.labels.clone()
    split = split_train_objective_meta(
        train_labels,
        objective_per_class=config.objective_per_class,
        meta_validation_per_class=config.meta_validation_per_class,
        seed=config.seed,
        min_train_per_class=config.min_train_per_class,
    )
    if config.exposed_train_indices_input is None:
        exposed_train_indices = cap_exposed_pool_indices(
            split.train_indices,
            train_labels,
            max_train_examples=config.max_train_examples,
        )
    else:
        exposed_train_indices = torch.load(config.exposed_train_indices_input, map_location="cpu")
        if (
            not isinstance(exposed_train_indices, Tensor)
            or exposed_train_indices.ndim != 1
            or exposed_train_indices.dtype != torch.long
        ):
            raise ValueError("exposed_train_indices_input must contain a one-dimensional torch.long tensor")
        if exposed_train_indices.numel() == 0:
            raise ValueError("exposed_train_indices_input must be non-empty")
        if torch.unique(exposed_train_indices).numel() != exposed_train_indices.numel():
            raise ValueError("exposed_train_indices_input must not contain duplicate indices")
        allowed = torch.zeros(train_labels.numel(), dtype=torch.bool)
        allowed[split.train_indices] = True
        if torch.any(exposed_train_indices < 0) or torch.any(exposed_train_indices >= train_labels.numel()):
            raise ValueError("exposed_train_indices_input contains an out-of-range index")
        if not bool(allowed[exposed_train_indices].all()):
            raise ValueError("exposed_train_indices_input must be a subset of the exposed train split")
    inner_schedule = batch_index_schedule(
        torch.arange(exposed_train_indices.numel(), dtype=torch.long),
        batch_size=config.batch_size,
        epochs=config.inner_epochs,
        seed=config.seed + 1,
        drop_last=False,
    )
    if config.max_inner_steps > 0:
        inner_schedule = inner_schedule[: config.max_inner_steps]
    num_classes = max(class_counts(train_labels)) + 1
    exposed_train_labels = train_labels.index_select(0, exposed_train_indices)
    return DatasetPlan(
        source_mode=bundle.source_mode,
        train_labels=train_labels,
        exposed_train_labels=exposed_train_labels,
        split_train_indices=exposed_train_indices,
        objective_indices=split.objective_indices,
        meta_validation_indices=split.meta_validation_indices,
        inner_schedule=inner_schedule,
        num_classes=num_classes,
        val_labels=None if bundle.val_dataset is None else bundle.val_dataset.labels.clone(),
        test_labels=None if bundle.test_dataset is None else bundle.test_dataset.labels.clone(),
        available_splits=bundle.available_splits,
    )


def load_feature_artifact(
    path: str | Path,
    *,
    expected_labels: Tensor,
    device: torch.device,
    expected_split: str | None,
) -> FeatureArtifact:
    resolved_path = Path(path).resolve()
    payload = torch.load(resolved_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"feature artifact {resolved_path} must be a dictionary")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"feature artifact {resolved_path} metadata must be a dictionary")
    features = payload.get("embeddings")
    labels = payload.get("labels")
    global_indices = payload.get("global_indices")
    if not isinstance(features, Tensor) or features.ndim != 2 or not torch.is_floating_point(features):
        raise ValueError(f"feature artifact {resolved_path} embeddings must be a floating-point matrix")
    if not isinstance(labels, Tensor) or labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError(f"feature artifact {resolved_path} labels must be one-dimensional torch.long")
    if not isinstance(global_indices, Tensor) or global_indices.ndim != 1 or global_indices.dtype != torch.long:
        raise ValueError(
            f"feature artifact {resolved_path} global_indices must be one-dimensional torch.long"
        )
    target_examples = int(metadata.get("target_examples", -1))
    completed = int(payload.get("completed", -1))
    embedding_dim = int(payload.get("embedding_dim", -1))
    if completed != target_examples or target_examples != features.shape[0]:
        raise ValueError(f"feature artifact {resolved_path} must be complete")
    if embedding_dim <= 0 or features.shape[1] != embedding_dim:
        raise ValueError(f"feature artifact {resolved_path} embedding_dim is invalid")
    if labels.shape != global_indices.shape or labels.numel() != features.shape[0]:
        raise ValueError(f"feature artifact {resolved_path} tensors disagree on example count")
    if global_indices.numel() == 0 or torch.unique(global_indices).numel() != global_indices.numel():
        raise ValueError(f"feature artifact {resolved_path} global_indices must be non-empty and unique")
    if torch.any(global_indices < 0) or torch.any(global_indices >= expected_labels.numel()):
        raise ValueError(f"feature artifact {resolved_path} global_indices are out of range")
    expected_artifact_labels = expected_labels.index_select(0, global_indices)
    if not torch.equal(labels, expected_artifact_labels):
        raise ValueError(f"feature artifact {resolved_path} labels do not align with the dataset")
    split = metadata.get("split")
    if expected_split is not None and split is not None and split != expected_split:
        raise ValueError(
            f"feature artifact {resolved_path} is for split {split!r}, expected {expected_split!r}"
        )

    position_by_global_index = torch.full(
        (expected_labels.numel(),),
        -1,
        dtype=torch.long,
        device=device,
    )
    device_global_indices = global_indices.to(device=device)
    position_by_global_index[device_global_indices] = torch.arange(
        global_indices.numel(),
        dtype=torch.long,
        device=device,
    )
    return FeatureArtifact(
        path=str(resolved_path),
        features=features.to(device=device, dtype=torch.float32),
        labels=labels.to(device=device),
        position_by_global_index=position_by_global_index,
        embedding_dim=embedding_dim,
        split=None if split is None else str(split),
        backbone=None if metadata.get("backbone") is None else str(metadata["backbone"]),
    )


def load_feature_bundle(
    config: Config,
    bundle: DatasetBundle,
    plan: DatasetPlan,
    *,
    device: torch.device,
) -> FeatureBundle | None:
    if config.features_artifact is None:
        return None
    train = load_feature_artifact(
        config.features_artifact,
        expected_labels=plan.train_labels,
        device=device,
        expected_split=config.train_split,
    )
    required_train_indices = torch.unique(
        torch.cat(
            (
                plan.split_train_indices,
                plan.objective_indices,
                plan.meta_validation_indices,
            )
        )
    )
    train.positions_for(required_train_indices)

    val = None
    if bundle.val_dataset is not None and plan.val_labels is not None:
        if config.val_features_artifact is None:
            raise ValueError("val_features_artifact is required when feature mode evaluates val")
        val = load_feature_artifact(
            config.val_features_artifact,
            expected_labels=plan.val_labels,
            device=device,
            expected_split=config.val_split,
        )
        val.positions_for(torch.arange(plan.val_labels.numel(), dtype=torch.long))

    test = None
    if bundle.test_dataset is not None and plan.test_labels is not None:
        if config.test_features_artifact is None:
            raise ValueError("test_features_artifact is required when feature mode evaluates test")
        test = load_feature_artifact(
            config.test_features_artifact,
            expected_labels=plan.test_labels,
            device=device,
            expected_split=config.test_split,
        )
        test.positions_for(torch.arange(plan.test_labels.numel(), dtype=torch.long))

    artifacts = [artifact for artifact in (train, val, test) if artifact is not None]
    if any(artifact.embedding_dim != train.embedding_dim for artifact in artifacts):
        raise ValueError("all feature artifacts must share the same embedding dimension")
    if any(artifact.backbone != train.backbone for artifact in artifacts):
        raise ValueError("all feature artifacts must share the same backbone")
    return FeatureBundle(
        train=train,
        val=val,
        test=test,
        embedding_dim=train.embedding_dim,
        backbone=train.backbone,
    )


def capped_class_counts(counts: Tensor, max_examples: int) -> Tensor:
    if counts.ndim != 1 or counts.dtype != torch.long:
        raise ValueError("counts must be a one-dimensional torch.long tensor")
    total = int(counts.sum().item())
    if max_examples <= 0 or max_examples >= total:
        return counts.clone()
    if max_examples < int((counts > 0).sum().item()):
        raise ValueError(
            "max_train_examples is too small to retain one example for every represented class"
        )
    represented = counts > 0
    # Reserve one example per represented class first. Scaling and flooring the
    # raw counts before enforcing this minimum can overshoot the budget, which
    # then silently drops later classes when indices are selected in order.
    capped = represented.to(torch.long)
    remaining = max_examples - int(capped.sum().item())
    capacities = counts - capped
    capacity_total = int(capacities.sum().item())
    expected_extra = (
        capacities.to(torch.float64) * (float(remaining) / float(capacity_total))
        if capacity_total > 0
        else capacities.to(torch.float64)
    )
    extra = torch.minimum(torch.floor(expected_extra).to(torch.long), capacities)
    capped = capped + extra
    remaining = max_examples - int(capped.sum().item())
    fractional = expected_extra - torch.floor(expected_extra)
    priorities = sorted(
        range(counts.numel()),
        key=lambda class_id: (-float(fractional[class_id].item()), class_id),
    )
    while remaining > 0:
        progress = False
        for class_id in priorities:
            if capped[class_id] < counts[class_id]:
                capped[class_id] += 1
                remaining -= 1
                progress = True
                if remaining == 0:
                    break
        if not progress:
            raise RuntimeError("failed to allocate capped class counts")
    return capped


def cap_exposed_pool_indices(
    train_indices: Tensor,
    train_labels: Tensor,
    *,
    max_train_examples: int,
) -> Tensor:
    if max_train_examples <= 0 or max_train_examples >= train_indices.numel():
        return train_indices.clone()
    split_labels = train_labels.index_select(0, train_indices)
    counts = torch.bincount(split_labels, minlength=max(class_counts(train_labels)) + 1)
    target_counts = capped_class_counts(counts.to(torch.long), max_train_examples)
    kept_per_class = torch.zeros_like(target_counts)
    selected: list[int] = []
    for original_index in train_indices.tolist():
        class_id = int(train_labels[int(original_index)].item())
        if kept_per_class[class_id] < target_counts[class_id]:
            selected.append(int(original_index))
            kept_per_class[class_id] += 1
            if len(selected) == max_train_examples:
                break
    result = torch.tensor(selected, dtype=torch.long)
    if result.numel() != max_train_examples:
        raise RuntimeError("failed to cap train pool to the requested size")
    return result


def class_base_masses_from_labels(labels: Tensor, *, num_classes: int) -> Tensor:
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("every class in the experiment must appear in the training split")
    return counts / counts.sum()


def build_granularity_spec(
    granularity: Granularity,
    train_labels: Tensor,
    *,
    num_classes: int,
    aligned_clusters: ClusterAlignment | None = None,
) -> GranularitySpec:
    device = train_labels.device
    if granularity == "per_class":
        group_ids = train_labels.clone()
        base_group_masses = class_base_masses_from_labels(train_labels, num_classes=num_classes)
        group_class_ids = torch.arange(num_classes, dtype=torch.long, device=device)
        class_tied_fairness = True
    elif granularity == "per_cluster":
        if aligned_clusters is None:
            raise ValueError("aligned_clusters is required for per_cluster")
        group_ids = aligned_clusters.subset_group_ids.to(device=device)
        base_group_masses = aligned_clusters.subset_base_group_masses.to(device=device)
        group_class_ids = aligned_clusters.subset_group_class_ids.to(device=device)
        class_tied_fairness = aligned_clusters.label_pure
    elif granularity == "per_example":
        group_ids = torch.arange(train_labels.numel(), dtype=torch.long, device=device)
        base_group_masses = torch.full(
            (train_labels.numel(),),
            1.0 / float(train_labels.numel()),
            dtype=torch.float32,
            device=device,
        )
        group_class_ids = train_labels.clone()
        class_tied_fairness = True
    else:
        raise ValueError(f"unknown granularity: {granularity}")
    return GranularitySpec(
        name=granularity,
        group_ids=group_ids,
        base_group_masses=base_group_masses,
        group_class_ids=group_class_ids,
        train_class_ids=train_labels.clone(),
        class_tied_fairness=class_tied_fairness,
    )


def class_mass_summary(logits: Tensor, spec: GranularitySpec, temperature: float = 1.0) -> dict[str, Any]:
    masses = effective_group_masses(logits, spec, temperature)
    class_masses = torch.zeros(
        int(spec.group_class_ids.max().item()) + 1,
        dtype=masses.dtype,
        device=masses.device,
    )
    class_masses.scatter_add_(0, spec.group_class_ids.to(masses.device), masses)
    base_class_masses = torch.zeros_like(class_masses)
    base_class_masses.scatter_add_(
        0,
        spec.group_class_ids.to(masses.device),
        spec.base_group_masses.to(device=masses.device, dtype=masses.dtype),
    )
    return {
        "group_masses": masses.detach().cpu().tolist(),
        "class_masses": class_masses.detach().cpu().tolist(),
        "base_class_masses": base_class_masses.detach().cpu().tolist(),
        "entropy": float(-(masses * masses.log()).sum().item()),
    }


def class_logits_to_fine_logits(class_logits: Tensor, spec: GranularitySpec) -> Tensor:
    return class_logits[spec.group_class_ids]


def aggregate_class_masses(base_group_masses: Tensor, group_class_ids: Tensor, num_classes: int) -> Tensor:
    result = torch.zeros(num_classes, dtype=base_group_masses.dtype, device=base_group_masses.device)
    result.scatter_add_(0, group_class_ids, base_group_masses)
    return result


def effective_group_logits(logits: Tensor, spec: GranularitySpec, temperature: float) -> Tensor:
    return logits + temperature * torch.log(spec.base_group_masses.to(logits.device))


def effective_group_masses(logits: Tensor, spec: GranularitySpec, temperature: float) -> Tensor:
    return group_masses(effective_group_logits(logits, spec, temperature), temperature)


def labels_from_source(source: Sequence[ImageNetLTEntry] | Tensor) -> Tensor:
    if isinstance(source, Tensor):
        if source.ndim != 1 or source.dtype != torch.long:
            raise ValueError("labels tensor must be one-dimensional torch.long")
        return source.detach().cpu()
    return torch.tensor([entry.label for entry in source], dtype=torch.long)


def load_cluster_basis_source_indices(
    artifact_prefix: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tensor | None:
    indices_path = Path(artifact_prefix).with_suffix(".indices.pt")
    if not indices_path.exists():
        return None
    indices = torch.load(indices_path, map_location=map_location)
    if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError(f"{indices_path} must contain a one-dimensional torch.long tensor")
    return indices.detach().cpu()


def align_cluster_basis_to_split(
    basis: FixedClusterBasis | ClusterBasis,
    entries_or_labels: Sequence[ImageNetLTEntry] | Tensor,
    train_indices: Tensor,
    basis_source_indices: Tensor | None = None,
) -> ClusterAlignment:
    labels = labels_from_source(entries_or_labels)
    source_indices = (
        torch.arange(labels.numel(), dtype=torch.long)
        if basis_source_indices is None
        else basis_source_indices.detach().cpu().to(dtype=torch.long)
    )
    if source_indices.ndim != 1:
        raise ValueError("basis_source_indices must be one-dimensional")
    if basis.group_ids.numel() != source_indices.numel():
        raise ValueError(
            f"cluster basis has {basis.group_ids.numel()} assignments for {source_indices.numel()} source indices"
        )
    if source_indices.numel() == 0:
        raise ValueError("cluster basis source indices must be non-empty")
    if int(source_indices.min().item()) < 0 or int(source_indices.max().item()) >= labels.numel():
        raise ValueError("cluster basis source indices are out of range for the source labels")
    train_indices = train_indices.detach().cpu().to(dtype=torch.long)
    basis_group_ids = basis.group_ids.detach().cpu()
    if basis_source_indices is None:
        subset_original_ids = basis_group_ids.index_select(0, train_indices)
    else:
        if torch.unique(source_indices).numel() != source_indices.numel():
            raise ValueError(
                "cluster basis source indices must contain each exposed train index exactly once; "
                "duplicate source indices found"
            )
        positions_by_source_index = {
            int(source_index): position
            for position, source_index in enumerate(source_indices.tolist())
        }
        missing = [index for index in train_indices.tolist() if index not in positions_by_source_index]
        if missing:
            raise ValueError(
                "cluster basis source indices must contain each exposed train index exactly once; "
                f"missing {missing}"
            )
        subset_positions = torch.tensor(
            [positions_by_source_index[int(index)] for index in train_indices.tolist()],
            dtype=torch.long,
        )
        subset_original_ids = basis_group_ids.index_select(0, subset_positions)
    label_pure = isinstance(basis, FixedClusterBasis)
    group_class_ids = torch.full((basis.num_groups,), -1, dtype=torch.long)
    realized_class_to_groups: dict[int, list[int]] = {}
    for group_id in range(basis.num_groups):
        member_labels = labels.index_select(0, source_indices[basis_group_ids == group_id])
        if member_labels.numel() == 0:
            continue
        unique_labels = torch.unique(member_labels)
        if label_pure and unique_labels.numel() != 1:
            raise ValueError(f"cluster {group_id} mixes labels {unique_labels.tolist()}")
        class_id = int(torch.bincount(member_labels).argmax().item())
        group_class_ids[group_id] = class_id
        realized_class_to_groups.setdefault(class_id, []).append(group_id)
    if isinstance(basis, FixedClusterBasis):
        for class_id, cluster_ids in basis.class_to_cluster_ids.items():
            realized = tuple(realized_class_to_groups.get(class_id, ()))
            if tuple(cluster_ids) != realized:
                raise ValueError(
                    f"cluster artifact metadata mismatch for class {class_id}: expected {tuple(cluster_ids)}, realized {realized}"
                )
    present_original_ids = torch.unique(subset_original_ids, sorted=True)
    remapped = torch.empty_like(subset_original_ids)
    subset_group_class_ids = torch.empty(present_original_ids.numel(), dtype=torch.long)
    for new_group_id, old_group_id_tensor in enumerate(present_original_ids):
        old_group_id = int(old_group_id_tensor.item())
        remapped[subset_original_ids == old_group_id] = new_group_id
        subset_group_class_ids[new_group_id] = group_class_ids[old_group_id]
    subset_counts = torch.bincount(remapped, minlength=present_original_ids.numel()).to(torch.float32)
    subset_base_group_masses = subset_counts / subset_counts.sum()
    return ClusterAlignment(
        subset_group_ids=remapped,
        subset_base_group_masses=subset_base_group_masses,
        subset_group_class_ids=subset_group_class_ids,
        original_group_ids=present_original_ids,
        label_pure=label_pure,
    )


def preprocess_image(image: Tensor, *, image_size: int) -> Tensor:
    if image.ndim != 3:
        raise ValueError("expected image tensor with shape [channels, height, width]")
    image = image.to(dtype=torch.float32).div_(255.0).unsqueeze(0)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((image - mean) / std).squeeze(0)


def load_batch(
    dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact,
    indices: Tensor,
    *,
    image_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if isinstance(dataset, FeatureArtifact):
        positions = dataset.positions_for(indices)
        return (
            dataset.features.index_select(0, positions),
            dataset.labels.index_select(0, positions),
        )
    if isinstance(dataset, MaterializedImageArtifact):
        positions = dataset.positions_for(indices)
        return (
            dataset.images.index_select(0, positions),
            dataset.labels.index_select(0, positions),
        )
    # Parquet-backed datasets cache one row group. Reading random indices in
    # schedule order repeatedly evicts that cache, so read in storage order and
    # restore the requested order before returning. The model trajectory is
    # unchanged, while held-out evaluation becomes dramatically less I/O-bound.
    order = torch.argsort(indices)
    sorted_indices = indices.index_select(0, order)
    sorted_images: list[Tensor] = []
    sorted_labels: list[int] = []
    for index in sorted_indices.tolist():
        image, label = dataset[int(index)]
        sorted_images.append(preprocess_image(image, image_size=image_size))
        sorted_labels.append(int(label))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), dtype=order.dtype)
    images = torch.stack(sorted_images, dim=0).index_select(0, inverse)
    labels = torch.tensor(sorted_labels, dtype=torch.long).index_select(0, inverse)
    return images.to(device), labels.to(device)


def materialize_image_artifact(
    dataset: LazyImageDataset,
    indices: Tensor,
    *,
    image_size: int,
    device: torch.device,
) -> MaterializedImageArtifact:
    ordered_indices = torch.unique(indices.detach().cpu().to(torch.long), sorted=True)
    if ordered_indices.numel() == 0:
        raise ValueError("cannot materialize an empty image selection")
    images = torch.empty(
        (ordered_indices.numel(), 3, image_size, image_size),
        dtype=torch.float32,
        device=device,
    )
    labels = torch.empty(ordered_indices.numel(), dtype=torch.long, device=device)
    for position, global_index in enumerate(ordered_indices.tolist()):
        image, label = dataset[int(global_index)]
        images[position].copy_(preprocess_image(image, image_size=image_size).to(device))
        labels[position] = int(label)
    position_by_global_index = torch.full(
        (len(dataset),),
        -1,
        dtype=torch.long,
        device=device,
    )
    position_by_global_index[ordered_indices.to(device)] = torch.arange(
        ordered_indices.numel(),
        dtype=torch.long,
        device=device,
    )
    return MaterializedImageArtifact(
        images=images,
        labels=labels,
        position_by_global_index=position_by_global_index,
        image_size=image_size,
    )


def save_result(path: str | Path, payload: dict[str, Any], *, indent: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")
    temporary.replace(destination)


def state_checksum(state: TrainState) -> float:
    return float(sum(value.detach().double().sum().item() for value in state.parameters.values()))


def fixed_kl_signed_update(
    logits: Tensor,
    gradient: Tensor,
    spec: GranularitySpec,
    *,
    temperature: float,
    target_kl: float,
) -> dict[str, float]:
    old_masses = effective_group_masses(logits.detach(), spec, temperature)
    direction = -gradient.sign()
    if not torch.any(direction):
        return {"distribution_kl": 0.0, "logit_scale": 0.0}

    def kl_at(scale: float) -> float:
        proposal = effective_group_masses(logits.detach() + direction * scale, spec, temperature)
        return float((proposal * (proposal.log() - old_masses.log())).sum().item())

    lower = 0.0
    upper = 1.0
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
        logits.add_(direction, alpha=scale)
        logits.sub_(logits.mean())
        new_masses = effective_group_masses(logits, spec, temperature)
        actual_kl = float((new_masses * (new_masses.log() - old_masses.log())).sum().item())
    return {"distribution_kl": actual_kl, "logit_scale": float(scale)}


def inner_learning_rate_at_step(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float,
    peak_learning_rate: float,
) -> float:
    warmup_steps = round(warmup_fraction * total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return peak_learning_rate * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_state_for_logits(
    model: nn.Module,
    initial_state: TrainState,
    train_dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact,
    train_dataset_indices: Tensor,
    schedule: tuple[Tensor, ...],
    spec: GranularitySpec,
    optimizer_config: InnerOptimizerConfig,
    image_size: int,
    logits: Tensor,
    config: Config,
    device: torch.device,
) -> TrainState:
    def replay_step(state: TrainState, step_index: int, replay_logits: Tensor, create_graph: bool) -> TrainState:
        batch_positions = schedule[step_index]
        batch_indices = train_dataset_indices.index_select(0, batch_positions)
        images, labels = load_batch(
            train_dataset,
            batch_indices,
            image_size=image_size,
            device=device,
        )
        local_group_ids = spec.group_ids.index_select(0, batch_positions.to(spec.group_ids.device)).to(device)
        effective_logits = effective_group_logits(replay_logits, spec, config.temperature)
        step_optimizer_config = optimizer_config
        if isinstance(optimizer_config, SmoothSGDConfig):
            step_optimizer_config = replace(
                optimizer_config,
                learning_rate=inner_learning_rate_at_step(
                    step_index,
                    total_steps=len(schedule),
                    warmup_fraction=config.inner_warmup_fraction,
                    peak_learning_rate=config.inner_lr,
                ),
            )
        next_state, _ = weighted_inner_step(
            model,
            state,
            images,
            labels,
            local_group_ids,
            effective_logits,
            spec.base_group_masses.to(device),
            optimizer_config,
            temperature=config.temperature,
            create_graph=create_graph,
        )
        return next_state

    if config.inner_backward == "replay":
        return recursive_replay_state(
            initial_state,
            logits,
            len(schedule),
            replay_step,
            branching_factor=config.branching_factor,
        )

    state = initial_state
    for step_index in range(len(schedule)):
        state = replay_step(state, step_index, logits, True)
    return state


def differentiable_mean_cross_entropy(
    model: nn.Module,
    state: TrainState,
    dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact,
    indices: Tensor,
    *,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> Tensor:
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    total_count = 0
    # Objective order is immaterial to the mean; storage order lets the
    # Parquet adapter reuse row-group cache across consecutive batches.
    ordered_indices = torch.sort(indices).values
    for chunk in chunk_indices(ordered_indices, batch_size):
        images, labels = load_batch(dataset, chunk, image_size=image_size, device=device)
        logits = functional_call(model, (state.parameters, state.buffers), (images,))
        total_loss = total_loss + per_example_cross_entropy_loss(logits, labels).sum()
        total_count += labels.numel()
    return total_loss / float(total_count)


@torch.no_grad()
def evaluate_indices(
    model: nn.Module,
    state: TrainState,
    dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact,
    indices: Tensor,
    labels_cpu: Tensor,
    *,
    batch_size: int,
    image_size: int,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    loss_sums = torch.zeros(num_classes, dtype=torch.float64, device=device)
    correct_sums = torch.zeros(num_classes, dtype=torch.float64, device=device)
    count_sums = torch.zeros(num_classes, dtype=torch.float64, device=device)
    ordered_indices = torch.sort(indices).values
    for chunk in chunk_indices(ordered_indices, batch_size):
        images, labels = load_batch(dataset, chunk, image_size=image_size, device=device)
        predictions = functional_call(model, (state.parameters, state.buffers), (images,))
        losses = per_example_cross_entropy_loss(predictions, labels).to(torch.float64)
        correct = (predictions.argmax(dim=1) == labels).to(torch.float64)
        ones = torch.ones_like(correct)
        loss_sums.scatter_add_(0, labels, losses)
        correct_sums.scatter_add_(0, labels, correct)
        count_sums.scatter_add_(0, labels, ones)
    nonzero = count_sums > 0
    per_class_ce = torch.zeros_like(loss_sums)
    per_class_acc = torch.zeros_like(correct_sums)
    per_class_ce[nonzero] = loss_sums[nonzero] / count_sums[nonzero]
    per_class_acc[nonzero] = correct_sums[nonzero] / count_sums[nonzero]
    weights = torch.bincount(labels_cpu.index_select(0, indices), minlength=num_classes).to(torch.float64)
    weights = weights / weights.sum()
    return {
        "mean_ce": float(loss_sums.sum().item() / count_sums.sum().item()),
        "balanced_ce": float(per_class_ce[nonzero].mean().item()),
        "balanced_acc": float(per_class_acc[nonzero].mean().item()),
        "empirical_ce": float((weights.to(device) * per_class_ce).sum().item()),
        "per_class_ce": per_class_ce.cpu().tolist(),
        "per_class_acc": per_class_acc.cpu().tolist(),
        "counts": count_sums.to(torch.long).cpu().tolist(),
    }


def chunk_indices(indices: Tensor, batch_size: int) -> Iterable[Tensor]:
    for start in range(0, indices.numel(), batch_size):
        yield indices[start : start + batch_size]


def add_loss_deltas(history: list[dict[str, Any]]) -> None:
    if not history:
        raise ValueError("history must be non-empty")
    baseline = history[0]
    for record in history:
        deltas = {
            "objective_ce": baseline["objective_ce"] - record["objective_ce"],
            "objective_balanced_ce": (
                baseline["objective_metrics"]["balanced_ce"]
                - record["objective_metrics"]["balanced_ce"]
            ),
            "meta_validation_ce": (
                baseline["meta_validation_metrics"]["mean_ce"]
                - record["meta_validation_metrics"]["mean_ce"]
            ),
            "meta_validation_balanced_ce": (
                baseline["meta_validation_metrics"]["balanced_ce"]
                - record["meta_validation_metrics"]["balanced_ce"]
            ),
        }
        for split_name in ("val", "test"):
            metrics_key = f"{split_name}_metrics"
            if metrics_key in baseline and metrics_key in record:
                deltas[f"{split_name}_ce"] = (
                    baseline[metrics_key]["mean_ce"] - record[metrics_key]["mean_ce"]
                )
                deltas[f"{split_name}_balanced_ce"] = (
                    baseline[metrics_key]["balanced_ce"]
                    - record[metrics_key]["balanced_ce"]
                )
        record["loss_deltas"] = deltas


def class_prior_oracle_logits(spec: GranularitySpec, *, num_classes: int) -> Tensor:
    if spec.name != "per_class" or spec.num_groups != num_classes:
        raise ValueError("class-prior oracle requires the per-class granularity")
    target = torch.full_like(spec.base_group_masses, 1.0 / float(num_classes))
    logits = torch.log(target) - torch.log(spec.base_group_masses)
    return logits - logits.mean()


def build_preflight_payload(
    config: Config,
    plan: DatasetPlan,
    specs: dict[Granularity, GranularitySpec],
    aligned_clusters: ClusterAlignment | None,
    feature_bundle: FeatureBundle | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": asdict(config),
        "design": {
            "source_mode": plan.source_mode,
            "available_splits": list(plan.available_splits),
            "num_source_train_entries": int(plan.train_labels.numel()),
            "num_exposed_train_entries": int(plan.exposed_train_labels.numel()),
            "num_val_entries": 0 if plan.val_labels is None else int(plan.val_labels.numel()),
            "num_test_entries": 0 if plan.test_labels is None else int(plan.test_labels.numel()),
            "num_classes": plan.num_classes,
            "train_split_counts": torch.bincount(
                plan.train_labels.index_select(0, plan.split_train_indices),
                minlength=plan.num_classes,
            ).tolist(),
            "objective_counts": torch.bincount(
                plan.train_labels.index_select(0, plan.objective_indices),
                minlength=plan.num_classes,
            ).tolist(),
            "meta_validation_counts": torch.bincount(
                plan.train_labels.index_select(0, plan.meta_validation_indices),
                minlength=plan.num_classes,
            ).tolist(),
            "val_counts": None
            if plan.val_labels is None
            else torch.bincount(plan.val_labels, minlength=plan.num_classes).tolist(),
            "test_counts": None
            if plan.test_labels is None
            else torch.bincount(plan.test_labels, minlength=plan.num_classes).tolist(),
            "inner_steps": len(plan.inner_schedule),
            "max_train_examples": config.max_train_examples,
            "max_inner_steps": config.max_inner_steps,
            "exposed_train_indices_artifact": config.exposed_train_indices_output,
            "exposed_train_indices_input": config.exposed_train_indices_input,
            "training_backend": "features" if feature_bundle is not None else "images",
            "model_backend": "linear_head" if feature_bundle is not None else config.model_backend,
            "resnet_normalization": config.resnet_normalization,
            "initial_checkpoint": config.initial_checkpoint,
            "inner_optimizer": config.inner_optimizer,
            "materialize_train_images": config.materialize_train_images,
            "feature_embedding_dim": (
                None if feature_bundle is None else feature_bundle.embedding_dim
            ),
            "feature_backbone": None if feature_bundle is None else feature_bundle.backbone,
        },
        "granularities": {},
        "fairness_checks": {},
    }
    for name, spec in specs.items():
        zero_logits = torch.zeros(spec.num_groups, dtype=torch.float32)
        payload["granularities"][name] = {
            "num_groups": spec.num_groups,
            "base_group_masses": spec.base_group_masses.tolist(),
            "zero_logit_summary": class_mass_summary(zero_logits, spec),
        }
    if aligned_clusters is not None:
        payload["granularities"]["per_cluster"]["original_group_ids"] = aligned_clusters.original_group_ids.tolist()
    payload["fairness_checks"] = fairness_checks(specs)
    return payload


def fairness_checks(specs: dict[Granularity, GranularitySpec]) -> dict[str, Any]:
    if "per_class" not in specs:
        return {"checked": False}
    class_spec = specs["per_class"]
    class_logits = torch.linspace(-1.0, 1.0, steps=class_spec.num_groups, dtype=torch.float64)
    baseline = torch.tensor(
        class_mass_summary(class_logits, class_spec)["class_masses"],
        dtype=torch.float64,
    )
    result: dict[str, Any] = {"checked": True, "comparisons": {}}
    for name, spec in specs.items():
        if name == "per_class":
            continue
        if not spec.class_tied_fairness:
            result["comparisons"][name] = {
                "checked": False,
                "reason": "label-agnostic groups do not define an exact class-tied parameterization",
            }
            continue
        fine_logits = class_logits_to_fine_logits(class_logits, spec)
        summary = torch.tensor(
            class_mass_summary(fine_logits, spec)["class_masses"],
            dtype=torch.float64,
        )
        diff = (summary - baseline).abs()
        result["comparisons"][name] = {
            "max_abs_diff": float(diff.max().item()),
            "relative_l2": float(diff.norm().item() / baseline.norm().item()),
        }
    checked = [
        comparison
        for comparison in result["comparisons"].values()
        if comparison.get("checked", True)
    ]
    result["passed"] = all(
        comparison["max_abs_diff"] <= 1e-6 and comparison["relative_l2"] <= 1e-6
        for comparison in checked
    )
    return result


def build_training_model(
    config: Config,
    *,
    num_classes: int,
    feature_dim: int | None,
) -> tuple[nn.Module, int, str]:
    if feature_dim is not None:
        model = ScaledLinearHead(
            feature_dim,
            num_classes,
            final_logit_scale=config.final_logit_scale,
        )
        return model, config.image_size, "linear_head"
    if config.model_backend == "resnet9":
        routine = (
            ms.SMOOTH_GN_ROUTINE
            if config.resnet_normalization == "gn"
            else ms.SMOOTH_ROUTINE
        )
        model = ms.ResNet9(routine, num_classes=num_classes)
        return model, config.image_size, f"resnet9_smooth_{config.resnet_normalization}"
    model_config = make_model_config(config, num_classes=num_classes)
    return VisionTransformerClassifier(model_config), model_config.image_size, "vit"


def run_granularity(
    config: Config,
    plan: DatasetPlan,
    spec: GranularitySpec,
    *,
    train_dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact,
    val_dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact | None,
    test_dataset: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact | None,
    feature_dim: int | None,
    device: torch.device,
    initial_logits: Tensor | None = None,
) -> dict[str, Any]:
    configure_replay_determinism(config.seed, tf32=False)
    model, image_size, model_name = build_training_model(
        config,
        num_classes=plan.num_classes,
        feature_dim=feature_dim,
    )
    model = model.to(device=device, dtype=torch.float32)
    if config.initial_checkpoint is not None:
        checkpoint = torch.load(config.initial_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError("initial_checkpoint must contain a model state dictionary")
        model.load_state_dict(state_dict)
    if config.model_backend == "resnet9" and config.resnet_normalization == "frozen_bn":
        model.eval()
    else:
        model.train()
    if config.inner_optimizer == "sgd":
        optimizer_config: InnerOptimizerConfig = SmoothSGDConfig(
            learning_rate=config.inner_lr,
            momentum=config.inner_momentum,
            weight_decay=config.inner_weight_decay,
            nesterov=True,
        )
    else:
        optimizer_config = SmoothAdamWConfig(
            learning_rate=config.inner_lr,
            betas=(config.inner_beta1, config.inner_beta2),
            eps=config.inner_eps,
            weight_decay=config.inner_weight_decay,
        )
    initial_state = initialize_train_state(model)
    if initial_logits is None:
        logits = torch.zeros(spec.num_groups, dtype=torch.float32, device=device)
    else:
        if initial_logits.shape != (spec.num_groups,):
            raise ValueError("initial_logits has the wrong shape")
        logits = initial_logits.detach().to(device=device, dtype=torch.float32).clone()
    logits.requires_grad_(True)
    history: list[dict[str, Any]] = []

    for step in range(config.meta_steps + 1):
        step_start = time.perf_counter()
        state = train_state_for_logits(
            model,
            initial_state,
            train_dataset,
            plan.split_train_indices,
            plan.inner_schedule,
            spec,
            optimizer_config,
            image_size,
            logits,
            config,
            device,
        )
        objective = differentiable_mean_cross_entropy(
            model,
            state,
            train_dataset,
            plan.objective_indices,
            batch_size=config.eval_batch_size,
            image_size=image_size,
            device=device,
        )
        record = {
            "step": step,
            "objective_ce": float(objective.detach().item()),
            "objective_metrics": evaluate_indices(
                model,
                state,
                train_dataset,
                plan.objective_indices,
                plan.train_labels,
                batch_size=config.eval_batch_size,
                image_size=image_size,
                device=device,
                num_classes=plan.num_classes,
            ),
            "meta_validation_metrics": evaluate_indices(
                model,
                state,
                train_dataset,
                plan.meta_validation_indices,
                plan.train_labels,
                batch_size=config.eval_batch_size,
                image_size=image_size,
                device=device,
                num_classes=plan.num_classes,
            ),
            "weights": class_mass_summary(logits.detach().cpu(), spec),
            "state_checksum": state_checksum(state),
            "seconds": time.perf_counter() - step_start,
        }
        if val_dataset is not None and plan.val_labels is not None:
            val_indices = torch.arange(plan.val_labels.numel(), dtype=torch.long)
            record["val_metrics"] = evaluate_indices(
                model,
                state,
                val_dataset,
                val_indices,
                plan.val_labels,
                batch_size=config.eval_batch_size,
                image_size=image_size,
                device=device,
                num_classes=plan.num_classes,
            )
        if test_dataset is not None and plan.test_labels is not None:
            test_indices = torch.arange(plan.test_labels.numel(), dtype=torch.long)
            record["test_metrics"] = evaluate_indices(
                model,
                state,
                test_dataset,
                test_indices,
                plan.test_labels,
                batch_size=config.eval_batch_size,
                image_size=image_size,
                device=device,
                num_classes=plan.num_classes,
            )
        if step < config.meta_steps:
            (gradient,) = torch.autograd.grad(objective, logits)
            record["gradient_norm"] = float(gradient.norm().item())
            record["update"] = fixed_kl_signed_update(
                logits,
                gradient,
                spec,
                temperature=config.temperature,
                target_kl=config.outer_step_kl,
            )
        history.append(record)

    add_loss_deltas(history)
    best_index = min(
        range(len(history)),
        key=lambda idx: history[idx]["meta_validation_metrics"]["balanced_ce"],
    )
    return {
        "granularity": spec.name,
        "model": model_name,
        "inner_optimizer": config.inner_optimizer,
        "history": history,
        "selected_step": best_index,
        "selection_rule": "minimum balanced meta-validation CE",
    }


def main(config: Config) -> None:
    validate_config(config)
    device = resolve_device(config.device)
    configure_replay_determinism(config.seed, tf32=False)
    bundle = build_dataset_bundle(config)
    plan = load_dataset_plan(config, bundle)
    feature_bundle = load_feature_bundle(config, bundle, plan, device=device)
    if config.exposed_train_indices_output is not None:
        indices_output = Path(config.exposed_train_indices_output)
        indices_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(plan.split_train_indices.cpu(), indices_output)
    aligned_clusters = None
    if "per_cluster" in config.granularities:
        cluster_artifact_prefix = config.cluster_artifact_prefix or ""
        basis = load_cluster_basis_artifacts(cluster_artifact_prefix, map_location="cpu")
        basis_source_indices = load_cluster_basis_source_indices(cluster_artifact_prefix, map_location="cpu")
        aligned_clusters = align_cluster_basis_to_split(
            basis,
            bundle.train_alignment_labels,
            plan.split_train_indices,
            basis_source_indices=basis_source_indices,
        )
    specs = {
        granularity: build_granularity_spec(
            granularity,
            plan.exposed_train_labels,
            num_classes=plan.num_classes,
            aligned_clusters=aligned_clusters,
        )
        for granularity in config.granularities
    }
    payload = build_preflight_payload(config, plan, specs, aligned_clusters, feature_bundle)
    save_result(config.output, payload, indent=config.json_indent)
    if config.mode in ("dry_run", "preflight"):
        return
    train_source: LazyImageDataset | FeatureArtifact | MaterializedImageArtifact = (
        bundle.train_dataset if feature_bundle is None else feature_bundle.train
    )
    val_source: LazyImageDataset | FeatureArtifact | None = (
        bundle.val_dataset if feature_bundle is None else feature_bundle.val
    )
    test_source: LazyImageDataset | FeatureArtifact | None = (
        bundle.test_dataset if feature_bundle is None else feature_bundle.test
    )
    feature_dim = None if feature_bundle is None else feature_bundle.embedding_dim
    if feature_bundle is None and config.materialize_train_images:
        required_train_indices = torch.cat(
            (
                plan.split_train_indices,
                plan.objective_indices,
                plan.meta_validation_indices,
            )
        )
        train_source = materialize_image_artifact(
            bundle.train_dataset,
            required_train_indices,
            image_size=config.image_size,
            device=device,
        )
    run_kwargs = {
        "train_dataset": train_source,
        "val_dataset": val_source,
        "test_dataset": test_source,
        "feature_dim": feature_dim,
        "device": device,
    }
    control_config = replace(config, meta_steps=0)
    control_spec = build_granularity_spec(
        "per_class",
        plan.exposed_train_labels,
        num_classes=plan.num_classes,
    )
    payload["controls"] = {
        "uniform": run_granularity(
            control_config,
            plan,
            control_spec,
            **run_kwargs,
        ),
        "class_prior_oracle": run_granularity(
            control_config,
            plan,
            control_spec,
            initial_logits=class_prior_oracle_logits(
                control_spec,
                num_classes=plan.num_classes,
            ),
            **run_kwargs,
        ),
    }
    save_result(config.output, payload, indent=config.json_indent)
    payload["methods"] = {}
    for granularity in config.granularities:
        payload["methods"][granularity] = run_granularity(
            config,
            plan,
            specs[granularity],
            **run_kwargs,
        )
        save_result(config.output, payload, indent=config.json_indent)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run ImageNet-LT MGD across weighting granularities.")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-manifest", default=None, help="legacy alias for --val-manifest")
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--parquet-root", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--cluster-artifact-prefix", default=None)
    parser.add_argument("--exposed-train-indices-input", default=None)
    parser.add_argument("--exposed-train-indices-output", default=None)
    parser.add_argument("--features-artifact", default=None)
    parser.add_argument("--val-features-artifact", default=None)
    parser.add_argument("--test-features-artifact", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective-per-class", type=int, default=1)
    parser.add_argument("--meta-validation-per-class", type=int, default=1)
    parser.add_argument("--min-train-per-class", type=int, default=1)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--inner-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-inner-steps", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--inner-weight-decay", type=float, default=0.0)
    parser.add_argument("--inner-beta1", type=float, default=0.9)
    parser.add_argument("--inner-beta2", type=float, default=0.999)
    parser.add_argument("--inner-eps", type=float, default=1e-8)
    parser.add_argument("--inner-optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--inner-momentum", type=float, default=0.9)
    parser.add_argument("--inner-warmup-fraction", type=float, default=0.1)
    parser.add_argument("--meta-steps", type=int, default=0)
    parser.add_argument("--outer-step-kl", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--inner-backward", choices=("replay", "store_all"), default="replay")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("run", "preflight", "dry_run"), default="run")
    parser.add_argument(
        "--granularities",
        type=parse_granularities,
        default=("per_class", "per_cluster", "per_example"),
    )
    parser.add_argument("--model-backend", choices=("vit", "resnet9"), default="vit")
    parser.add_argument("--resnet-normalization", choices=("gn", "frozen_bn"), default="gn")
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument("--model-profile", choices=("vit_s", "smoke"), default="vit_s")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--encoder-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--encoder-heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--pre-norm", dest="pre_norm", action="store_true")
    norm_group.add_argument("--post-norm", dest="pre_norm", action="store_false")
    parser.set_defaults(pre_norm=None)
    parser.add_argument("--pool", choices=("mean", "cls"), default=None)
    parser.add_argument("--final-logit-scale", type=float, default=ViTConfig().final_logit_scale)
    parser.add_argument("--materialize-train-images", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args()
    return Config(
        train_manifest=args.train_manifest,
        output=args.output,
        eval_manifest=args.eval_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        dataset_root=args.dataset_root,
        parquet_root=args.parquet_root,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        cluster_artifact_prefix=args.cluster_artifact_prefix,
        exposed_train_indices_input=args.exposed_train_indices_input,
        exposed_train_indices_output=args.exposed_train_indices_output,
        features_artifact=args.features_artifact,
        val_features_artifact=args.val_features_artifact,
        test_features_artifact=args.test_features_artifact,
        seed=args.seed,
        objective_per_class=args.objective_per_class,
        meta_validation_per_class=args.meta_validation_per_class,
        min_train_per_class=args.min_train_per_class,
        max_train_examples=args.max_train_examples,
        inner_epochs=args.inner_epochs,
        batch_size=args.batch_size,
        max_inner_steps=args.max_inner_steps,
        eval_batch_size=args.eval_batch_size,
        inner_lr=args.inner_lr,
        inner_weight_decay=args.inner_weight_decay,
        inner_beta1=args.inner_beta1,
        inner_beta2=args.inner_beta2,
        inner_eps=args.inner_eps,
        inner_optimizer=args.inner_optimizer,
        inner_momentum=args.inner_momentum,
        inner_warmup_fraction=args.inner_warmup_fraction,
        meta_steps=args.meta_steps,
        outer_step_kl=args.outer_step_kl,
        temperature=args.temperature,
        branching_factor=args.branching_factor,
        inner_backward=args.inner_backward,
        device=args.device,
        mode=args.mode,
        granularities=args.granularities,
        model_backend=args.model_backend,
        resnet_normalization=args.resnet_normalization,
        initial_checkpoint=args.initial_checkpoint,
        model_profile=args.model_profile,
        image_size=args.image_size,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        mlp_ratio=args.mlp_ratio,
        pre_norm=args.pre_norm,
        pool=args.pool,
        final_logit_scale=args.final_logit_scale,
        materialize_train_images=args.materialize_train_images,
        json_indent=args.json_indent,
    )


if __name__ == "__main__":
    main(parse_args())
