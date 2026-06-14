from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from clustering import (
    ClusterTierConfig,
    build_fixed_intermediate_cluster_basis,
    save_cluster_basis_artifacts,
)
from experiments.extract_imagenet_lt_embeddings import load_existing_artifact


@dataclass(frozen=True)
class BuildConfig:
    embeddings_artifact: str
    artifact_prefix: str
    index_artifact: str | None = None
    index_key: str = "auto"
    max_examples: int | None = None
    many_threshold: int = 100
    few_threshold: int = 20
    many_clusters: int = 4
    medium_clusters: int = 2
    few_clusters: int = 1
    max_iterations: int = 100
    json_indent: int = 2


@dataclass(frozen=True)
class BuildSummary:
    embeddings_artifact: str
    artifact_prefix: str
    cluster_json: str
    cluster_pt: str
    cluster_indices_pt: str
    summary_json: str
    split: str | None
    backbone: str | None
    total_input_examples: int
    selected_examples: int
    embedding_dim: int
    index_artifact: str | None
    index_key: str | None
    max_examples: int | None
    selected_global_index_min: int
    selected_global_index_max: int
    selected_global_index_preview: list[int]
    many_threshold: int
    few_threshold: int
    tier_config: dict[str, int]
    max_iterations: int
    num_groups: int
    class_counts: dict[str, int]
    clusters_per_class: dict[str, int]
    inertia: float


def parse_args(argv: Sequence[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build fixed ImageNet-LT cluster artifacts from a completed embedding artifact."
    )
    parser.add_argument("--embeddings-artifact", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--index-artifact", default=None)
    parser.add_argument("--index-key", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--many-threshold", type=int, default=100)
    parser.add_argument("--few-threshold", type=int, default=20)
    parser.add_argument("--many-clusters", type=int, default=4)
    parser.add_argument("--medium-clusters", type=int, default=2)
    parser.add_argument("--few-clusters", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args(argv)
    return BuildConfig(
        embeddings_artifact=args.embeddings_artifact,
        artifact_prefix=args.artifact_prefix,
        index_artifact=args.index_artifact,
        index_key=args.index_key,
        max_examples=args.max_examples,
        many_threshold=args.many_threshold,
        few_threshold=args.few_threshold,
        many_clusters=args.many_clusters,
        medium_clusters=args.medium_clusters,
        few_clusters=args.few_clusters,
        max_iterations=args.max_iterations,
        json_indent=args.json_indent,
    )


def validate_config(config: BuildConfig) -> None:
    if config.max_examples is not None and config.max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    if config.many_threshold < config.few_threshold:
        raise ValueError("many_threshold must be at least few_threshold")
    if config.few_threshold <= 0:
        raise ValueError("few_threshold must be positive")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.json_indent < 0:
        raise ValueError("json_indent must be non-negative")
    ClusterTierConfig(
        many=config.many_clusters,
        medium=config.medium_clusters,
        few=config.few_clusters,
    )


def _ensure_long_vector(name: str, value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer indices")
    return value.to(dtype=torch.long, device="cpu")


def load_completed_embedding_artifact(path: str | Path) -> tuple[dict[str, Any], Tensor, Tensor, Tensor]:
    artifact = load_existing_artifact(path)
    if artifact is None:
        raise ValueError(f"embedding artifact does not exist: {Path(path)}")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("embedding artifact metadata must be a dictionary")
    target_examples = int(metadata.get("target_examples", -1))
    if target_examples <= 0:
        raise ValueError("embedding artifact target_examples must be positive")
    embedding_dim = int(artifact.get("embedding_dim", -1))
    if embedding_dim <= 0:
        raise ValueError("embedding artifact embedding_dim must be positive")
    completed = int(artifact.get("completed", -1))
    if completed < 0 or completed > target_examples:
        raise ValueError("embedding artifact completed count is invalid")
    if completed != target_examples:
        raise ValueError(
            f"embedding artifact is incomplete: completed={completed}, target_examples={target_examples}"
        )
    embeddings = artifact["embeddings"]
    labels = artifact["labels"]
    global_indices = artifact["global_indices"]
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        raise ValueError("embedding artifact embeddings must be a two-dimensional tensor")
    if not torch.is_floating_point(embeddings):
        raise ValueError("embedding artifact embeddings must be floating point")
    labels = _ensure_long_vector("labels", labels)
    global_indices = _ensure_long_vector("global_indices", global_indices)
    if embeddings.shape != (target_examples, embedding_dim):
        raise ValueError("embedding artifact embeddings tensor has the wrong shape")
    if embeddings.shape[0] != labels.numel() or labels.shape != global_indices.shape:
        raise ValueError("embedding artifact tensors must agree on the number of examples")
    if torch.unique(global_indices).numel() != global_indices.numel():
        raise ValueError("embedding artifact global indices must be unique")
    total_dataset_examples = metadata.get("total_dataset_examples")
    if isinstance(total_dataset_examples, int) and total_dataset_examples > 0:
        if torch.any(global_indices < 0) or torch.any(global_indices >= total_dataset_examples):
            raise ValueError("embedding artifact global indices are out of range for the full dataset")
    return artifact, embeddings.to(device="cpu"), labels, global_indices


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


def load_subset_indices(
    index_artifact: str | None,
    *,
    index_key: str,
    total_examples: int,
) -> tuple[Tensor, str | None]:
    if index_artifact is None:
        return torch.arange(total_examples, dtype=torch.long), None
    payload = torch.load(index_artifact, map_location="cpu")
    indices, resolved_key = _extract_indices_from_payload(payload, index_key=index_key)
    if indices.numel() == 0:
        raise ValueError("index artifact must select at least one example")
    if torch.any(indices < 0) or torch.any(indices >= total_examples):
        raise ValueError("index artifact contains an out-of-range example index")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("index artifact must not contain duplicate example indices")
    return indices, resolved_key


def apply_subset(
    embeddings: Tensor,
    labels: Tensor,
    global_indices: Tensor,
    *,
    subset_indices: Tensor,
    max_examples: int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    selected = subset_indices
    if max_examples is not None:
        selected = selected[:max_examples]
    if selected.numel() == 0:
        raise ValueError("selection is empty after applying index_artifact/max_examples")
    return (
        embeddings.index_select(0, selected),
        labels.index_select(0, selected),
        global_indices.index_select(0, selected),
    )


def save_summary(path: str | Path, payload: dict[str, Any], *, indent: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def save_selected_global_indices(path: str | Path, global_indices: Tensor) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_indices.to(dtype=torch.long, device="cpu"), destination)
    return destination


def build_clusters(config: BuildConfig) -> BuildSummary:
    validate_config(config)
    artifact, embeddings, labels, global_indices = load_completed_embedding_artifact(config.embeddings_artifact)
    metadata = artifact["metadata"]
    subset_indices, resolved_index_key = load_subset_indices(
        config.index_artifact,
        index_key=config.index_key,
        total_examples=embeddings.shape[0],
    )
    selected_embeddings, selected_labels, selected_global_indices = apply_subset(
        embeddings,
        labels,
        global_indices,
        subset_indices=subset_indices,
        max_examples=config.max_examples,
    )

    tier_config = ClusterTierConfig(
        many=config.many_clusters,
        medium=config.medium_clusters,
        few=config.few_clusters,
    )
    basis = build_fixed_intermediate_cluster_basis(
        selected_embeddings,
        selected_labels,
        many_threshold=config.many_threshold,
        few_threshold=config.few_threshold,
        tier_config=tier_config,
        max_iterations=config.max_iterations,
    )
    cluster_json, cluster_pt = save_cluster_basis_artifacts(basis, config.artifact_prefix)
    cluster_indices_pt = save_selected_global_indices(
        Path(config.artifact_prefix).with_suffix(".indices.pt"),
        selected_global_indices,
    )
    summary_path = Path(config.artifact_prefix).with_suffix(".summary.json")
    summary = BuildSummary(
        embeddings_artifact=str(Path(config.embeddings_artifact).resolve()),
        artifact_prefix=str(Path(config.artifact_prefix).resolve()),
        cluster_json=str(cluster_json.resolve()),
        cluster_pt=str(cluster_pt.resolve()),
        cluster_indices_pt=str(cluster_indices_pt.resolve()),
        summary_json=str(summary_path.resolve()),
        split=metadata.get("split"),
        backbone=metadata.get("backbone"),
        total_input_examples=int(embeddings.shape[0]),
        selected_examples=int(selected_embeddings.shape[0]),
        embedding_dim=int(selected_embeddings.shape[1]),
        index_artifact=None if config.index_artifact is None else str(Path(config.index_artifact).resolve()),
        index_key=resolved_index_key,
        max_examples=config.max_examples,
        selected_global_index_min=int(selected_global_indices.min().item()),
        selected_global_index_max=int(selected_global_indices.max().item()),
        selected_global_index_preview=[int(value) for value in selected_global_indices[:16].tolist()],
        many_threshold=config.many_threshold,
        few_threshold=config.few_threshold,
        tier_config=asdict(tier_config),
        max_iterations=config.max_iterations,
        num_groups=basis.num_groups,
        class_counts={str(class_id): count for class_id, count in basis.class_counts.items()},
        clusters_per_class={
            str(class_id): count for class_id, count in basis.clusters_per_class.items()
        },
        inertia=float(basis.inertia),
    )
    save_summary(summary_path, asdict(summary), indent=config.json_indent)
    return summary


def summary_to_json(summary: BuildSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> None:
    summary = build_clusters(parse_args(argv))
    print(summary_to_json(summary))


if __name__ == "__main__":
    main()
