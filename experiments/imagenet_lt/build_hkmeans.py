from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from clustering import hierarchical_kmeans_cluster_basis, save_cluster_basis_artifacts
from experiments.imagenet_lt.build_clusters import (
    apply_subset,
    load_completed_embedding_artifact,
    load_subset_indices,
    save_selected_global_indices,
    save_summary,
)


@dataclass(frozen=True)
class BuildConfig:
    embeddings_artifact: str
    artifact_prefix: str
    index_artifact: str | None = None
    index_key: str = "auto"
    max_examples: int | None = None
    num_clusters: int = 1000
    levels: int = 2
    max_iterations: int = 100
    seed: int = 0
    device: str = "auto"
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
    num_clusters: int
    levels: int
    max_iterations: int
    seed: int
    device: str
    cluster_sizes: list[int]
    inertia: float


def parse_args(argv: Sequence[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build global label-agnostic hierarchical k-means ImageNet-LT clusters."
    )
    parser.add_argument("--embeddings-artifact", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--index-artifact", default=None)
    parser.add_argument("--index-key", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-clusters", type=int, default=1000)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json-indent", type=int, default=2)
    args = parser.parse_args(argv)
    return BuildConfig(
        embeddings_artifact=args.embeddings_artifact,
        artifact_prefix=args.artifact_prefix,
        index_artifact=args.index_artifact,
        index_key=args.index_key,
        max_examples=args.max_examples,
        num_clusters=args.num_clusters,
        levels=args.levels,
        max_iterations=args.max_iterations,
        seed=args.seed,
        device=args.device,
        json_indent=args.json_indent,
    )


def validate_config(config: BuildConfig) -> None:
    if config.max_examples is not None and config.max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    if config.num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if config.levels != 2:
        raise ValueError("faithful hierarchical k-means currently requires levels=2")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if config.json_indent < 0:
        raise ValueError("json_indent must be non-negative")


def build_clusters(config: BuildConfig) -> BuildSummary:
    validate_config(config)
    artifact, embeddings, labels, global_indices = load_completed_embedding_artifact(
        config.embeddings_artifact
    )
    metadata = artifact["metadata"]
    subset_indices, resolved_index_key = load_subset_indices(
        config.index_artifact,
        index_key=config.index_key,
        total_examples=embeddings.shape[0],
    )
    selected_embeddings, _, selected_global_indices = apply_subset(
        embeddings,
        labels,
        global_indices,
        subset_indices=subset_indices,
        max_examples=config.max_examples,
    )
    if config.num_clusters > selected_embeddings.shape[0]:
        raise ValueError("num_clusters cannot exceed the selected example count")

    device = resolve_device(config.device)
    basis = hierarchical_kmeans_cluster_basis(
        selected_embeddings.to(device=device),
        num_clusters=config.num_clusters,
        levels=config.levels,
        max_iterations=config.max_iterations,
        seed=config.seed,
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
        index_artifact=None
        if config.index_artifact is None
        else str(Path(config.index_artifact).resolve()),
        index_key=resolved_index_key,
        max_examples=config.max_examples,
        selected_global_index_min=int(selected_global_indices.min().item()),
        selected_global_index_max=int(selected_global_indices.max().item()),
        selected_global_index_preview=[int(value) for value in selected_global_indices[:16].tolist()],
        num_clusters=basis.num_groups,
        levels=basis.levels,
        max_iterations=config.max_iterations,
        seed=basis.seed,
        device=str(device),
        cluster_sizes=[int(value) for value in basis.cluster_sizes.tolist()],
        inertia=float(basis.inertia),
    )
    save_summary(summary_path, asdict(summary), indent=config.json_indent)
    return summary


def resolve_device(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def summary_to_json(summary: BuildSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> None:
    print(summary_to_json(build_clusters(parse_args(argv))))


if __name__ == "__main__":
    main()
