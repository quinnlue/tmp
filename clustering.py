from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ClusterTierConfig:
    many: int = 4
    medium: int = 2
    few: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("many", self.many),
            ("medium", self.medium),
            ("few", self.few),
        ):
            if value <= 0:
                raise ValueError(f"{name} clusters per class must be positive")


@dataclass(frozen=True)
class ClusterBasis:
    group_ids: Tensor
    base_group_masses: Tensor
    cluster_sizes: Tensor
    cluster_centers: Tensor
    levels: int = 2
    seed: int = 0
    inertia: float = 0.0

    def __post_init__(self) -> None:
        _validate_cluster_tensors(
            self.group_ids,
            self.base_group_masses,
            self.cluster_sizes,
            self.cluster_centers,
        )
        if self.levels <= 0:
            raise ValueError("levels must be positive")

    @property
    def num_groups(self) -> int:
        return int(self.base_group_masses.numel())

    def diagnostics(self) -> dict[str, Any]:
        return {
            "basis_type": "label_agnostic_hierarchical_kmeans",
            "cluster_sizes": self.cluster_sizes.tolist(),
            "inertia": self.inertia,
            "levels": self.levels,
            "num_groups": self.num_groups,
            "num_examples": int(self.group_ids.numel()),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class FixedClusterBasis:
    group_ids: Tensor
    base_group_masses: Tensor
    cluster_sizes: Tensor
    cluster_centers: Tensor
    class_to_cluster_ids: dict[int, tuple[int, ...]]
    class_counts: dict[int, int]
    clusters_per_class: dict[int, int]
    many_classes: tuple[int, ...]
    medium_classes: tuple[int, ...]
    few_classes: tuple[int, ...]
    many_threshold: int
    few_threshold: int
    tier_config: ClusterTierConfig
    inertia: float

    def __post_init__(self) -> None:
        _validate_cluster_tensors(
            self.group_ids,
            self.base_group_masses,
            self.cluster_sizes,
            self.cluster_centers,
        )

    @property
    def num_groups(self) -> int:
        return int(self.base_group_masses.numel())

    def diagnostics(self) -> dict[str, Any]:
        return {
            "basis_type": "class_pure_tiered_kmeans",
            "cluster_sizes": self.cluster_sizes.tolist(),
            "class_to_cluster_ids": {
                str(class_id): list(cluster_ids)
                for class_id, cluster_ids in self.class_to_cluster_ids.items()
            },
            "class_counts": {str(class_id): count for class_id, count in self.class_counts.items()},
            "clusters_per_class": {
                str(class_id): count for class_id, count in self.clusters_per_class.items()
            },
            "many_classes": list(self.many_classes),
            "medium_classes": list(self.medium_classes),
            "few_classes": list(self.few_classes),
            "many_threshold": self.many_threshold,
            "few_threshold": self.few_threshold,
            "tier_config": {
                "many": self.tier_config.many,
                "medium": self.tier_config.medium,
                "few": self.tier_config.few,
            },
            "inertia": self.inertia,
            "num_groups": self.num_groups,
            "num_examples": int(self.group_ids.numel()),
        }


def _validate_cluster_tensors(
    group_ids: Tensor,
    base_group_masses: Tensor,
    cluster_sizes: Tensor,
    cluster_centers: Tensor,
) -> None:
    if group_ids.ndim != 1 or group_ids.dtype != torch.long:
        raise ValueError("group_ids must be a one-dimensional torch.long tensor")
    if base_group_masses.ndim != 1 or cluster_sizes.ndim != 1:
        raise ValueError("base_group_masses and cluster_sizes must be one-dimensional")
    if cluster_centers.ndim != 2:
        raise ValueError("cluster_centers must be a two-dimensional tensor")
    num_groups = base_group_masses.numel()
    if cluster_sizes.numel() != num_groups:
        raise ValueError("cluster_sizes must have the same length as base_group_masses")
    if cluster_centers.shape[0] != num_groups:
        raise ValueError("cluster_centers must have one row per cluster")
    if group_ids.numel() and (
        torch.any(group_ids < 0) or torch.any(group_ids >= num_groups)
    ):
        raise ValueError("group_ids contains an invalid cluster id")
    if torch.any(base_group_masses <= 0):
        raise ValueError("base_group_masses must be strictly positive")
    if torch.any(cluster_sizes <= 0):
        raise ValueError("cluster_sizes must be strictly positive")
    if not torch.isclose(
        base_group_masses.sum(),
        base_group_masses.new_tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("base_group_masses must sum to one")


def hierarchical_kmeans_cluster_basis(
    embeddings: Tensor,
    *,
    num_clusters: int,
    levels: int = 2,
    max_iterations: int = 100,
    seed: int = 0,
) -> ClusterBasis:
    if embeddings.ndim != 2 or embeddings.numel() == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional tensor")
    if not torch.is_floating_point(embeddings):
        raise ValueError("embeddings must be floating point")
    if num_clusters <= 0 or num_clusters > embeddings.shape[0]:
        raise ValueError("num_clusters must be between 1 and the number of embeddings")
    if levels != 2:
        raise ValueError("faithful hierarchical k-means currently requires levels=2")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    normalized = _normalize_embeddings(embeddings)
    level_one_clusters = min(
        num_clusters,
        max(1, int(round(math.sqrt(num_clusters)))),
    )
    level_one_assignments, _, _ = _deterministic_kmeans(
        normalized,
        num_clusters=level_one_clusters,
        max_iterations=max_iterations,
        seed=seed,
    )
    level_one_sizes = torch.bincount(
        level_one_assignments,
        minlength=level_one_clusters,
    )
    leaf_counts = _balanced_leaf_counts(level_one_sizes, num_clusters=num_clusters)

    group_ids = torch.empty(
        normalized.shape[0],
        dtype=torch.long,
        device=normalized.device,
    )
    cluster_sizes: list[int] = []
    cluster_centers: list[Tensor] = []
    total_inertia = 0.0
    next_group_id = 0
    for parent_id, leaf_count_tensor in enumerate(leaf_counts):
        parent_indices = torch.where(level_one_assignments == parent_id)[0]
        leaf_count = int(leaf_count_tensor.item())
        parent_points = normalized.index_select(0, parent_indices)
        assignments, centers, inertia = _deterministic_kmeans(
            parent_points,
            num_clusters=leaf_count,
            max_iterations=max_iterations,
            seed=seed + parent_id + 1,
        )
        group_ids[parent_indices] = assignments + next_group_id
        sizes = torch.bincount(assignments, minlength=leaf_count)
        cluster_sizes.extend(int(size) for size in sizes.tolist())
        cluster_centers.extend(centers.unbind(0))
        total_inertia += inertia
        next_group_id += leaf_count

    if next_group_id != num_clusters:
        raise RuntimeError(
            f"hierarchical k-means produced {next_group_id} clusters instead of {num_clusters}"
        )
    cluster_sizes_tensor = torch.tensor(
        cluster_sizes,
        dtype=torch.long,
        device=embeddings.device,
    )
    return ClusterBasis(
        group_ids=group_ids,
        base_group_masses=cluster_sizes_tensor.to(embeddings.dtype) / float(embeddings.shape[0]),
        cluster_sizes=cluster_sizes_tensor,
        cluster_centers=torch.stack(cluster_centers, dim=0).to(embeddings.device),
        levels=levels,
        seed=seed,
        inertia=total_inertia,
    )


def _balanced_leaf_counts(level_one_sizes: Tensor, *, num_clusters: int) -> Tensor:
    if level_one_sizes.ndim != 1 or level_one_sizes.dtype != torch.long:
        raise ValueError("level_one_sizes must be a one-dimensional torch.long tensor")
    if torch.any(level_one_sizes <= 0):
        raise ValueError("level-one clusters must be non-empty")
    if num_clusters < level_one_sizes.numel() or num_clusters > int(level_one_sizes.sum().item()):
        raise ValueError("num_clusters is incompatible with level-one cluster capacities")
    result = torch.ones_like(level_one_sizes)
    remaining = num_clusters - result.numel()
    while remaining > 0:
        progressed = False
        for parent_id in range(result.numel()):
            if result[parent_id] < level_one_sizes[parent_id]:
                result[parent_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("failed to allocate hierarchical leaf clusters")
    return result


def build_fixed_intermediate_cluster_basis(
    embeddings: Tensor,
    labels: Tensor,
    *,
    many_threshold: int = 100,
    few_threshold: int = 20,
    tier_config: ClusterTierConfig | None = None,
    max_iterations: int = 100,
) -> FixedClusterBasis:
    if embeddings.ndim != 2 or embeddings.numel() == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional tensor")
    if not torch.is_floating_point(embeddings):
        raise ValueError("embeddings must be floating point")
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("labels must be a one-dimensional torch.long tensor")
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels and embeddings must agree on the number of examples")
    if many_threshold < few_threshold:
        raise ValueError("many_threshold must be at least few_threshold")
    if few_threshold <= 0:
        raise ValueError("few_threshold must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    tier_config = tier_config or ClusterTierConfig()
    normalized = _normalize_embeddings(embeddings)
    class_ids = sorted(int(class_id) for class_id in torch.unique(labels, sorted=True).tolist())
    class_counts = {
        class_id: int((labels == class_id).sum().item())
        for class_id in class_ids
    }

    many_classes, medium_classes, few_classes = _partition_classes(
        class_counts,
        many_threshold=many_threshold,
        few_threshold=few_threshold,
    )

    group_ids = torch.empty_like(labels)
    cluster_sizes: list[int] = []
    cluster_centers: list[Tensor] = []
    class_to_cluster_ids: dict[int, tuple[int, ...]] = {}
    clusters_per_class: dict[int, int] = {}
    total_inertia = 0.0
    next_group_id = 0

    for class_id in class_ids:
        class_mask = labels == class_id
        class_indices = torch.where(class_mask)[0]
        class_embeddings = normalized[class_indices]
        requested_clusters = _requested_clusters_for_class(
            class_id,
            many_classes=many_classes,
            medium_classes=medium_classes,
            few_classes=few_classes,
            tier_config=tier_config,
        )
        num_clusters = min(requested_clusters, class_embeddings.shape[0])
        assignments, centers, inertia = _deterministic_kmeans(
            class_embeddings,
            num_clusters=num_clusters,
            max_iterations=max_iterations,
        )
        global_cluster_ids = assignments + next_group_id
        group_ids[class_indices] = global_cluster_ids
        sizes = torch.bincount(assignments, minlength=num_clusters)
        cluster_sizes.extend(int(size) for size in sizes.tolist())
        cluster_centers.extend(centers.unbind(0))
        class_to_cluster_ids[class_id] = tuple(range(next_group_id, next_group_id + num_clusters))
        clusters_per_class[class_id] = num_clusters
        total_inertia += float(inertia)
        next_group_id += num_clusters

    cluster_sizes_tensor = torch.tensor(
        cluster_sizes,
        dtype=torch.long,
        device=labels.device,
    )
    base_group_masses = cluster_sizes_tensor.to(dtype=embeddings.dtype) / float(labels.numel())
    cluster_centers_tensor = torch.stack(cluster_centers, dim=0).to(device=embeddings.device)
    return FixedClusterBasis(
        group_ids=group_ids,
        base_group_masses=base_group_masses,
        cluster_sizes=cluster_sizes_tensor,
        cluster_centers=cluster_centers_tensor,
        class_to_cluster_ids=class_to_cluster_ids,
        class_counts=class_counts,
        clusters_per_class=clusters_per_class,
        many_classes=many_classes,
        medium_classes=medium_classes,
        few_classes=few_classes,
        many_threshold=many_threshold,
        few_threshold=few_threshold,
        tier_config=tier_config,
        inertia=total_inertia,
    )


def save_cluster_basis_artifacts(
    basis: FixedClusterBasis | ClusterBasis,
    artifact_prefix: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(artifact_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    pt_path = prefix.with_suffix(".pt")
    json_path.write_text(json.dumps(basis.diagnostics(), indent=2, sort_keys=True), encoding="utf-8")
    torch.save(
        {
            "group_ids": basis.group_ids.cpu(),
            "base_group_masses": basis.base_group_masses.cpu(),
            "cluster_sizes": basis.cluster_sizes.cpu(),
            "cluster_centers": basis.cluster_centers.cpu(),
        },
        pt_path,
    )
    return json_path, pt_path


def load_cluster_basis_artifacts(
    artifact_prefix: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> FixedClusterBasis | ClusterBasis:
    prefix = Path(artifact_prefix)
    json_path = prefix.with_suffix(".json")
    pt_path = prefix.with_suffix(".pt")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    payload = torch.load(pt_path, map_location=map_location)
    if metadata.get("basis_type") == "label_agnostic_hierarchical_kmeans":
        return ClusterBasis(
            group_ids=payload["group_ids"].to(dtype=torch.long),
            base_group_masses=payload["base_group_masses"],
            cluster_sizes=payload["cluster_sizes"].to(dtype=torch.long),
            cluster_centers=payload["cluster_centers"],
            levels=int(metadata["levels"]),
            seed=int(metadata["seed"]),
            inertia=float(metadata["inertia"]),
        )
    tier_config = ClusterTierConfig(**metadata["tier_config"])
    return FixedClusterBasis(
        group_ids=payload["group_ids"].to(dtype=torch.long),
        base_group_masses=payload["base_group_masses"],
        cluster_sizes=payload["cluster_sizes"].to(dtype=torch.long),
        cluster_centers=payload["cluster_centers"],
        class_to_cluster_ids={
            int(class_id): tuple(int(group_id) for group_id in cluster_ids)
            for class_id, cluster_ids in metadata["class_to_cluster_ids"].items()
        },
        class_counts={
            int(class_id): int(count)
            for class_id, count in metadata["class_counts"].items()
        },
        clusters_per_class={
            int(class_id): int(count)
            for class_id, count in metadata["clusters_per_class"].items()
        },
        many_classes=tuple(int(class_id) for class_id in metadata["many_classes"]),
        medium_classes=tuple(int(class_id) for class_id in metadata["medium_classes"]),
        few_classes=tuple(int(class_id) for class_id in metadata["few_classes"]),
        many_threshold=int(metadata["many_threshold"]),
        few_threshold=int(metadata["few_threshold"]),
        tier_config=tier_config,
        inertia=float(metadata["inertia"]),
    )


def _normalize_embeddings(embeddings: Tensor) -> Tensor:
    norms = embeddings.norm(dim=1)
    if torch.any(norms <= 0):
        raise ValueError("embeddings must have non-zero row norm")
    return F.normalize(embeddings, dim=1)


def _partition_classes(
    class_counts: dict[int, int],
    *,
    many_threshold: int,
    few_threshold: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    many: list[int] = []
    medium: list[int] = []
    few: list[int] = []
    for class_id, count in class_counts.items():
        if count > many_threshold:
            many.append(class_id)
        elif count < few_threshold:
            few.append(class_id)
        else:
            medium.append(class_id)
    return tuple(many), tuple(medium), tuple(few)


def _requested_clusters_for_class(
    class_id: int,
    *,
    many_classes: tuple[int, ...],
    medium_classes: tuple[int, ...],
    few_classes: tuple[int, ...],
    tier_config: ClusterTierConfig,
) -> int:
    if class_id in many_classes:
        return tier_config.many
    if class_id in medium_classes:
        return tier_config.medium
    if class_id in few_classes:
        return tier_config.few
    raise AssertionError(f"class {class_id} was not assigned to any frequency tier")


def _deterministic_kmeans(
    points: Tensor,
    *,
    num_clusters: int,
    max_iterations: int,
    seed: int = 0,
) -> tuple[Tensor, Tensor, float]:
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty two-dimensional tensor")
    if num_clusters <= 0 or num_clusters > points.shape[0]:
        raise ValueError("num_clusters must be between 1 and the number of points")

    center_indices = _deterministic_kmeans_plus_plus(points, num_clusters, seed=seed)
    centers = points[center_indices].clone()
    assignments = torch.full(
        (points.shape[0],),
        -1,
        dtype=torch.long,
        device=points.device,
    )

    for _ in range(max_iterations):
        distances = torch.cdist(points, centers).square()
        next_assignments = distances.argmin(dim=1)
        if torch.equal(next_assignments, assignments):
            break
        assignments = next_assignments
        updated_centers = centers.clone()
        point_to_center = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
        for cluster_id in range(num_clusters):
            members = points[assignments == cluster_id]
            if members.numel() == 0:
                replacement = point_to_center.argmax()
                updated_centers[cluster_id] = points[replacement]
            else:
                updated_centers[cluster_id] = members.mean(dim=0)
        centers = updated_centers

    final_distances = torch.cdist(points, centers).square()
    assignments = final_distances.argmin(dim=1)
    assignments = _repair_empty_assignments(assignments, final_distances, num_clusters)
    centers = torch.stack(
        [points[assignments == cluster_id].mean(dim=0) for cluster_id in range(num_clusters)],
        dim=0,
    )
    final_distances = torch.cdist(points, centers).square()
    inertia = float(final_distances.gather(1, assignments.unsqueeze(1)).sum().item())
    return assignments, centers, inertia


def _repair_empty_assignments(assignments: Tensor, distances: Tensor, num_clusters: int) -> Tensor:
    repaired = assignments.clone()
    counts = torch.bincount(repaired, minlength=num_clusters)
    for empty_cluster in torch.where(counts == 0)[0].tolist():
        assigned_distances = distances.gather(1, repaired.unsqueeze(1)).squeeze(1)
        donors = counts[repaired] > 1
        if not torch.any(donors):
            raise RuntimeError("cannot repair empty cluster assignments")
        candidate_scores = assigned_distances.masked_fill(~donors, -1.0)
        candidate = int(candidate_scores.argmax().item())
        donor_cluster = int(repaired[candidate].item())
        repaired[candidate] = empty_cluster
        counts[donor_cluster] -= 1
        counts[empty_cluster] += 1
    return repaired


def _deterministic_kmeans_plus_plus(points: Tensor, num_clusters: int, *, seed: int = 0) -> Tensor:
    first = int(seed) % points.shape[0]
    chosen = [first]
    min_distance_sq = torch.cdist(points, points[[first]]).square().squeeze(1)
    while len(chosen) < num_clusters:
        candidate = int(min_distance_sq.argmax().item())
        if candidate in chosen:
            remaining = [index for index in range(points.shape[0]) if index not in chosen]
            candidate = remaining[0]
        chosen.append(candidate)
        distance_sq = torch.cdist(points, points[[candidate]]).square().squeeze(1)
        min_distance_sq = torch.minimum(min_distance_sq, distance_sq)
    return torch.tensor(chosen, dtype=torch.long, device=points.device)
