"""Label-agnostic hierarchical k-means cluster basis (standalone port).

This is the slice of the ImageNet-LT worktree's ``clustering.py`` needed for the
CIFAR100-LT / ViT-Tiny per-cluster MGD comparison: a deterministic two-level
k-means over embeddings, ignoring labels.  Clusters may mix classes.  The
class-pure tiered basis from the worktree is intentionally omitted -- only the
label-agnostic basis is used here.

Determinism: k-means++ seeding picks the ``seed``-th point first, then always
takes the farthest remaining point (no RNG), and Lloyd iterations are exact, so
``hierarchical_kmeans_cluster_basis`` is a pure function of (embeddings, knobs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ClusterBasis:
    """Result of a label-agnostic hierarchical k-means over embeddings."""

    group_ids: Tensor  # [n] cluster assignment per embedding
    base_group_masses: Tensor  # [num_clusters] cluster mass = size / n
    cluster_sizes: Tensor  # [num_clusters] example count per cluster
    cluster_centers: Tensor  # [num_clusters, dim] centroids (normalized space)
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
    """Two-level k-means: sqrt(num_clusters) parents, then balanced leaves."""
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
    level_one_sizes = torch.bincount(level_one_assignments, minlength=level_one_clusters)
    leaf_counts = _balanced_leaf_counts(level_one_sizes, num_clusters=num_clusters)

    group_ids = torch.empty(normalized.shape[0], dtype=torch.long, device=normalized.device)
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
        cluster_sizes, dtype=torch.long, device=embeddings.device
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


def _normalize_embeddings(embeddings: Tensor) -> Tensor:
    norms = embeddings.norm(dim=1)
    if torch.any(norms <= 0):
        raise ValueError("embeddings must have non-zero row norm")
    return F.normalize(embeddings, dim=1)


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
    assignments = torch.full((points.shape[0],), -1, dtype=torch.long, device=points.device)

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
