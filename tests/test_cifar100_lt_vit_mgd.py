"""CPU unit tests for the CIFAR100-LT / ViT-Tiny MGD comparison driver.

Fast and GPU-free: tiny synthetic tensors and a 1-layer ViT exercise the
granularity specs, the z=0 == ordinary-training invariant, the clustering
basis, and one end-to-end REPLAY metagradient search step.
"""
from __future__ import annotations

import torch

import _run_cifar100_lt_vit_mgd as mgd
from _cluster_basis import hierarchical_kmeans_cluster_basis
from model import ViTConfig, VisionTransformerClassifier
from weighting import weighted_example_loss


def _materialized(images, labels) -> mgd.Materialized:
    counts = torch.bincount(labels, minlength=int(labels.max()) + 1)
    return mgd.Materialized(
        train_images=images, train_labels=labels,
        val_images=images, val_labels=labels,
        test_images=images, test_labels=labels, train_counts=counts,
    )


# --------------------------------------------------------------------------- #
# Granularity specs
# --------------------------------------------------------------------------- #
def test_build_spec_dims_and_base_masses() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], dtype=torch.long)  # long-tailed
    num_classes = 4

    per_class = mgd.build_spec("per_class", labels, num_classes=num_classes)
    assert per_class.num_groups == num_classes
    assert torch.equal(per_class.group_ids, labels)
    torch.testing.assert_close(per_class.base_group_masses.sum(), torch.tensor(1.0))
    # base masses follow class frequencies (class 0 most common).
    assert per_class.base_group_masses.argmax().item() == 0

    per_example = mgd.build_spec("per_example", labels, num_classes=num_classes)
    assert per_example.num_groups == labels.numel()
    torch.testing.assert_close(
        per_example.base_group_masses,
        torch.full((labels.numel(),), 1.0 / labels.numel()),
    )

    cluster_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    cluster_base = torch.bincount(cluster_ids).float()
    cluster_base = cluster_base / cluster_base.sum()
    per_cluster = mgd.build_spec(
        "per_cluster", labels, num_classes=num_classes,
        cluster_group_ids=cluster_ids, cluster_base_masses=cluster_base,
    )
    assert per_cluster.num_groups == 3
    torch.testing.assert_close(per_cluster.base_group_masses.sum(), torch.tensor(1.0))
    assert per_cluster.group_class_ids.numel() == 3


# --------------------------------------------------------------------------- #
# z = 0 is ordinary (unweighted) training for ANY base distribution
# --------------------------------------------------------------------------- #
def test_zero_z_recovers_unweighted_loss() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], dtype=torch.long)
    spec = mgd.build_spec("per_class", labels, num_classes=4)  # long-tailed base
    per_example_loss = torch.tensor([1.0, 2.0, 0.5, 3.0, 0.2, 1.5, 0.8, 2.2])

    z = torch.zeros(spec.num_groups)
    effective = mgd.effective_group_logits(z, spec.base_group_masses, temperature=1.0)
    weighted = weighted_example_loss(
        per_example_loss, effective, spec.group_ids, spec.base_group_masses, 1.0
    )
    torch.testing.assert_close(weighted, per_example_loss.mean(), atol=1e-6, rtol=0.0)
    # And the target group distribution at z=0 equals the base.
    masses = mgd.effective_group_masses(z, spec.base_group_masses, 1.0)
    torch.testing.assert_close(masses, spec.base_group_masses, atol=1e-6, rtol=0.0)


# --------------------------------------------------------------------------- #
# Clustering basis
# --------------------------------------------------------------------------- #
def test_hierarchical_kmeans_is_deterministic_and_covers() -> None:
    embeddings = torch.randn(64, 8, generator=torch.Generator().manual_seed(3))
    first = hierarchical_kmeans_cluster_basis(embeddings, num_clusters=8, seed=0)
    second = hierarchical_kmeans_cluster_basis(embeddings, num_clusters=8, seed=0)
    assert torch.equal(first.group_ids, second.group_ids)
    assert first.num_groups == 8
    assert set(first.group_ids.tolist()) == set(range(8))  # every cluster used
    torch.testing.assert_close(first.base_group_masses.sum(), torch.tensor(1.0))


# --------------------------------------------------------------------------- #
# End-to-end REPLAY metagradient smoke
# --------------------------------------------------------------------------- #
def test_search_smoke_moves_weights() -> None:
    generator = torch.Generator().manual_seed(0)
    n_pool = 16
    labels = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
    images = torch.randn(n_pool, 3, 32, 32, generator=generator)
    data = _materialized(images, labels)
    objective_images = torch.randn(8, 3, 32, 32, generator=generator)
    objective_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

    model = VisionTransformerClassifier(
        ViTConfig(
            image_size=32, patch_size=16, encoder_dim=16, encoder_depth=1,
            encoder_heads=2, mlp_ratio=1.0, num_classes=4, final_logit_scale=10.0,
        )
    )
    config = mgd.Config(
        seed=0, search_epochs=2, batch_size=8, eval_batch_size=8,
        meta_steps=1, inner_lr=1e-3, weight_decay=0.0, temperature=1.0,
        outer_step_kl=1e-3, branching_factor=2,
    )
    spec = mgd.build_spec("per_class", labels, num_classes=4)
    try:
        result = mgd.run_search(
            config, spec, data,
            objective_images=objective_images, objective_labels=objective_labels,
            meta_images=objective_images, meta_labels=objective_labels,
            device=torch.device("cpu"), num_classes=4, model=model,
        )
    finally:
        torch.use_deterministic_algorithms(False)

    assert len(result["history"]) == config.meta_steps + 1
    for record in result["history"]:
        assert torch.isfinite(torch.tensor(record["objective_ce"]))
        assert torch.isfinite(torch.tensor(record["meta_validation_ce"]))
    assert torch.isfinite(torch.tensor(result["history"][0]["gradient_norm"]))
    assert result["history"][0]["gradient_norm"] > 0.0  # metagradient is non-trivial
    multipliers = torch.tensor(result["selected_multipliers"])
    assert multipliers.numel() == n_pool
    assert torch.all(multipliers > 0)
    assert torch.isfinite(multipliers).all()
