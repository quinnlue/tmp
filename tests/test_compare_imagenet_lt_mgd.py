from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import metasmooth as ms
from torch.func import functional_call
from torchvision.io import write_png

from clustering import (
    ClusterBasis,
    ClusterTierConfig,
    build_fixed_intermediate_cluster_basis,
    hierarchical_kmeans_cluster_basis,
    save_cluster_basis_artifacts,
)
from experiments.compare_imagenet_lt_mgd import (
    ClusterAlignment,
    Config,
    FeatureArtifact,
    MaterializedImageArtifact,
    ScaledLinearHead,
    aggregate_class_masses,
    align_cluster_basis_to_split,
    build_dataset_bundle,
    build_granularity_spec,
    build_training_model,
    cap_exposed_pool_indices,
    capped_class_counts,
    class_logits_to_fine_logits,
    class_mass_summary,
    fairness_checks,
    fixed_kl_signed_update,
    load_feature_artifact,
    load_dataset_plan,
    load_batch,
    materialize_image_artifact,
    inner_learning_rate_at_step,
    main,
    make_model_config,
    validate_config,
)
from functional_train import (
    SmoothAdamWConfig,
    SmoothSGDConfig,
    initialize_train_state,
    weighted_inner_step,
)
from imagenet_lt import ImageNetLTEntry
from model import ViTConfig

pyarrow = pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq


def _entries_from_labels(tmp_path: Path, labels: list[int]) -> tuple[ImageNetLTEntry, ...]:
    entries: list[ImageNetLTEntry] = []
    for index, label in enumerate(labels):
        relative = f"class_{label}/image_{index}.png"
        entries.append(
            ImageNetLTEntry(
                relative_path=relative,
                label=label,
                absolute_path=tmp_path / relative,
            )
        )
    return tuple(entries)


def _aligned_clusters(tmp_path: Path) -> tuple[tuple[ImageNetLTEntry, ...], torch.Tensor, ClusterAlignment]:
    labels = [0, 0, 0, 1, 1, 2]
    entries = _entries_from_labels(tmp_path, labels)
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 0.8],
            [0.0, 0.7],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    basis = build_fixed_intermediate_cluster_basis(
        embeddings,
        torch.tensor(labels, dtype=torch.long),
        many_threshold=2,
        few_threshold=1,
        tier_config=ClusterTierConfig(many=2, medium=1, few=1),
    )
    train_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    aligned = align_cluster_basis_to_split(basis, entries, train_indices)
    return entries, torch.tensor(labels, dtype=torch.long), aligned


def _png_bytes(path: Path, value: int) -> bytes:
    image = torch.stack(
        (
            torch.full((4, 5), value, dtype=torch.uint8),
            torch.full((4, 5), value + 1, dtype=torch.uint8),
            torch.full((4, 5), value + 2, dtype=torch.uint8),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(image, str(path))
    return path.read_bytes()


def _write_hf_shard(path: Path, rows: list[dict[str, object]], *, row_group_size: int = 2) -> None:
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "image": pa.array([row["image"] for row in rows], type=image_type),
            "label": pa.array([row["label"] for row in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _make_synthetic_parquet_dataset(root: Path) -> Path:
    data_dir = root / "data"
    image_dir = root / "images"
    data_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    train_rows = [
        {"image": {"bytes": _png_bytes(image_dir / f"train_{index}.png", 10 + index), "path": f"images/train_{index}.png"}, "label": label}
        for index, label in enumerate([0, 0, 0, 1, 1, 1, 2, 2, 2])
    ]
    val_rows = [
        {"image": {"bytes": _png_bytes(image_dir / "val.png", 40), "path": "images/val.png"}, "label": 1}
    ]
    test_rows = [
        {"image": {"bytes": _png_bytes(image_dir / "test.png", 50), "path": "images/test.png"}, "label": 2}
    ]
    _write_hf_shard(data_dir / "train-00000-of-00001.parquet", train_rows, row_group_size=3)
    _write_hf_shard(data_dir / "val-00000-of-00001.parquet", val_rows, row_group_size=1)
    _write_hf_shard(data_dir / "test-00000-of-00001.parquet", test_rows, row_group_size=1)
    return root


def _write_feature_artifact(
    path: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    split: str,
) -> None:
    torch.save(
        {
            "metadata": {
                "target_examples": features.shape[0],
                "total_dataset_examples": features.shape[0],
                "split": split,
                "backbone": "synthetic",
            },
            "completed": features.shape[0],
            "embedding_dim": features.shape[1],
            "embeddings": features,
            "labels": labels,
            "global_indices": torch.arange(features.shape[0], dtype=torch.long),
        },
        path,
    )


def test_parameterization_uses_empirical_base_masses_for_all_granularities(tmp_path: Path) -> None:
    _, labels, aligned = _aligned_clusters(tmp_path)
    class_spec = build_granularity_spec("per_class", labels, num_classes=3)
    cluster_spec = build_granularity_spec(
        "per_cluster",
        labels,
        num_classes=3,
        aligned_clusters=aligned,
    )
    example_spec = build_granularity_spec("per_example", labels, num_classes=3)

    torch.testing.assert_close(
        class_spec.base_group_masses,
        torch.tensor([3.0, 2.0, 1.0]) / 6.0,
    )
    torch.testing.assert_close(
        aggregate_class_masses(cluster_spec.base_group_masses, cluster_spec.group_class_ids, 3),
        torch.tensor([3.0, 2.0, 1.0]) / 6.0,
    )
    torch.testing.assert_close(
        aggregate_class_masses(example_spec.base_group_masses, example_spec.group_class_ids, 3),
        torch.tensor([3.0, 2.0, 1.0]) / 6.0,
    )
    for spec in (class_spec, cluster_spec, example_spec):
        summary = class_mass_summary(torch.zeros(spec.num_groups), spec)
        torch.testing.assert_close(
            torch.tensor(summary["class_masses"]),
            torch.tensor([3.0, 2.0, 1.0]) / 6.0,
        )


def test_cluster_artifact_alignment_accepts_tensor_labels_and_preserves_class_masses(tmp_path: Path) -> None:
    _, labels, aligned = _aligned_clusters(tmp_path)
    assert aligned.subset_group_ids.shape == labels.shape
    torch.testing.assert_close(
        aggregate_class_masses(aligned.subset_base_group_masses, aligned.subset_group_class_ids, 3),
        torch.tensor([3.0, 2.0, 1.0]) / 6.0,
    )


def test_align_cluster_basis_supports_pool_aligned_source_indices() -> None:
    source_labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 2], dtype=torch.long)
    basis_source_indices = torch.tensor([5, 7, 3, 1], dtype=torch.long)
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    basis = build_fixed_intermediate_cluster_basis(
        embeddings,
        source_labels.index_select(0, basis_source_indices),
        many_threshold=10,
        few_threshold=1,
        tier_config=ClusterTierConfig(many=1, medium=1, few=1),
    )

    aligned = align_cluster_basis_to_split(
        basis,
        source_labels,
        torch.tensor([7, 1, 5, 3], dtype=torch.long),
        basis_source_indices=basis_source_indices,
    )

    assert aligned.original_group_ids.tolist() == [0, 1, 2]
    assert aligned.subset_group_ids.tolist() == [2, 0, 2, 1]
    assert aligned.subset_group_class_ids.tolist() == [0, 1, 2]
    torch.testing.assert_close(
        aligned.subset_base_group_masses,
        torch.tensor([0.25, 0.25, 0.5], dtype=torch.float32),
    )


def test_align_cluster_basis_rejects_duplicate_pool_aligned_source_indices() -> None:
    source_labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 2], dtype=torch.long)
    basis_source_indices = torch.tensor([5, 7, 3, 1], dtype=torch.long)
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    basis = build_fixed_intermediate_cluster_basis(
        embeddings,
        source_labels.index_select(0, basis_source_indices),
        many_threshold=10,
        few_threshold=1,
        tier_config=ClusterTierConfig(many=1, medium=1, few=1),
    )

    with pytest.raises(ValueError, match="exactly once"):
        align_cluster_basis_to_split(
            basis,
            source_labels,
            torch.tensor([7, 1, 5, 3], dtype=torch.long),
            basis_source_indices=torch.tensor([5, 7, 3, 5], dtype=torch.long),
        )


def test_align_label_agnostic_cluster_basis_accepts_mixed_labels_and_uses_majority() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    basis = ClusterBasis(
        group_ids=torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.long),
        base_group_masses=torch.tensor([0.5, 0.5], dtype=torch.float32),
        cluster_sizes=torch.tensor([3, 3], dtype=torch.long),
        cluster_centers=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )

    aligned = align_cluster_basis_to_split(
        basis,
        labels,
        torch.arange(labels.numel(), dtype=torch.long),
    )
    spec = build_granularity_spec(
        "per_cluster",
        labels,
        num_classes=2,
        aligned_clusters=aligned,
    )
    checks = fairness_checks(
        {
            "per_class": build_granularity_spec("per_class", labels, num_classes=2),
            "per_cluster": spec,
            "per_example": build_granularity_spec("per_example", labels, num_classes=2),
        }
    )

    assert aligned.label_pure is False
    assert aligned.subset_group_class_ids.tolist() == [0, 1]
    assert checks["comparisons"]["per_cluster"]["checked"] is False
    assert checks["comparisons"]["per_example"]["max_abs_diff"] <= 1e-6
    assert checks["passed"] is True


def test_tied_fine_to_coarse_logits_match_per_class_masses(tmp_path: Path) -> None:
    _, labels, aligned = _aligned_clusters(tmp_path)
    specs = {
        "per_class": build_granularity_spec("per_class", labels, num_classes=3),
        "per_cluster": build_granularity_spec(
            "per_cluster",
            labels,
            num_classes=3,
            aligned_clusters=aligned,
        ),
        "per_example": build_granularity_spec("per_example", labels, num_classes=3),
    }
    class_logits = torch.tensor([-0.8, 0.3, 0.7], dtype=torch.float32)
    class_summary = torch.tensor(class_mass_summary(class_logits, specs["per_class"])["class_masses"])
    for name in ("per_cluster", "per_example"):
        fine_logits = class_logits_to_fine_logits(class_logits, specs[name])
        fine_summary = torch.tensor(class_mass_summary(fine_logits, specs[name])["class_masses"])
        torch.testing.assert_close(fine_summary, class_summary)
    checks = fairness_checks(specs)  # type: ignore[arg-type]
    assert checks["checked"] is True
    assert checks["passed"] is True


def test_fixed_kl_signed_update_hits_requested_kl() -> None:
    logits = torch.zeros(5, dtype=torch.float32, requires_grad=True)
    gradient = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    spec = build_granularity_spec("per_example", labels, num_classes=3)
    update = fixed_kl_signed_update(
        logits,
        gradient,
        spec,
        temperature=1.0,
        target_kl=2.5e-3,
    )

    assert abs(update["distribution_kl"] - 2.5e-3) < 1e-6
    assert update["logit_scale"] > 0.0


def test_validate_config_rejects_missing_cluster_artifact() -> None:
    config = Config(
        train_manifest="train.txt",
        output="out.json",
        cluster_artifact_prefix=None,
        granularities=("per_class", "per_cluster"),
    )
    with pytest.raises(ValueError, match="cluster_artifact_prefix is required"):
        validate_config(config)


def test_validate_config_rejects_dual_source_modes() -> None:
    config = Config(
        train_manifest="train.txt",
        parquet_root="hf-root",
        output="out.json",
        granularities=("per_class",),
    )
    with pytest.raises(ValueError, match="exactly one train source"):
        validate_config(config)


def test_align_cluster_basis_rejects_misaligned_metadata(tmp_path: Path) -> None:
    entries, labels, _ = _aligned_clusters(tmp_path)
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 0.8],
            [0.0, 0.7],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    basis = build_fixed_intermediate_cluster_basis(
        embeddings,
        labels,
        many_threshold=2,
        few_threshold=1,
        tier_config=ClusterTierConfig(many=2, medium=1, few=1),
    )
    broken = type(basis)(
        group_ids=basis.group_ids,
        base_group_masses=basis.base_group_masses,
        cluster_sizes=basis.cluster_sizes,
        cluster_centers=basis.cluster_centers,
        class_to_cluster_ids={0: (999,), 1: basis.class_to_cluster_ids[1], 2: basis.class_to_cluster_ids[2]},
        class_counts=basis.class_counts,
        clusters_per_class=basis.clusters_per_class,
        many_classes=basis.many_classes,
        medium_classes=basis.medium_classes,
        few_classes=basis.few_classes,
        many_threshold=basis.many_threshold,
        few_threshold=basis.few_threshold,
        tier_config=basis.tier_config,
        inertia=basis.inertia,
    )
    with pytest.raises(ValueError, match="cluster artifact metadata mismatch"):
        align_cluster_basis_to_split(broken, entries, torch.arange(len(entries), dtype=torch.long))


def test_model_config_inherits_vit_defaults_and_allows_pool_override() -> None:
    inherited = make_model_config(Config(train_manifest="train.txt", output="out.json"), num_classes=5)
    defaults = ViTConfig()
    assert inherited.pre_norm == defaults.pre_norm
    assert inherited.pool == defaults.pool

    overridden = make_model_config(
        Config(train_manifest="train.txt", output="out.json", pre_norm=False, pool="cls", model_profile="smoke"),
        num_classes=5,
    )
    assert overridden.pre_norm is False
    assert overridden.pool == "cls"
    assert overridden.encoder_depth == 1


def test_capped_class_counts_preserve_distribution_and_cover_classes() -> None:
    counts = torch.tensor([5, 3, 2], dtype=torch.long)
    capped = capped_class_counts(counts, 6)
    assert capped.tolist() == [3, 2, 1]

    with pytest.raises(ValueError, match="too small"):
        capped_class_counts(counts, 2)


def test_capped_class_counts_preserves_every_class_near_minimum_budget() -> None:
    counts = torch.arange(1, 1001, dtype=torch.long)

    capped = capped_class_counts(counts, 1024)

    assert int(capped.sum()) == 1024
    assert torch.all(capped >= 1)
    assert torch.all(capped <= counts)


def test_cap_exposed_pool_indices_uses_only_exposed_examples() -> None:
    train_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    train_indices = torch.tensor([6, 0, 3, 7, 1, 4, 8, 2, 5], dtype=torch.long)
    capped = cap_exposed_pool_indices(train_indices, train_labels, max_train_examples=6)

    assert capped.tolist() == [6, 0, 3, 7, 1, 4]
    torch.testing.assert_close(
        torch.bincount(train_labels.index_select(0, capped), minlength=3),
        torch.tensor([2, 2, 2]),
    )


def test_load_batch_reads_in_storage_order_and_restores_requested_order() -> None:
    class RecordingDataset:
        def __init__(self) -> None:
            self.accessed: list[int] = []

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            self.accessed.append(index)
            return torch.full((3, 2, 2), index, dtype=torch.uint8), index

    dataset = RecordingDataset()
    images, labels = load_batch(
        dataset,  # type: ignore[arg-type]
        torch.tensor([4, 1, 3], dtype=torch.long),
        image_size=2,
        device=torch.device("cpu"),
    )

    assert dataset.accessed == [1, 3, 4]
    assert labels.tolist() == [4, 1, 3]
    assert images[:, 0, 0, 0].tolist() == pytest.approx(
        [
            (4.0 / 255.0 - 0.485) / 0.229,
            (1.0 / 255.0 - 0.485) / 0.229,
            (3.0 / 255.0 - 0.485) / 0.229,
        ]
    )


def test_feature_artifact_batch_is_gpu_ready_and_preserves_metagradients(tmp_path: Path) -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    artifact_path = tmp_path / "features.pt"
    torch.save(
        {
            "metadata": {"target_examples": 4, "split": "train", "backbone": "synthetic"},
            "completed": 4,
            "embedding_dim": 2,
            "embeddings": features,
            "labels": labels,
            "global_indices": torch.arange(4, dtype=torch.long),
        },
        artifact_path,
    )
    artifact = load_feature_artifact(
        artifact_path,
        expected_labels=labels,
        device=torch.device("cpu"),
        expected_split="train",
    )
    assert isinstance(artifact, FeatureArtifact)
    batch_features, batch_labels = load_batch(
        artifact,
        torch.tensor([3, 0, 2], dtype=torch.long),
        image_size=224,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(batch_features, features[[3, 0, 2]])
    torch.testing.assert_close(batch_labels, labels[[3, 0, 2]])

    torch.manual_seed(0)
    model = ScaledLinearHead(2, 2, final_logit_scale=2.0)
    state = initialize_train_state(model)
    group_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    weight_logits = torch.zeros(3, requires_grad=True)
    next_state, _ = weighted_inner_step(
        model,
        state,
        batch_features,
        batch_labels,
        group_ids,
        weight_logits,
        torch.full((3,), 1.0 / 3.0),
        SmoothAdamWConfig(learning_rate=1e-2),
    )
    predictions = functional_call(model, (next_state.parameters, next_state.buffers), (batch_features,))
    objective = torch.nn.functional.cross_entropy(predictions, batch_labels)
    gradient = torch.autograd.grad(objective, weight_logits)[0]

    assert state.buffers == {}
    assert next_state.buffers == {}
    assert torch.isfinite(gradient).all()
    assert not torch.equal(gradient, torch.zeros_like(gradient))


def test_resnet9_backend_builds_full_smooth_groupnorm_model() -> None:
    config = Config(
        train_manifest="train.txt",
        output="out.json",
        model_backend="resnet9",
        image_size=64,
    )

    model, image_size, name = build_training_model(config, num_classes=1000, feature_dim=None)

    assert name == "resnet9_smooth_gn"
    assert image_size == 64
    assert model.routine == ms.SMOOTH_GN_ROUTINE
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_081_000


def test_resnet9_backend_can_build_exact_smooth_batchnorm_model() -> None:
    config = Config(
        train_manifest="train.txt",
        output="out.json",
        model_backend="resnet9",
        resnet_normalization="frozen_bn",
        image_size=128,
    )

    model, image_size, name = build_training_model(config, num_classes=1000, feature_dim=None)

    assert name == "resnet9_smooth_frozen_bn"
    assert image_size == 128
    assert model.routine == ms.SMOOTH_ROUTINE


def test_full_resnet_inner_step_updates_convolution_and_head_and_has_metagradient() -> None:
    torch.manual_seed(4)
    model = ms.ResNet9(ms.SMOOTH_ROUTINE, num_classes=3).eval()
    state = initialize_train_state(model)
    images = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0, 2], dtype=torch.long)
    group_ids = labels.clone()
    weight_logits = torch.zeros(3, requires_grad=True)
    next_state, _ = weighted_inner_step(
        model,
        state,
        images,
        labels,
        group_ids,
        weight_logits,
        torch.full((3,), 1.0 / 3.0),
        SmoothSGDConfig(learning_rate=1e-3, momentum=0.9),
    )
    predictions = functional_call(model, (next_state.parameters, next_state.buffers), (images,))
    meta_gradient = torch.autograd.grad(
        torch.nn.functional.cross_entropy(predictions, labels),
        weight_logits,
    )[0]

    assert not torch.equal(next_state.parameters["conv1.conv.weight"], state.parameters["conv1.conv.weight"])
    assert not torch.equal(next_state.parameters["head.weight"], state.parameters["head.weight"])
    assert torch.isfinite(meta_gradient).all()
    assert not torch.equal(meta_gradient, torch.zeros_like(meta_gradient))


def test_materialized_image_artifact_preserves_global_index_lookup() -> None:
    class TinyDataset:
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.full((3, 3, 5), index, dtype=torch.uint8), index

    artifact = materialize_image_artifact(
        TinyDataset(),  # type: ignore[arg-type]
        torch.tensor([3, 1], dtype=torch.long),
        image_size=8,
        device=torch.device("cpu"),
    )
    images, labels = load_batch(
        artifact,
        torch.tensor([1, 3], dtype=torch.long),
        image_size=8,
        device=torch.device("cpu"),
    )

    assert isinstance(artifact, MaterializedImageArtifact)
    assert images.shape == (2, 3, 8, 8)
    assert labels.tolist() == [1, 3]


def test_inner_learning_rate_matches_warmup_cosine_schedule() -> None:
    values = [
        inner_learning_rate_at_step(
            step,
            total_steps=10,
            warmup_fraction=0.2,
            peak_learning_rate=1.0,
        )
        for step in range(10)
    ]

    assert values[:3] == pytest.approx([0.5, 1.0, 1.0])
    assert values[-1] < values[3]


def test_parquet_plan_supports_exposed_pool_and_inner_step_caps(tmp_path: Path) -> None:
    root = _make_synthetic_parquet_dataset(tmp_path)
    config = Config(
        parquet_root=str(root),
        output=str(tmp_path / "out.json"),
        mode="preflight",
        granularities=("per_class", "per_example"),
        objective_per_class=1,
        meta_validation_per_class=1,
        max_train_examples=3,
        inner_epochs=3,
        batch_size=2,
        max_inner_steps=2,
    )

    bundle = build_dataset_bundle(config)
    plan = load_dataset_plan(config, bundle)

    assert bundle.source_mode == "parquet"
    assert plan.available_splits == ("test", "train", "val")
    assert plan.exposed_train_labels.numel() == 3
    torch.testing.assert_close(
        torch.bincount(plan.exposed_train_labels, minlength=3),
        torch.tensor([1, 1, 1]),
    )
    assert len(plan.inner_schedule) == 2
    assert all(int(batch.max().item()) < plan.exposed_train_labels.numel() for batch in plan.inner_schedule)


def test_parquet_plan_accepts_explicit_exposed_pool_indices(tmp_path: Path) -> None:
    root = _make_synthetic_parquet_dataset(tmp_path)
    selected_path = tmp_path / "selected.pt"
    torch.save(torch.tensor([2, 4, 6], dtype=torch.long), selected_path)
    config = Config(
        parquet_root=str(root),
        output=str(tmp_path / "out.json"),
        exposed_train_indices_input=str(selected_path),
        mode="preflight",
        granularities=("per_class",),
        objective_per_class=0,
        meta_validation_per_class=0,
        batch_size=2,
    )

    plan = load_dataset_plan(config, build_dataset_bundle(config))

    assert plan.split_train_indices.tolist() == [2, 4, 6]
    assert plan.exposed_train_labels.tolist() == [0, 1, 2]


def test_main_preflight_writes_parquet_summary_with_val_and_test(tmp_path: Path) -> None:
    root = _make_synthetic_parquet_dataset(tmp_path / "hf")
    output = tmp_path / "preflight.json"
    indices_output = tmp_path / "exposed.indices.pt"
    config = Config(
        parquet_root=str(root),
        output=str(output),
        exposed_train_indices_output=str(indices_output),
        mode="preflight",
        granularities=("per_class", "per_example"),
        objective_per_class=1,
        meta_validation_per_class=1,
        max_train_examples=3,
        max_inner_steps=1,
    )

    bundle = build_dataset_bundle(config)
    plan = load_dataset_plan(config, bundle)
    main(config)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["design"]["source_mode"] == "parquet"
    assert payload["design"]["num_exposed_train_entries"] == 3
    assert payload["design"]["num_val_entries"] == 1
    assert payload["design"]["num_test_entries"] == 1
    assert payload["design"]["inner_steps"] == 1
    assert payload["design"]["exposed_train_indices_artifact"] == str(indices_output)
    assert payload["fairness_checks"]["passed"] is True
    torch.testing.assert_close(torch.load(indices_output), plan.split_train_indices)


def test_feature_backend_end_to_end_smoke_records_controls_and_loss_deltas(tmp_path: Path) -> None:
    root = _make_synthetic_parquet_dataset(tmp_path / "hf")
    output = tmp_path / "feature-smoke.json"
    train_features_path = tmp_path / "train-features.pt"
    val_features_path = tmp_path / "val-features.pt"
    test_features_path = tmp_path / "test-features.pt"
    cluster_prefix = tmp_path / "clusters" / "hkmeans"
    generator = torch.Generator().manual_seed(9)
    train_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    train_features = torch.randn(9, 4, generator=generator)
    _write_feature_artifact(train_features_path, train_features, train_labels, split="train")
    _write_feature_artifact(
        val_features_path,
        torch.randn(1, 4, generator=generator),
        torch.tensor([1], dtype=torch.long),
        split="val",
    )
    _write_feature_artifact(
        test_features_path,
        torch.randn(1, 4, generator=generator),
        torch.tensor([2], dtype=torch.long),
        split="test",
    )
    config = Config(
        parquet_root=str(root),
        output=str(output),
        cluster_artifact_prefix=str(cluster_prefix),
        features_artifact=str(train_features_path),
        val_features_artifact=str(val_features_path),
        test_features_artifact=str(test_features_path),
        mode="run",
        granularities=("per_class", "per_cluster", "per_example"),
        objective_per_class=1,
        meta_validation_per_class=1,
        inner_epochs=1,
        batch_size=3,
        max_inner_steps=1,
        eval_batch_size=3,
        inner_backward="store_all",
        meta_steps=1,
        inner_lr=1e-2,
    )
    bundle = build_dataset_bundle(config)
    plan = load_dataset_plan(config, bundle)
    exposed_features = train_features.index_select(0, plan.split_train_indices)
    basis = hierarchical_kmeans_cluster_basis(exposed_features, num_clusters=2, seed=2)
    save_cluster_basis_artifacts(basis, cluster_prefix)
    torch.save(plan.split_train_indices, cluster_prefix.with_suffix(".indices.pt"))

    main(config)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["design"]["training_backend"] == "features"
    assert set(payload["controls"]) == {"uniform", "class_prior_oracle"}
    assert set(payload["methods"]) == {"per_class", "per_cluster", "per_example"}
    for method in payload["methods"].values():
        assert len(method["history"]) == 2
        assert method["history"][0]["loss_deltas"]["objective_ce"] == pytest.approx(0.0)
        assert "meta_validation_balanced_ce" in method["history"][1]["loss_deltas"]
