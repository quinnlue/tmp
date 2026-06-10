"""Shared oracle fixtures for the metagradient / REPLAY verification suites.

Not a test module (the name does not match ``test_*``), so pytest does not
collect it. Imported by both ``test_metagrad.py`` (L1/L2) and
``test_replay.py`` (L3/GATE) so the CPU/float64 oracle configuration lives in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from functional_train import (
    SmoothAdamWConfig,
    TrainState,
    initialize_train_state,
)
from model import ViTConfig, VisionTransformerClassifier
from metagrad import InnerBatch, ObjectiveBatch


@dataclass(frozen=True)
class OracleFixture:
    model: VisionTransformerClassifier
    initial_state: TrainState
    trajectory: tuple[InnerBatch, ...]
    objective_batch: ObjectiveBatch
    optimizer_config: SmoothAdamWConfig
    temperature: float


def make_oracle_fixture(steps: int = 8) -> OracleFixture:
    torch.manual_seed(123)
    config = ViTConfig(
        image_size=8,
        patch_size=4,
        encoder_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        mlp_ratio=2.0,
        num_classes=2,
    )
    model = VisionTransformerClassifier(config).double()
    initial_state = initialize_train_state(model)
    cluster_ids = torch.tensor([0, 0, 1, 1])
    # Clusters coincide with class labels in the cluster-basis experiment.
    training_labels = torch.tensor([0, 0, 1, 1])
    training_images = torch.randn(
        4,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(100),
    )
    trajectory = [
        InnerBatch(training_images, training_labels, cluster_ids)
        for _ in range(steps)
    ]

    objective_images = torch.randn(
        2,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(999),
    )
    # The held-out objective scores the target class c* = 0.
    objective_labels = torch.tensor([0, 0])
    return OracleFixture(
        model=model,
        initial_state=initial_state,
        trajectory=tuple(trajectory),
        objective_batch=ObjectiveBatch(objective_images, objective_labels),
        optimizer_config=SmoothAdamWConfig(
            learning_rate=2e-3,
            betas=(0.8, 0.9),
            eps=1e-4,
            weight_decay=0.03,
        ),
        temperature=0.7,
    )


def retarget_trajectory(
    fixture: OracleFixture, group_ids: Tensor
) -> tuple[InnerBatch, ...]:
    """Rebuild the trajectory with a different group-id assignment."""
    return tuple(
        InnerBatch(batch.images, batch.labels, group_ids)
        for batch in fixture.trajectory
    )


def assert_states_close(
    actual: TrainState,
    expected: TrainState,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    assert actual.step == expected.step
    for actual_map, expected_map in (
        (actual.parameters, expected.parameters),
        (actual.buffers, expected.buffers),
        (actual.first_moments, expected.first_moments),
        (actual.second_moments, expected.second_moments),
    ):
        assert actual_map.keys() == expected_map.keys()
        for name in actual_map:
            torch.testing.assert_close(
                actual_map[name], expected_map[name], rtol=rtol, atol=atol
            )
