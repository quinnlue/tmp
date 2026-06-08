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
from mae import MaeConfig, MaskedAutoencoderViT, make_patch_mask
from metagrad import InnerBatch, ObjectiveBatch


@dataclass(frozen=True)
class OracleFixture:
    model: MaskedAutoencoderViT
    initial_state: TrainState
    trajectory: tuple[InnerBatch, ...]
    objective_batch: ObjectiveBatch
    optimizer_config: SmoothAdamWConfig
    temperature: float


def make_oracle_fixture(steps: int = 8) -> OracleFixture:
    torch.manual_seed(123)
    config = MaeConfig(
        image_size=8,
        patch_size=4,
        encoder_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        decoder_dim=8,
        decoder_depth=1,
        decoder_heads=2,
        mlp_ratio=2.0,
        mask_ratio=0.5,
    )
    model = MaskedAutoencoderViT(config).double()
    initial_state = initialize_train_state(model)
    cluster_ids = torch.tensor([0, 0, 1, 1])
    training_images = torch.randn(
        4,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(100),
    )
    trajectory = []
    for step in range(steps):
        patch_mask = make_patch_mask(
            [10, 11, 12, 13],
            step=step,
            seed=9,
            num_patches=config.num_patches,
            mask_ratio=config.mask_ratio,
        )
        trajectory.append(InnerBatch(training_images, patch_mask, cluster_ids))

    objective_images = torch.randn(
        2,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(999),
    )
    objective_mask = make_patch_mask(
        [1000, 1001],
        step=0,
        seed=77,
        num_patches=config.num_patches,
        mask_ratio=config.mask_ratio,
    )
    return OracleFixture(
        model=model,
        initial_state=initial_state,
        trajectory=tuple(trajectory),
        objective_batch=ObjectiveBatch(objective_images, objective_mask),
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
        InnerBatch(batch.images, batch.patch_mask, group_ids)
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
