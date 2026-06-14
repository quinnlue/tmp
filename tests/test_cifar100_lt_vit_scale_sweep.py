from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.cifar100_lt.data import balanced_eval_split
from experiments.cifar100_lt.vit_scale_sweep import (
    PROFILES,
    build_model,
    frequency_tiers,
    learning_rate_at_step,
)


def test_balanced_eval_split_is_disjoint_and_class_balanced() -> None:
    labels = np.repeat(np.arange(4), 10)

    validation, test = balanced_eval_split(labels, validation_per_class=3, seed=4)

    assert not set(validation.tolist()).intersection(test.tolist())
    assert torch.bincount(torch.from_numpy(labels)[validation], minlength=4).tolist() == [3] * 4
    assert torch.bincount(torch.from_numpy(labels)[test], minlength=4).tolist() == [7] * 4


def test_corrected_smooth_profiles_build_tiny_through_base() -> None:
    parameter_counts = []
    for name in ("tiny", "small", "base"):
        model = build_model(PROFILES[name])
        assert model.config.pre_norm is True
        assert model.config.pool == "mean"
        assert model.config.smooth_activation is True
        assert model.config.final_logit_scale == 10.0
        parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))

    assert parameter_counts[0] < parameter_counts[1] < parameter_counts[2]


def test_frequency_tiers_use_long_tail_thresholds() -> None:
    tiers = frequency_tiers(torch.tensor([101, 100, 20, 19]))

    assert tiers["many"].tolist() == [True, False, False, False]
    assert tiers["medium"].tolist() == [False, True, True, False]
    assert tiers["few"].tolist() == [False, False, False, True]


def test_learning_rate_warms_up_and_cosine_decays() -> None:
    values = [
        learning_rate_at_step(
            step,
            total_steps=10,
            warmup_fraction=0.2,
            peak_learning_rate=1e-3,
        )
        for step in range(10)
    ]

    assert values[:3] == pytest.approx([5e-4, 1e-3, 1e-3])
    assert values[-1] < values[3]
