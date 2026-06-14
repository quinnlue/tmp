from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from experiments.compare_mgd_granularity import (
    DEFAULT_TARGET,
    Split,
    apply_meta_update,
    class_mass_summary,
    exact_counts,
    oracle_logits,
    select_split_indices,
)


def fake_split(labels: torch.Tensor) -> Split:
    images = torch.zeros(labels.numel(), 3, 1, 1)
    empty_images = torch.zeros(0, 3, 1, 1)
    empty_labels = torch.zeros(0, dtype=torch.long)
    return Split(
        pool_images=images,
        pool_labels=labels,
        objective_images=empty_images,
        objective_labels=empty_labels,
        validation_images=empty_images,
        validation_labels=empty_labels,
        test_images=empty_images,
        test_labels=empty_labels,
        pool_source_indices=torch.arange(labels.numel()),
        objective_source_indices=torch.zeros(0, dtype=torch.long),
        validation_source_indices=torch.zeros(0, dtype=torch.long),
    )


def test_exact_counts_preserve_total_and_expected_default_skew() -> None:
    counts = exact_counts(2000, DEFAULT_TARGET)
    assert counts == (800, 400, 200, 200, 100, 100, 50, 50, 50, 50)
    assert sum(counts) == 2000


def test_split_is_balanced_skewed_and_disjoint() -> None:
    labels = np.repeat(np.arange(10), 100)
    objective_counts = (20, 10, 5, 5, 2, 2, 2, 2, 1, 1)
    validation_counts = tuple(reversed(objective_counts))
    pool, objective, validation = select_split_indices(
        labels,
        pool_per_class=25,
        objective_counts=objective_counts,
        validation_counts=validation_counts,
        seed=3,
    )
    np.testing.assert_array_equal(np.bincount(labels[pool], minlength=10), np.full(10, 25))
    np.testing.assert_array_equal(
        np.bincount(labels[objective], minlength=10), objective_counts
    )
    np.testing.assert_array_equal(
        np.bincount(labels[validation], minlength=10), validation_counts
    )
    assert not set(pool).intersection(objective)
    assert not set(pool).intersection(validation)
    assert not set(objective).intersection(validation)


def test_tied_per_example_logits_match_per_class_mass_summary() -> None:
    labels = torch.arange(10).repeat_interleave(4)
    split = fake_split(labels)
    class_logits = torch.linspace(-1.0, 1.0, 10)
    sample_logits = class_logits[labels]
    class_summary = class_mass_summary(class_logits, split, "per_class")
    sample_summary = class_mass_summary(sample_logits, split, "per_example")
    torch.testing.assert_close(
        torch.tensor(class_summary["class_masses"]),
        torch.tensor(sample_summary["class_masses"]),
    )
    torch.testing.assert_close(
        torch.tensor(class_summary["class_multipliers"]),
        torch.tensor(sample_summary["class_multipliers"]),
    )


def test_oracle_logits_recover_target_class_masses_for_both_granularities() -> None:
    labels = torch.arange(10).repeat_interleave(4)
    split = fake_split(labels)
    target = torch.tensor(DEFAULT_TARGET)
    for granularity in ("per_class", "per_example"):
        logits = oracle_logits(split, granularity, target)
        summary = class_mass_summary(logits, split, granularity)
        torch.testing.assert_close(
            torch.tensor(summary["class_masses"]), target, rtol=1e-6, atol=1e-7
        )


def test_sign_update_has_requested_distribution_kl() -> None:
    logits = torch.zeros(10, requires_grad=True)
    gradient = torch.linspace(-1.0, 1.0, 10)
    config = SimpleNamespace(temperature=1.0, step_kl=0.0025, meta_lr=0.05)
    update = apply_meta_update(logits, gradient, config, optimizer=None)
    assert abs(update["distribution_kl"] - config.step_kl) < 1e-6
