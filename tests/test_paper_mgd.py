from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from _fixtures import assert_states_close, make_oracle_fixture
from functional_train import SmoothAdamWConfig, initialize_train_state
from metagrad import InnerBatch, ObjectiveBatch, train_unrolled
from paper_mgd import (
    CandidatePool,
    PaperMGDConfig,
    PaperProbeBatch,
    bernoulli_coordinates,
    build_count_trajectory,
    count_schedule_indices,
    fixed_budget_ranked_update,
    initialize_counts,
    make_probe_batch,
    paper_mgd_outer_step,
    paper_replay_metagradient,
    paper_replay_objective,
    paper_train_unrolled,
    paper_unrolled_objective,
    projected_sign_update,
    shard_coordinates,
)
from recursive_replay import ReplayCheckpointConfig


def oracle_pool() -> CandidatePool:
    fixture = make_oracle_fixture(steps=1)
    images = fixture.trajectory[0].images
    return CandidatePool(images, torch.tensor([10, 11, 12, 13]))


def oracle_paper_setup(steps: int = 4, counts: Tensor | None = None):
    fixture = make_oracle_fixture(steps=steps)
    pool = CandidatePool(
        fixture.trajectory[0].images, torch.tensor([10, 11, 12, 13])
    )
    counts = counts if counts is not None else initialize_counts(pool.num_candidates)
    trajectory = build_count_trajectory(
        pool,
        counts,
        inner_steps=steps,
        batch_size=4,
        num_patches=fixture.model.config.num_patches,
        mask_ratio=fixture.model.config.mask_ratio,
        mask_seed=9,
        shuffle_seed=17,
    )
    selected = torch.tensor([0, 2], dtype=torch.long)
    probe = make_probe_batch(
        pool,
        selected,
        perturbation_step=1,
        num_patches=fixture.model.config.num_patches,
        mask_ratio=fixture.model.config.mask_ratio,
        mask_seed=21,
    )
    return fixture, pool, counts, trajectory, selected, probe


def central_coordinate_gradient(objective, values: Tensor, step: float = 1e-4) -> Tensor:
    estimates = []
    for coordinate in range(values.numel()):
        offset = torch.zeros_like(values)
        offset[coordinate] = step
        estimates.append(
            (objective(values + offset) - objective(values - offset)) / (2.0 * step)
        )
    return torch.stack(estimates)


def test_count_schedule_is_deterministic_honors_counts_and_preserves_budget() -> None:
    counts = torch.tensor([2, 0, 1])
    first = count_schedule_indices(counts, sample_budget=8, shuffle_seed=7)
    second = count_schedule_indices(counts, sample_budget=8, shuffle_seed=7)

    assert torch.equal(first, second)
    assert first.shape == (8,)
    assert 1 not in first
    torch.testing.assert_close(
        torch.bincount(first[:3], minlength=3), counts
    )


def test_count_trajectory_has_stable_ids_masks_and_fixed_batch_budget() -> None:
    pool = oracle_pool()
    counts = torch.tensor([2, 0, 1, 1])
    first = build_count_trajectory(
        pool,
        counts,
        inner_steps=3,
        batch_size=2,
        num_patches=4,
        mask_ratio=0.5,
        mask_seed=5,
        shuffle_seed=6,
    )
    second = build_count_trajectory(
        pool,
        counts,
        inner_steps=3,
        batch_size=2,
        num_patches=4,
        mask_ratio=0.5,
        mask_seed=5,
        shuffle_seed=6,
    )

    assert len(first) == 3
    assert sum(batch.images.shape[0] for batch in first) == 6
    for actual, repeated in zip(first, second, strict=True):
        assert 1 not in actual.candidate_indices
        assert torch.equal(actual.candidate_indices, repeated.candidate_indices)
        assert torch.equal(actual.images, repeated.images)
        assert torch.equal(actual.patch_mask, repeated.patch_mask)


def test_zero_surrogate_exactly_matches_ordinary_unweighted_training() -> None:
    fixture, _, _, trajectory, selected, probe = oracle_paper_setup(steps=3)
    perturbations = torch.zeros(
        selected.numel(), dtype=torch.float64, requires_grad=True
    )
    paper_state = paper_train_unrolled(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        1,
        perturbations,
        fixture.optimizer_config,
        probe_chunk_size=1,
        create_graph=True,
    )
    ordinary_trajectory = tuple(
        InnerBatch(
            batch.images,
            batch.patch_mask,
            torch.zeros(batch.images.shape[0], dtype=torch.long),
        )
        for batch in trajectory
    )
    ordinary_state = train_unrolled(
        fixture.model,
        fixture.initial_state,
        ordinary_trajectory,
        torch.zeros(1, dtype=torch.float64, requires_grad=True),
        torch.ones(1, dtype=torch.float64),
        fixture.optimizer_config,
        create_graph=True,
    )

    assert_states_close(paper_state, ordinary_state, rtol=0.0, atol=0.0)


def test_only_selected_coordinates_receive_gradients_including_zero_count() -> None:
    fixture = make_oracle_fixture(steps=3)
    pool = CandidatePool(
        fixture.trajectory[0].images, torch.tensor([10, 11, 12, 13])
    )
    selected = shard_coordinates(pool.num_candidates, fraction=0.5, seed=4)
    counts = torch.ones(4, dtype=torch.long)
    counts[selected[0]] = 0
    config = PaperMGDConfig(
        inner_steps=3,
        batch_size=4,
        perturbation_step=1,
        update_policy="fixed_budget_ranked",
        coordinate_fraction=0.5,
        exchange_fraction=0.5,
        mask_seed=9,
        probe_mask_seed=21,
        shuffle_seed=17,
        selection_seed=4,
        branching_factor=2,
    )

    result = paper_mgd_outer_step(
        fixture.model,
        fixture.initial_state,
        pool,
        counts,
        fixture.objective_batch,
        fixture.optimizer_config,
        config,
    )

    assert torch.equal(result.selected_indices, selected)
    unselected = torch.ones(pool.num_candidates, dtype=torch.bool)
    unselected[selected] = False
    assert torch.equal(
        result.metagradient[unselected],
        torch.zeros_like(result.metagradient[unselected]),
    )
    assert torch.isfinite(result.metagradient[selected[0]])
    assert result.selected_metagradient.shape == selected.shape


def test_paper_unrolled_metagradient_matches_finite_differences() -> None:
    fixture, _, _, trajectory, selected, probe = oracle_paper_setup(steps=3)

    def objective(values: Tensor, create_graph: bool) -> Tensor:
        return paper_unrolled_objective(
            fixture.model,
            fixture.initial_state,
            trajectory,
            probe,
            fixture.objective_batch,
            1,
            values,
            fixture.optimizer_config,
            probe_chunk_size=1,
            create_graph=create_graph,
        )

    perturbations = torch.tensor(
        [-0.1, 0.15], dtype=torch.float64, requires_grad=True
    )
    objective_value = objective(perturbations, True)
    (metagradient,) = torch.autograd.grad(objective_value, perturbations)
    finite_difference = central_coordinate_gradient(
        lambda values: objective(values.requires_grad_(True), False),
        perturbations.detach(),
    )

    assert selected.numel() == metagradient.numel()
    torch.testing.assert_close(
        metagradient, finite_difference, rtol=1e-4, atol=1e-7
    )


@pytest.mark.parametrize("branching_factor", [2, 3, 8])
def test_paper_recursive_replay_matches_unrolled_reference(
    branching_factor: int,
) -> None:
    fixture, _, _, trajectory, selected, probe = oracle_paper_setup(steps=5)
    unrolled_z = torch.tensor(
        [-0.1, 0.15], dtype=torch.float64, requires_grad=True
    )
    unrolled_objective = paper_unrolled_objective(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        fixture.objective_batch,
        1,
        unrolled_z,
        fixture.optimizer_config,
        probe_chunk_size=1,
        create_graph=True,
    )
    (unrolled_gradient,) = torch.autograd.grad(unrolled_objective, unrolled_z)

    replay_z = unrolled_z.detach().clone().requires_grad_(True)
    replay_objective, replay_gradient = paper_replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        fixture.objective_batch,
        1,
        replay_z,
        fixture.optimizer_config,
        probe_chunk_size=1,
        branching_factor=branching_factor,
    )

    assert selected.numel() == replay_gradient.numel()
    torch.testing.assert_close(
        replay_objective, unrolled_objective.detach(), rtol=1e-9, atol=1e-11
    )
    torch.testing.assert_close(
        replay_gradient, unrolled_gradient, rtol=1e-9, atol=1e-11
    )


def test_paper_disk_backed_replay_matches_memory_and_cleans_up(tmp_path) -> None:
    fixture, _, _, trajectory, _, probe = oracle_paper_setup(steps=5)
    perturbation_values = torch.tensor([-0.1, 0.15], dtype=torch.float64)

    memory_objective, memory_gradient = paper_replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        fixture.objective_batch,
        1,
        perturbation_values.clone().requires_grad_(True),
        fixture.optimizer_config,
        probe_chunk_size=1,
        branching_factor=3,
    )
    disk_objective, disk_gradient = paper_replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        fixture.objective_batch,
        1,
        perturbation_values.clone().requires_grad_(True),
        fixture.optimizer_config,
        probe_chunk_size=1,
        branching_factor=3,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=2),
    )

    torch.testing.assert_close(disk_objective, memory_objective, rtol=0.0, atol=0.0)
    torch.testing.assert_close(disk_gradient, memory_gradient, rtol=0.0, atol=0.0)
    assert list(tmp_path.iterdir()) == []


def test_paper_mgd_config_threads_disk_checkpoint_storage(tmp_path) -> None:
    fixture, pool, counts, _, _, _ = oracle_paper_setup(steps=2)
    config = PaperMGDConfig(
        inner_steps=2,
        batch_size=4,
        perturbation_step=0,
        branching_factor=2,
        coordinate_fraction=0.5,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=1),
    )

    result = paper_mgd_outer_step(
        fixture.model,
        fixture.initial_state,
        pool,
        counts,
        fixture.objective_batch,
        fixture.optimizer_config,
        config,
    )

    assert torch.isfinite(result.objective)
    assert list(tmp_path.iterdir()) == []


def test_projected_sign_update_steps_and_projects_counts() -> None:
    counts = torch.tensor([1, 0, 2, 1])
    selected = torch.arange(4)
    gradient = torch.tensor([2.0, -3.0, 0.0, 1.0])

    update = projected_sign_update(counts, selected, gradient)

    assert torch.equal(update.counts, torch.tensor([0, 1, 2, 0]))
    assert torch.equal(update.incremented_indices, torch.tensor([1]))
    assert torch.equal(update.decremented_indices, torch.tensor([0, 3]))


def test_fixed_budget_ranked_update_preserves_budget_and_revives_removed_sample() -> None:
    counts = torch.tensor([0, 1, 2, 1])
    selected = torch.arange(4)
    gradient = torch.tensor([-3.0, -2.0, 4.0, 3.0])

    update = fixed_budget_ranked_update(
        counts, selected, gradient, exchange_fraction=0.25
    )

    assert torch.equal(update.counts, torch.tensor([1, 1, 1, 1]))
    assert torch.equal(update.incremented_indices, torch.tensor([0]))
    assert torch.equal(update.decremented_indices, torch.tensor([2]))
    assert update.counts.sum() == counts.sum()


class ConstantPatchModel(nn.Module):
    """One-parameter reconstruction model for the count-shift sanity test."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_patches=1, mask_ratio=1.0)
        self.prediction = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, images: Tensor, patch_mask: Tensor) -> Tensor:
        return self.prediction.expand(images.shape[0], 1, 3 * 2 * 2)


def test_tiny_end_to_end_count_mgd_shifts_toward_useful_samples_and_lowers_loss() -> None:
    model = ConstantPatchModel()
    initial_state = initialize_train_state(model)
    useful = torch.ones(2, 3, 2, 2, dtype=torch.float64)
    distractors = torch.zeros(2, 3, 2, 2, dtype=torch.float64)
    pool = CandidatePool(
        torch.cat((useful, distractors)),
        torch.tensor([10, 11, 12, 13]),
    )
    objective_batch = ObjectiveBatch(
        torch.ones(2, 3, 2, 2, dtype=torch.float64),
        torch.ones(2, 1, dtype=torch.bool),
    )
    optimizer_config = SmoothAdamWConfig(
        learning_rate=0.1, betas=(0.8, 0.9), eps=0.1, weight_decay=0.0
    )
    config = PaperMGDConfig(
        inner_steps=3,
        batch_size=4,
        perturbation_step=1,
        update_policy="fixed_budget_ranked",
        coordinate_fraction=1.0,
        exchange_fraction=0.5,
        mask_seed=5,
        probe_mask_seed=6,
        shuffle_seed=7,
        selection_seed=8,
        branching_factor=2,
    )
    counts = initialize_counts(pool.num_candidates)

    first = paper_mgd_outer_step(
        model,
        initial_state,
        pool,
        counts,
        objective_batch,
        optimizer_config,
        config,
        outer_step=0,
    )
    second = paper_mgd_outer_step(
        model,
        initial_state,
        pool,
        first.updated_counts,
        objective_batch,
        optimizer_config,
        config,
        outer_step=1,
    )

    assert first.updated_counts[:2].sum() > counts[:2].sum()
    assert first.updated_counts[2:].sum() < counts[2:].sum()
    assert second.objective < first.objective


def test_invalid_paper_mgd_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="perturbation_step"):
        PaperMGDConfig(
            inner_steps=2,
            batch_size=2,
            perturbation_step=2,
            branching_factor=2,
        )
    with pytest.raises(ValueError, match="exchange_fraction"):
        PaperMGDConfig(
            inner_steps=2,
            batch_size=2,
            perturbation_step=1,
            branching_factor=2,
            exchange_fraction=0.6,
        )
    with pytest.raises(ValueError, match="empty active dataset"):
        count_schedule_indices(torch.zeros(3, dtype=torch.long), 4, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        count_schedule_indices(torch.tensor([1, -1]), 4, 0)
    with pytest.raises(ValueError, match="exchange_fraction"):
        fixed_budget_ranked_update(
            torch.ones(2, dtype=torch.long),
            torch.arange(2),
            torch.tensor([0.0, 1.0]),
            exchange_fraction=0.75,
        )
    fixture, _, _, trajectory, selected, probe = oracle_paper_setup(steps=2)
    with pytest.raises(ValueError, match="branching_factor"):
        paper_replay_objective(
            fixture.model,
            fixture.initial_state,
            trajectory,
            probe,
            fixture.objective_batch,
            0,
            torch.zeros(selected.numel(), dtype=torch.float64, requires_grad=True),
            fixture.optimizer_config,
            branching_factor=1,
        )


def test_empty_bernoulli_probe_is_a_valid_noop_metagradient() -> None:
    fixture, pool, counts, trajectory, _, _ = oracle_paper_setup(steps=2)
    selected = bernoulli_coordinates(pool.num_candidates, probability=0.0, seed=1)
    probe = PaperProbeBatch(
        pool.images[:0],
        torch.empty(
            0, fixture.model.config.num_patches, dtype=torch.bool
        ),
        selected,
    )
    perturbations = torch.zeros(0, dtype=torch.float64, requires_grad=True)

    objective, gradient = paper_replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        fixture.objective_batch,
        0,
        perturbations,
        fixture.optimizer_config,
        branching_factor=2,
    )
    ordinary_objective = paper_replay_objective(
        fixture.model,
        fixture.initial_state,
        trajectory,
        probe,
        fixture.objective_batch,
        0,
        perturbations,
        fixture.optimizer_config,
        branching_factor=2,
    )

    assert counts.sum() > 0
    assert torch.isfinite(objective)
    assert gradient.shape == (0,)
    torch.testing.assert_close(objective, ordinary_objective)
