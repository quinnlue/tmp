from __future__ import annotations

import math
from typing import Callable

import pytest
import torch
from torch import Tensor

from _fixtures import OracleFixture, assert_states_close, make_oracle_fixture
from functional_train import (
    SmoothAdamWConfig,
    TrainState,
    functional_smooth_adamw,
)
from metagrad import (
    InnerBatch,
    train_unrolled,
    unrolled_objective,
)
from weighting import weighted_example_loss


def objective_function(
    fixture: OracleFixture,
    group_ids: Tensor,
    base_group_masses: Tensor,
    create_graph: bool,
) -> Callable[[Tensor], Tensor]:
    trajectory = tuple(
        InnerBatch(batch.images, batch.patch_mask, group_ids)
        for batch in fixture.trajectory
    )

    def evaluate(logits: Tensor) -> Tensor:
        return unrolled_objective(
            fixture.model,
            fixture.initial_state,
            trajectory,
            fixture.objective_batch,
            logits,
            base_group_masses,
            fixture.optimizer_config,
            fixture.temperature,
            create_graph,
        )

    return evaluate


def richardson_coordinate_gradient(
    objective: Callable[[Tensor], Tensor],
    logits: Tensor,
    step_size: float = 1e-3,
) -> Tensor:
    estimates = []
    detached_logits = logits.detach()
    for coordinate in range(logits.numel()):
        offset = torch.zeros_like(detached_logits)
        offset[coordinate] = step_size
        coarse = (
            objective((detached_logits + offset).requires_grad_(True))
            - objective((detached_logits - offset).requires_grad_(True))
        ) / (2.0 * step_size)

        half_offset = offset / 2.0
        fine = (
            objective((detached_logits + half_offset).requires_grad_(True))
            - objective((detached_logits - half_offset).requires_grad_(True))
        ) / step_size
        estimates.append((4.0 * fine - coarse) / 3.0)
    return torch.stack(estimates)


def richardson_directional_derivative(
    objective: Callable[[Tensor], Tensor],
    logits: Tensor,
    direction: Tensor,
    step_size: float = 1e-3,
) -> Tensor:
    detached_logits = logits.detach()
    coarse = (
        objective((detached_logits + step_size * direction).requires_grad_(True))
        - objective((detached_logits - step_size * direction).requires_grad_(True))
    ) / (2.0 * step_size)
    half_step = step_size / 2.0
    fine = (
        objective((detached_logits + half_step * direction).requires_grad_(True))
        - objective((detached_logits - half_step * direction).requires_grad_(True))
    ) / (2.0 * half_step)
    return (4.0 * fine - coarse) / 3.0


def test_smooth_adamw_matches_independent_scalar_sensitivity_oracle() -> None:
    theta_value = 0.3
    temperature = 0.7
    targets = ((-1.0, 1.5), (0.25, -0.75), (1.25, -0.5))
    objective_target = 0.8
    config = SmoothAdamWConfig(
        learning_rate=0.07,
        betas=(0.6, 0.8),
        eps=0.2,
        weight_decay=0.1,
    )

    theta = torch.tensor(theta_value, dtype=torch.float64, requires_grad=True)
    parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    state = TrainState(
        parameters={"p": parameter},
        buffers={},
        first_moments={"p": torch.zeros_like(parameter)},
        second_moments={"p": torch.zeros_like(parameter)},
        step=0,
    )
    group_ids = torch.tensor([0, 1])
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)

    for first_target, second_target in targets:
        current_parameter = state.parameters["p"]
        losses = torch.stack(
            (
                (current_parameter - first_target).square(),
                (current_parameter - second_target).square(),
            )
        )
        logits = torch.stack((theta, theta.new_zeros(())))
        loss = weighted_example_loss(
            losses, logits, group_ids, base_group_masses, temperature
        )
        gradient = torch.autograd.grad(loss, current_parameter, create_graph=True)[0]
        state = functional_smooth_adamw(state, {"p": gradient}, config)

    objective = 0.5 * (state.parameters["p"] - objective_target).square()
    metagradient = torch.autograd.grad(objective, theta)[0]

    weight = 1.0 / (1.0 + math.exp(-theta_value / temperature))
    weight_sensitivity = weight * (1.0 - weight) / temperature
    p = 0.4
    dp = 0.0
    first_moment = 0.0
    d_first_moment = 0.0
    second_moment = 0.0
    d_second_moment = 0.0
    beta1, beta2 = config.betas

    for step, (first_target, second_target) in enumerate(targets, start=1):
        gradient = 2.0 * (
            weight * (p - first_target) + (1.0 - weight) * (p - second_target)
        )
        d_gradient = 2.0 * (
            dp + weight_sensitivity * (second_target - first_target)
        )
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        d_first_moment = (
            beta1 * d_first_moment + (1.0 - beta1) * d_gradient
        )
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        d_second_moment = (
            beta2 * d_second_moment
            + (1.0 - beta2) * 2.0 * gradient * d_gradient
        )
        first_unbiased = first_moment / (1.0 - beta1**step)
        d_first_unbiased = d_first_moment / (1.0 - beta1**step)
        second_unbiased = second_moment / (1.0 - beta2**step)
        d_second_unbiased = d_second_moment / (1.0 - beta2**step)
        denominator = math.sqrt(second_unbiased + config.eps**2)
        update = first_unbiased / denominator
        d_update = (
            d_first_unbiased / denominator
            - first_unbiased * d_second_unbiased / (2.0 * denominator**3)
        )
        decay = 1.0 - config.learning_rate * config.weight_decay
        p = decay * p - config.learning_rate * update
        dp = decay * dp - config.learning_rate * d_update

    expected_objective = 0.5 * (p - objective_target) ** 2
    expected_metagradient = (p - objective_target) * dp
    torch.testing.assert_close(
        state.parameters["p"].detach(),
        torch.tensor(p, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        state.first_moments["p"].detach(),
        torch.tensor(first_moment, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        state.second_moments["p"].detach(),
        torch.tensor(second_moment, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        objective.detach(),
        torch.tensor(expected_objective, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        metagradient,
        torch.tensor(expected_metagradient, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_unrolled_objective_passes_gradcheck_and_gradgradcheck() -> None:
    fixture = make_oracle_fixture(steps=2)
    group_ids = torch.tensor([0, 0, 1, 1])
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)
    objective = objective_function(fixture, group_ids, base_group_masses, True)
    logits = torch.tensor([-0.2, 0.3], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        objective, (logits,), eps=1e-5, rtol=1e-4, atol=1e-6
    )
    assert torch.autograd.gradgradcheck(
        objective, (logits,), eps=1e-5, rtol=1e-4, atol=1e-6
    )


@pytest.mark.parametrize("granularity", ["cluster", "sample"])
def test_unrolled_metagradient_matches_full_coordinate_finite_differences(
    granularity: str,
) -> None:
    fixture = make_oracle_fixture()
    if granularity == "cluster":
        group_ids = torch.tensor([0, 0, 1, 1])
        base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)
        logits = torch.tensor([-0.25, 0.35], dtype=torch.float64, requires_grad=True)
    else:
        group_ids = torch.arange(4)
        base_group_masses = torch.full((4,), 0.25, dtype=torch.float64)
        logits = torch.tensor(
            [-0.35, -0.1, 0.2, 0.4], dtype=torch.float64, requires_grad=True
        )

    differentiable_objective = objective_function(
        fixture, group_ids, base_group_masses, True
    )
    numerical_objective = objective_function(
        fixture, group_ids, base_group_masses, False
    )
    metagradient = torch.autograd.grad(differentiable_objective(logits), logits)[0]
    finite_difference = richardson_coordinate_gradient(numerical_objective, logits)

    torch.testing.assert_close(
        metagradient, finite_difference, rtol=1e-4, atol=1e-7
    )
    significant = metagradient.abs() > 1e-6
    assert torch.equal(
        metagradient[significant].sign(), finite_difference[significant].sign()
    )


def test_softmax_shift_and_directional_derivative_invariants() -> None:
    fixture = make_oracle_fixture()
    group_ids = torch.tensor([0, 0, 1, 1])
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)
    logits = torch.tensor([-0.25, 0.35], dtype=torch.float64, requires_grad=True)
    differentiable_objective = objective_function(
        fixture, group_ids, base_group_masses, True
    )
    numerical_objective = objective_function(
        fixture, group_ids, base_group_masses, False
    )

    objective = differentiable_objective(logits)
    shifted_objective = numerical_objective(
        (logits.detach() + 4.25).requires_grad_(True)
    )
    metagradient = torch.autograd.grad(objective, logits)[0]
    direction = torch.tensor([1.0, -1.0], dtype=torch.float64)
    direction /= direction.norm()
    finite_directional = richardson_directional_derivative(
        numerical_objective, logits, direction
    )

    torch.testing.assert_close(objective, shifted_objective, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        metagradient.sum(),
        torch.zeros((), dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        metagradient.dot(direction),
        finite_directional,
        rtol=1e-4,
        atol=1e-7,
    )


def test_tied_sample_logits_match_cluster_trajectory_and_metagradient() -> None:
    fixture = make_oracle_fixture()
    cluster_ids = torch.tensor([0, 0, 1, 1])
    sample_ids = torch.arange(4)
    cluster_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)
    sample_masses = torch.full((4,), 0.25, dtype=torch.float64)
    cluster_logits = torch.tensor(
        [-0.25, 0.35], dtype=torch.float64, requires_grad=True
    )
    sample_logits = (
        cluster_logits.detach()
        .repeat_interleave(2)
        .clone()
        .requires_grad_(True)
    )
    cluster_trajectory = tuple(
        InnerBatch(batch.images, batch.patch_mask, cluster_ids)
        for batch in fixture.trajectory
    )
    sample_trajectory = tuple(
        InnerBatch(batch.images, batch.patch_mask, sample_ids)
        for batch in fixture.trajectory
    )

    cluster_state = train_unrolled(
        fixture.model,
        fixture.initial_state,
        cluster_trajectory,
        cluster_logits,
        cluster_masses,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    sample_state = train_unrolled(
        fixture.model,
        fixture.initial_state,
        sample_trajectory,
        sample_logits,
        sample_masses,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    assert_states_close(cluster_state, sample_state, rtol=1e-12, atol=1e-12)

    cluster_objective = unrolled_objective(
        fixture.model,
        fixture.initial_state,
        cluster_trajectory,
        fixture.objective_batch,
        cluster_logits,
        cluster_masses,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    sample_objective = unrolled_objective(
        fixture.model,
        fixture.initial_state,
        sample_trajectory,
        fixture.objective_batch,
        sample_logits,
        sample_masses,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    cluster_gradient = torch.autograd.grad(cluster_objective, cluster_logits)[0]
    sample_gradient = torch.autograd.grad(sample_objective, sample_logits)[0]

    torch.testing.assert_close(
        cluster_objective, sample_objective, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        cluster_gradient,
        sample_gradient.reshape(2, 2).sum(dim=1),
        rtol=1e-10,
        atol=1e-12,
    )


def test_unrolled_training_is_deterministic_functional_and_graph_mode_invariant() -> None:
    fixture = make_oracle_fixture()
    group_ids = torch.tensor([0, 0, 1, 1])
    trajectory = tuple(
        InnerBatch(batch.images, batch.patch_mask, group_ids)
        for batch in fixture.trajectory
    )
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)
    logits = torch.tensor([-0.25, 0.35], dtype=torch.float64, requires_grad=True)
    initial_snapshot = TrainState(
        parameters={
            name: value.detach().clone()
            for name, value in fixture.initial_state.parameters.items()
        },
        buffers={
            name: value.detach().clone()
            for name, value in fixture.initial_state.buffers.items()
        },
        first_moments={
            name: value.detach().clone()
            for name, value in fixture.initial_state.first_moments.items()
        },
        second_moments={
            name: value.detach().clone()
            for name, value in fixture.initial_state.second_moments.items()
        },
        step=fixture.initial_state.step,
    )

    with_graph = train_unrolled(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base_group_masses,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    repeated = train_unrolled(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base_group_masses,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    without_graph = train_unrolled(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits.detach().requires_grad_(True),
        base_group_masses,
        fixture.optimizer_config,
        fixture.temperature,
        False,
    )

    assert with_graph.step == fixture.initial_state.step + len(trajectory)
    assert_states_close(with_graph, repeated)
    assert_states_close(with_graph, without_graph)
    assert_states_close(fixture.initial_state, initial_snapshot)


def test_train_unrolled_rejects_empty_trajectory() -> None:
    fixture = make_oracle_fixture(steps=1)
    with pytest.raises(ValueError, match="trajectory"):
        train_unrolled(
            fixture.model,
            fixture.initial_state,
            (),
            torch.zeros(2, dtype=torch.float64, requires_grad=True),
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            fixture.optimizer_config,
        )
