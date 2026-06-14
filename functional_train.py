from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.func import functional_call

from model import per_example_cross_entropy_loss
from weighting import weighted_example_loss


TensorMap = dict[str, Tensor]


@dataclass(frozen=True)
class SmoothAdamWConfig:
    """Configuration for the differentiable smooth-AdamW inner optimizer."""

    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SmoothSGDConfig:
    """Configuration for differentiable momentum SGD."""

    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = True


InnerOptimizerConfig = SmoothAdamWConfig | SmoothSGDConfig


@dataclass(frozen=True)
class TrainState:
    parameters: TensorMap
    buffers: TensorMap
    first_moments: TensorMap
    second_moments: TensorMap
    step: int


def initialize_train_state(model: nn.Module) -> TrainState:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return TrainState(
        parameters=parameters,
        buffers=buffers,
        first_moments={name: torch.zeros_like(value) for name, value in parameters.items()},
        second_moments={name: torch.zeros_like(value) for name, value in parameters.items()},
        step=0,
    )


def functional_smooth_adamw(
    state: TrainState,
    gradients: Mapping[str, Tensor],
    config: SmoothAdamWConfig,
) -> TrainState:
    """Apply smooth AdamW without mutating the input state.

    The parameter recurrence is

        p_next = (1 - lr * weight_decay) * p
                 - lr * m_hat / sqrt(v_hat + eps**2)

    Keeping epsilon inside the square root makes the recurrence smooth at
    zero second moment for backward-over-backward differentiation. This is
    intentionally different from torch.optim.AdamW.
    """
    beta1, beta2 = config.betas
    next_step = state.step + 1
    next_parameters: TensorMap = {}
    next_first_moments: TensorMap = {}
    next_second_moments: TensorMap = {}

    for name, parameter in state.parameters.items():
        gradient = gradients[name]
        first_moment = beta1 * state.first_moments[name] + (1.0 - beta1) * gradient
        second_moment = (
            beta2 * state.second_moments[name] + (1.0 - beta2) * gradient.square()
        )
        first_unbiased = first_moment / (1.0 - beta1**next_step)
        second_unbiased = second_moment / (1.0 - beta2**next_step)
        update = first_unbiased / (second_unbiased + config.eps**2).sqrt()
        next_parameters[name] = parameter * (
            1.0 - config.learning_rate * config.weight_decay
        ) - config.learning_rate * update
        next_first_moments[name] = first_moment
        next_second_moments[name] = second_moment

    return TrainState(
        parameters=next_parameters,
        buffers=state.buffers,
        first_moments=next_first_moments,
        second_moments=next_second_moments,
        step=next_step,
    )


def functional_smooth_sgd(
    state: TrainState,
    gradients: Mapping[str, Tensor],
    config: SmoothSGDConfig,
) -> TrainState:
    if not 0.0 <= config.momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    next_parameters: TensorMap = {}
    next_first_moments: TensorMap = {}
    for name, parameter in state.parameters.items():
        gradient = gradients[name] + config.weight_decay * parameter
        velocity = config.momentum * state.first_moments[name] + gradient
        update = gradient + config.momentum * velocity if config.nesterov else velocity
        next_parameters[name] = parameter - config.learning_rate * update
        next_first_moments[name] = velocity

    return TrainState(
        parameters=next_parameters,
        buffers=state.buffers,
        first_moments=next_first_moments,
        second_moments=state.second_moments,
        step=state.step + 1,
    )


def weighted_inner_step(
    model: nn.Module,
    state: TrainState,
    images: Tensor,
    labels: Tensor,
    group_ids: Tensor,
    logits: Tensor,
    base_group_masses: Tensor,
    optimizer_config: InnerOptimizerConfig,
    temperature: float = 1.0,
    create_graph: bool = True,
) -> tuple[TrainState, Tensor]:
    predictions = functional_call(
        model, (state.parameters, state.buffers), (images,)
    )
    per_example_loss = per_example_cross_entropy_loss(predictions, labels)
    loss = weighted_example_loss(
        per_example_loss, logits, group_ids, base_group_masses, temperature
    )
    parameter_names = tuple(state.parameters)
    gradients = torch.autograd.grad(
        loss,
        tuple(state.parameters.values()),
        create_graph=create_graph,
    )
    gradient_map = dict(zip(parameter_names, gradients, strict=True))
    if isinstance(optimizer_config, SmoothSGDConfig):
        next_state = functional_smooth_sgd(state, gradient_map, optimizer_config)
    else:
        next_state = functional_smooth_adamw(state, gradient_map, optimizer_config)
    return next_state, loss
