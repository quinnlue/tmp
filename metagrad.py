from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn
from torch.func import functional_call

from functional_train import SmoothAdamWConfig, TrainState, weighted_inner_step
from model import cross_entropy_loss


@dataclass(frozen=True)
class InnerBatch:
    images: Tensor
    labels: Tensor
    group_ids: Tensor


@dataclass(frozen=True)
class ObjectiveBatch:
    images: Tensor
    labels: Tensor


def train_unrolled(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[InnerBatch],
    logits: Tensor,
    base_group_masses: Tensor,
    optimizer_config: SmoothAdamWConfig,
    temperature: float = 1.0,
    create_graph: bool = True,
) -> TrainState:
    """Store and differentiate through every state in a fixed inner trajectory."""
    if not trajectory:
        raise ValueError("trajectory must contain at least one inner batch")

    state = initial_state
    for batch in trajectory:
        state, _ = weighted_inner_step(
            model,
            state,
            batch.images,
            batch.labels,
            batch.group_ids,
            logits,
            base_group_masses,
            optimizer_config,
            temperature,
            create_graph,
        )
    return state


def classification_objective(
    model: nn.Module,
    state: TrainState,
    batch: ObjectiveBatch,
) -> Tensor:
    """Evaluate the unweighted held-out cross-entropy objective."""
    predictions = functional_call(
        model,
        (state.parameters, state.buffers),
        (batch.images,),
    )
    return cross_entropy_loss(predictions, batch.labels)


def unrolled_objective(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[InnerBatch],
    objective_batch: ObjectiveBatch,
    logits: Tensor,
    base_group_masses: Tensor,
    optimizer_config: SmoothAdamWConfig,
    temperature: float = 1.0,
    create_graph: bool = True,
) -> Tensor:
    """Compose store-all inner training with the held-out objective."""
    final_state = train_unrolled(
        model,
        initial_state,
        trajectory,
        logits,
        base_group_masses,
        optimizer_config,
        temperature,
        create_graph,
    )
    return classification_objective(model, final_state, objective_batch)
