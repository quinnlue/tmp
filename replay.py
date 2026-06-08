from __future__ import annotations

import threading
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from tqdm import tqdm

from functional_train import SmoothAdamWConfig, TrainState, weighted_inner_step
from metagrad import InnerBatch, ObjectiveBatch, reconstruction_objective
from recursive_replay import (
    ReplayCheckpointConfig,
    recursive_replay_state as _recursive_replay_state,
)


def recursive_replay_state(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[InnerBatch],
    logits: Tensor,
    base_group_masses: Tensor,
    optimizer_config: SmoothAdamWConfig,
    temperature: float = 1.0,
    *,
    branching_factor: int,
    progress_callback: Callable[[], None] | None = None,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> TrainState:
    """Run persistent-weight training with paper-faithful lazy k-ary REPLAY."""

    def step(
        state: TrainState,
        step_index: int,
        replay_logits: Tensor,
        create_graph: bool,
    ) -> TrainState:
        batch = trajectory[step_index]
        next_state, _ = weighted_inner_step(
            model,
            state,
            batch.images,
            batch.patch_mask,
            batch.group_ids,
            replay_logits,
            base_group_masses,
            optimizer_config,
            temperature,
            create_graph,
        )
        return next_state

    return _recursive_replay_state(
        initial_state,
        logits,
        len(trajectory),
        step,
        branching_factor=branching_factor,
        progress_callback=progress_callback,
        checkpoint_config=checkpoint_config,
    )


def replay_objective(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[InnerBatch],
    objective_batch: ObjectiveBatch,
    logits: Tensor,
    base_group_masses: Tensor,
    optimizer_config: SmoothAdamWConfig,
    temperature: float = 1.0,
    *,
    branching_factor: int,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> Tensor:
    """Compose recursive REPLAY training with the held-out objective."""
    final_state = recursive_replay_state(
        model,
        initial_state,
        trajectory,
        logits,
        base_group_masses,
        optimizer_config,
        temperature,
        branching_factor=branching_factor,
        checkpoint_config=checkpoint_config,
    )
    return reconstruction_objective(model, final_state, objective_batch)


def replay_metagradient(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[InnerBatch],
    objective_batch: ObjectiveBatch,
    logits: Tensor,
    base_group_masses: Tensor,
    optimizer_config: SmoothAdamWConfig,
    temperature: float = 1.0,
    *,
    branching_factor: int,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> tuple[Tensor, Tensor]:
    """Return the objective value and exact lazy-tree REPLAY metagradient."""
    if not logits.requires_grad:
        raise ValueError("logits must require grad to compute the metagradient")

    # The REPLAY backward (the second half of the bar) runs on a CUDA autograd
    # worker thread, and ipykernel does not stream that thread's stdout to the
    # notebook until the cell returns -- so a bar refreshed inline only shows up
    # once the whole metagradient is already done. Instead, the step callback just
    # bumps a counter, and a dedicated Python thread -- which inherits the cell's
    # output context -- redraws the bar live while the compute blocks elsewhere.
    completed = 0  # only the forward XOR the backward increments at once; no lock needed

    def update_progress() -> None:
        nonlocal completed
        completed += 1

    with tqdm(total=2 * len(trajectory), desc="REPLAY metagradient") as progress:
        finished = threading.Event()

        def refresh_bar() -> None:
            while not finished.wait(0.1):
                progress.n = completed
                progress.refresh()
            progress.n = completed
            progress.refresh()

        bar_thread = threading.Thread(
            target=refresh_bar, name="replay-pbar", daemon=True
        )
        bar_thread.start()
        try:
            final_state = recursive_replay_state(
                model,
                initial_state,
                trajectory,
                logits,
                base_group_masses,
                optimizer_config,
                temperature,
                branching_factor=branching_factor,
                progress_callback=update_progress,
                checkpoint_config=checkpoint_config,
            )
            objective = reconstruction_objective(model, final_state, objective_batch)
            (metagradient,) = torch.autograd.grad(objective, logits)
        finally:
            finished.set()
            bar_thread.join()
    return objective.detach(), metagradient
