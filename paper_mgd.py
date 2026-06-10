"""Paper-faithful count-based metagradient dataset selection.

This module is intentionally separate from the repository's persistent
cluster/sample softmax-weighting path. It implements the count relaxation and
outer updates from Section 4.1 and Appendix C.2 of arXiv:2503.13751.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call

from functional_train import (
    SmoothAdamWConfig,
    TrainState,
    functional_smooth_adamw,
)
from model import per_example_cross_entropy_loss
from metagrad import ObjectiveBatch, classification_objective
from recursive_replay import (
    ReplayCheckpointConfig,
    recursive_replay_state as _recursive_replay_state,
)


UpdatePolicy = Literal["projected_sign", "fixed_budget_ranked"]


@dataclass(frozen=True)
class CandidatePool:
    """A fixed candidate pool addressed by stable, unique image IDs."""

    images: Tensor
    image_ids: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[0] == 0:
            raise ValueError("images must be a non-empty [candidate, C, H, W] tensor")
        if self.image_ids.ndim != 1 or self.image_ids.shape[0] != self.images.shape[0]:
            raise ValueError("image_ids must have one entry per candidate image")
        if self.image_ids.dtype != torch.long:
            raise ValueError("image_ids must have torch.long dtype")
        if torch.unique(self.image_ids.detach().cpu()).numel() != self.image_ids.numel():
            raise ValueError("image_ids must be unique")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.images.shape[0]:
            raise ValueError("labels must have one entry per candidate image")
        if self.labels.dtype != torch.long:
            raise ValueError("labels must have torch.long dtype")

    @property
    def num_candidates(self) -> int:
        return self.images.shape[0]


@dataclass(frozen=True)
class PaperInnerBatch:
    """One ordinary batch in the count-derived inner-training trajectory."""

    images: Tensor
    labels: Tensor
    candidate_indices: Tensor


@dataclass(frozen=True)
class PaperProbeBatch:
    """Original-pool candidates whose infinitesimal weights are probed at step k."""

    images: Tensor
    labels: Tensor
    candidate_indices: Tensor


@dataclass(frozen=True)
class CountUpdateResult:
    counts: Tensor
    incremented_indices: Tensor
    decremented_indices: Tensor


@dataclass(frozen=True)
class PaperMGDConfig:
    inner_steps: int
    batch_size: int
    perturbation_step: int
    branching_factor: int
    update_policy: UpdatePolicy = "projected_sign"
    coordinate_fraction: float = 1.0
    exchange_fraction: float = 0.1
    shuffle_seed: int = 2
    selection_seed: int = 3
    probe_chunk_size: int | None = None
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig()

    def __post_init__(self) -> None:
        if self.inner_steps <= 0:
            raise ValueError("inner_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.perturbation_step < self.inner_steps:
            raise ValueError("perturbation_step must index an inner training step")
        if self.update_policy not in ("projected_sign", "fixed_budget_ranked"):
            raise ValueError(f"unknown update_policy: {self.update_policy}")
        if not 0.0 <= self.coordinate_fraction <= 1.0:
            raise ValueError("coordinate_fraction must be between zero and one")
        if not 0.0 <= self.exchange_fraction <= 0.5:
            raise ValueError("exchange_fraction must be between zero and 0.5")
        if self.probe_chunk_size is not None and self.probe_chunk_size <= 0:
            raise ValueError("probe_chunk_size must be positive")
        if self.branching_factor < 2:
            raise ValueError("branching_factor must be at least 2")


@dataclass(frozen=True)
class PaperMGDStepResult:
    updated_counts: Tensor
    objective: Tensor
    metagradient: Tensor
    selected_indices: Tensor
    selected_metagradient: Tensor
    incremented_indices: Tensor
    decremented_indices: Tensor
    total_count_before: int
    total_count_after: int
    active_candidates_before: int
    active_candidates_after: int


def initialize_counts(num_candidates: int, *, device: torch.device | None = None) -> Tensor:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    return torch.ones(num_candidates, dtype=torch.long, device=device)


def _validate_counts(counts: Tensor, num_candidates: int | None = None) -> None:
    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")
    if counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("counts must have a signed integer dtype")
    if num_candidates is not None and counts.numel() != num_candidates:
        raise ValueError("counts must have one entry per candidate")
    if torch.any(counts < 0):
        raise ValueError("counts must be nonnegative")


def _require_active_dataset(counts: Tensor) -> None:
    _validate_counts(counts)
    if int(counts.sum().item()) == 0:
        raise ValueError("counts define an empty active dataset")


def count_schedule_indices(
    counts: Tensor,
    sample_budget: int,
    shuffle_seed: int,
) -> Tensor:
    """Expand, deterministically shuffle, then cycle/truncate counts to a budget."""
    _require_active_dataset(counts)
    if sample_budget <= 0:
        raise ValueError("sample_budget must be positive")

    cpu_counts = counts.detach().cpu().to(torch.long)
    expanded = torch.repeat_interleave(
        torch.arange(cpu_counts.numel(), dtype=torch.long), cpu_counts
    )
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    shuffled = expanded[torch.randperm(expanded.numel(), generator=generator)]
    repeats = math.ceil(sample_budget / shuffled.numel())
    return shuffled.repeat(repeats)[:sample_budget]


def build_count_trajectory(
    pool: CandidatePool,
    counts: Tensor,
    *,
    inner_steps: int,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[PaperInnerBatch, ...]:
    """Build the fixed-budget deterministic trajectory induced by integer counts."""
    _validate_counts(counts, pool.num_candidates)
    if inner_steps <= 0 or batch_size <= 0:
        raise ValueError("inner_steps and batch_size must be positive")

    schedule = count_schedule_indices(
        counts, inner_steps * batch_size, shuffle_seed
    ).reshape(inner_steps, batch_size)
    batches: list[PaperInnerBatch] = []
    for cpu_indices in schedule:
        image_indices = cpu_indices.to(pool.images.device)
        label_indices = cpu_indices.to(pool.labels.device)
        images = pool.images.index_select(0, image_indices)
        labels = pool.labels.index_select(0, label_indices)
        batches.append(PaperInnerBatch(images, labels, cpu_indices.clone()))
    return tuple(batches)


def make_probe_batch(
    pool: CandidatePool,
    selected_indices: Tensor,
) -> PaperProbeBatch:
    cpu_indices = _validate_selected_indices(selected_indices, pool.num_candidates)
    image_indices = cpu_indices.to(pool.images.device)
    label_indices = cpu_indices.to(pool.labels.device)
    images = pool.images.index_select(0, image_indices)
    labels = pool.labels.index_select(0, label_indices)
    return PaperProbeBatch(images, labels, cpu_indices)


def _validate_selected_indices(selected_indices: Tensor, size: int) -> Tensor:
    if selected_indices.ndim != 1 or selected_indices.dtype != torch.long:
        raise ValueError("selected_indices must be a one-dimensional torch.long tensor")
    cpu_indices = selected_indices.detach().cpu()
    if cpu_indices.numel() and (
        torch.any(cpu_indices < 0) or torch.any(cpu_indices >= size)
    ):
        raise ValueError("selected_indices contains an invalid candidate index")
    if torch.unique(cpu_indices).numel() != cpu_indices.numel():
        raise ValueError("selected_indices must not contain duplicates")
    return cpu_indices


def _validate_selected_gradient(
    counts: Tensor,
    selected_indices: Tensor,
    selected_gradient: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_counts(counts)
    cpu_indices = _validate_selected_indices(selected_indices, counts.numel())
    if selected_gradient.ndim != 1 or selected_gradient.numel() != cpu_indices.numel():
        raise ValueError("selected_gradient must have one entry per selected index")
    if not torch.isfinite(selected_gradient).all():
        raise ValueError("selected_gradient must be finite")
    return cpu_indices, selected_gradient.detach().cpu()


def bernoulli_coordinates(
    num_candidates: int,
    probability: float,
    seed: int,
) -> Tensor:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    mask = torch.rand(num_candidates, generator=generator) < probability
    return torch.where(mask)[0]


def shard_coordinates(
    num_candidates: int,
    fraction: float,
    seed: int,
) -> Tensor:
    """Select an exact random shard from the original pool, independent of counts."""
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    if fraction == 0.0:
        return torch.empty(0, dtype=torch.long)
    shard_size = min(num_candidates, max(1, math.ceil(fraction * num_candidates)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(num_candidates, generator=generator)[:shard_size]


def projected_sign_update(
    counts: Tensor,
    selected_indices: Tensor,
    selected_gradient: Tensor,
) -> CountUpdateResult:
    """Apply Algorithm 1's signed integer step and nonnegative projection."""
    cpu_indices, cpu_gradient = _validate_selected_gradient(
        counts, selected_indices, selected_gradient
    )
    updated = counts.clone()
    if cpu_indices.numel():
        device_indices = cpu_indices.to(updated.device)
        signed_step = torch.sign(cpu_gradient).to(
            dtype=updated.dtype, device=updated.device
        )
        updated.index_add_(0, device_indices, -signed_step)
        updated.clamp_(min=0)
    before = counts.detach().cpu()
    after = updated.detach().cpu()
    return CountUpdateResult(
        counts=updated,
        incremented_indices=torch.where(after > before)[0],
        decremented_indices=torch.where(after < before)[0],
    )


def fixed_budget_ranked_update(
    counts: Tensor,
    selected_indices: Tensor,
    selected_gradient: Tensor,
    exchange_fraction: float,
) -> CountUpdateResult:
    """Exchange low- and high-gradient selected counts while preserving the budget."""
    if not 0.0 <= exchange_fraction <= 0.5:
        raise ValueError("exchange_fraction must be between zero and 0.5")
    cpu_indices, cpu_gradient = _validate_selected_gradient(
        counts, selected_indices, selected_gradient
    )
    exchange_count = math.floor(cpu_indices.numel() * exchange_fraction)
    if exchange_count == 0:
        empty = torch.empty(0, dtype=torch.long)
        return CountUpdateResult(counts.clone(), empty, empty)

    low_order = torch.argsort(cpu_gradient, stable=True)
    tentative_up_local = low_order[:exchange_count]
    tentative_up = cpu_indices[tentative_up_local]
    up_set = set(tentative_up.tolist())
    cpu_counts = counts.detach().cpu()

    high_order = torch.argsort(cpu_gradient, descending=True, stable=True)
    eligible_down_local = [
        int(local)
        for local in high_order.tolist()
        if int(cpu_indices[local]) not in up_set
        and int(cpu_counts[cpu_indices[local]]) > 0
    ]
    actual_count = min(exchange_count, len(eligible_down_local))
    if actual_count == 0:
        empty = torch.empty(0, dtype=torch.long)
        return CountUpdateResult(counts.clone(), empty, empty)

    incremented = tentative_up[:actual_count]
    decremented = cpu_indices[
        torch.tensor(eligible_down_local[:actual_count], dtype=torch.long)
    ]
    updated = counts.clone()
    updated.index_add_(
        0,
        incremented.to(updated.device),
        torch.ones(actual_count, dtype=updated.dtype, device=updated.device),
    )
    updated.index_add_(
        0,
        decremented.to(updated.device),
        -torch.ones(actual_count, dtype=updated.dtype, device=updated.device),
    )
    if int(updated.sum().item()) != int(counts.sum().item()):
        raise RuntimeError("fixed-budget update changed the total count")
    return CountUpdateResult(updated, incremented, decremented)


def paper_inner_step(
    model: nn.Module,
    state: TrainState,
    batch: PaperInnerBatch,
    perturbations: Tensor,
    optimizer_config: SmoothAdamWConfig,
    *,
    probe_batch: PaperProbeBatch | None = None,
    probe_chunk_size: int | None = None,
    create_graph: bool = True,
) -> tuple[TrainState, Tensor]:
    """Take one normal step, optionally adding the paper's one-step surrogate."""
    if batch.images.shape[0] == 0:
        raise ValueError("inner batches must be non-empty")
    predictions = functional_call(
        model, (state.parameters, state.buffers), (batch.images,)
    )
    loss = per_example_cross_entropy_loss(predictions, batch.labels).mean()

    if probe_batch is not None:
        if perturbations.ndim != 1:
            raise ValueError("perturbations must be one-dimensional")
        if perturbations.numel() != probe_batch.candidate_indices.numel():
            raise ValueError("perturbations must have one entry per probe candidate")
        if perturbations.device != batch.images.device:
            raise ValueError("perturbations and training images must share a device")
        if probe_chunk_size is not None and probe_chunk_size <= 0:
            raise ValueError("probe_chunk_size must be positive")

        perturbation_loss = perturbations.sum() * 0.0
        chunk_size = probe_chunk_size or max(1, perturbations.numel())
        for start in range(0, perturbations.numel(), chunk_size):
            stop = min(start + chunk_size, perturbations.numel())
            probe_images = probe_batch.images[start:stop]
            probe_labels = probe_batch.labels[start:stop]
            probe_predictions = functional_call(
                model,
                (state.parameters, state.buffers),
                (probe_images,),
            )
            probe_losses = per_example_cross_entropy_loss(
                probe_predictions, probe_labels
            )
            perturbation_loss = perturbation_loss + (
                perturbations[start:stop] * probe_losses
            ).sum()

        # A unit perturbation has the scale of adding one example to the
        # existing mean-reduced batch loss.
        loss = loss + perturbation_loss / batch.images.shape[0]

    parameter_names = tuple(state.parameters)
    gradients = torch.autograd.grad(
        loss, tuple(state.parameters.values()), create_graph=create_graph
    )
    next_state = functional_smooth_adamw(
        state, dict(zip(parameter_names, gradients, strict=True)), optimizer_config
    )
    return next_state, loss


def paper_train_unrolled(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[PaperInnerBatch],
    probe_batch: PaperProbeBatch,
    perturbation_step: int,
    perturbations: Tensor,
    optimizer_config: SmoothAdamWConfig,
    *,
    probe_chunk_size: int | None = None,
    create_graph: bool = True,
) -> TrainState:
    if not trajectory:
        raise ValueError("trajectory must contain at least one inner batch")
    if not 0 <= perturbation_step < len(trajectory):
        raise ValueError("perturbation_step must index the trajectory")

    state = initial_state
    for step, batch in enumerate(trajectory):
        state, _ = paper_inner_step(
            model,
            state,
            batch,
            perturbations,
            optimizer_config,
            probe_batch=probe_batch if step == perturbation_step else None,
            probe_chunk_size=probe_chunk_size,
            create_graph=create_graph,
        )
    return state


def paper_unrolled_objective(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[PaperInnerBatch],
    probe_batch: PaperProbeBatch,
    objective_batch: ObjectiveBatch,
    perturbation_step: int,
    perturbations: Tensor,
    optimizer_config: SmoothAdamWConfig,
    *,
    probe_chunk_size: int | None = None,
    create_graph: bool = True,
) -> Tensor:
    final_state = paper_train_unrolled(
        model,
        initial_state,
        trajectory,
        probe_batch,
        perturbation_step,
        perturbations,
        optimizer_config,
        probe_chunk_size=probe_chunk_size,
        create_graph=create_graph,
    )
    return classification_objective(model, final_state, objective_batch)


def paper_recursive_replay_state(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[PaperInnerBatch],
    probe_batch: PaperProbeBatch,
    perturbation_step: int,
    perturbations: Tensor,
    optimizer_config: SmoothAdamWConfig,
    *,
    probe_chunk_size: int | None = None,
    branching_factor: int,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> TrainState:
    if not trajectory:
        raise ValueError("trajectory must contain at least one inner batch")
    if not 0 <= perturbation_step < len(trajectory):
        raise ValueError("perturbation_step must index the trajectory")

    def step(
        state: TrainState,
        step_index: int,
        replay_perturbations: Tensor,
        create_graph: bool,
    ) -> TrainState:
        next_state, _ = paper_inner_step(
            model,
            state,
            trajectory[step_index],
            replay_perturbations,
            optimizer_config,
            probe_batch=probe_batch if step_index == perturbation_step else None,
            probe_chunk_size=probe_chunk_size,
            create_graph=create_graph,
        )
        return next_state

    return _recursive_replay_state(
        initial_state,
        perturbations,
        len(trajectory),
        step,
        branching_factor=branching_factor,
        checkpoint_config=checkpoint_config,
    )


def paper_replay_objective(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[PaperInnerBatch],
    probe_batch: PaperProbeBatch,
    objective_batch: ObjectiveBatch,
    perturbation_step: int,
    perturbations: Tensor,
    optimizer_config: SmoothAdamWConfig,
    *,
    probe_chunk_size: int | None = None,
    branching_factor: int,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> Tensor:
    final_state = paper_recursive_replay_state(
        model,
        initial_state,
        trajectory,
        probe_batch,
        perturbation_step,
        perturbations,
        optimizer_config,
        probe_chunk_size=probe_chunk_size,
        branching_factor=branching_factor,
        checkpoint_config=checkpoint_config,
    )
    return classification_objective(model, final_state, objective_batch)


def paper_replay_metagradient(
    model: nn.Module,
    initial_state: TrainState,
    trajectory: Sequence[PaperInnerBatch],
    probe_batch: PaperProbeBatch,
    objective_batch: ObjectiveBatch,
    perturbation_step: int,
    perturbations: Tensor,
    optimizer_config: SmoothAdamWConfig,
    *,
    probe_chunk_size: int | None = None,
    branching_factor: int,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> tuple[Tensor, Tensor]:
    if not perturbations.requires_grad:
        raise ValueError("perturbations must require grad to compute the metagradient")
    objective = paper_replay_objective(
        model,
        initial_state,
        trajectory,
        probe_batch,
        objective_batch,
        perturbation_step,
        perturbations,
        optimizer_config,
        probe_chunk_size=probe_chunk_size,
        branching_factor=branching_factor,
        checkpoint_config=checkpoint_config,
    )
    (metagradient,) = torch.autograd.grad(
        objective, perturbations, allow_unused=True
    )
    if metagradient is None:
        metagradient = torch.zeros_like(perturbations)
    return objective.detach(), metagradient


def paper_mgd_outer_step(
    model: nn.Module,
    initial_state: TrainState,
    pool: CandidatePool,
    counts: Tensor,
    objective_batch: ObjectiveBatch,
    optimizer_config: SmoothAdamWConfig,
    config: PaperMGDConfig,
    *,
    outer_step: int = 0,
) -> PaperMGDStepResult:
    """Run one deterministic count-MGD step, restarting from initial_state."""
    _validate_counts(counts, pool.num_candidates)
    _require_active_dataset(counts)
    if outer_step < 0:
        raise ValueError("outer_step must be nonnegative")

    trajectory = build_count_trajectory(
        pool,
        counts,
        inner_steps=config.inner_steps,
        batch_size=config.batch_size,
        shuffle_seed=config.shuffle_seed + outer_step,
    )
    selection_seed = config.selection_seed + outer_step
    if config.update_policy == "projected_sign":
        selected_indices = bernoulli_coordinates(
            pool.num_candidates, config.coordinate_fraction, selection_seed
        )
    else:
        selected_indices = shard_coordinates(
            pool.num_candidates, config.coordinate_fraction, selection_seed
        )
    probe_batch = make_probe_batch(pool, selected_indices)
    perturbations = torch.zeros(
        selected_indices.numel(),
        dtype=pool.images.dtype,
        device=pool.images.device,
        requires_grad=True,
    )
    objective, selected_metagradient = paper_replay_metagradient(
        model,
        initial_state,
        trajectory,
        probe_batch,
        objective_batch,
        config.perturbation_step,
        perturbations,
        optimizer_config,
        probe_chunk_size=config.probe_chunk_size,
        branching_factor=config.branching_factor,
        checkpoint_config=config.checkpoint_config,
    )

    if config.update_policy == "projected_sign":
        update = projected_sign_update(
            counts, selected_indices, selected_metagradient
        )
    else:
        update = fixed_budget_ranked_update(
            counts,
            selected_indices,
            selected_metagradient,
            config.exchange_fraction,
        )
    if int(update.counts.sum().item()) == 0:
        raise ValueError("count update produced an empty active dataset")

    full_metagradient = torch.zeros(
        pool.num_candidates,
        dtype=selected_metagradient.dtype,
        device=selected_metagradient.device,
    )
    if selected_indices.numel():
        full_metagradient.index_copy_(
            0, selected_indices.to(full_metagradient.device), selected_metagradient
        )
    return PaperMGDStepResult(
        updated_counts=update.counts,
        objective=objective,
        metagradient=full_metagradient,
        selected_indices=selected_indices,
        selected_metagradient=selected_metagradient,
        incremented_indices=update.incremented_indices,
        decremented_indices=update.decremented_indices,
        total_count_before=int(counts.sum().item()),
        total_count_after=int(update.counts.sum().item()),
        active_candidates_before=int(torch.count_nonzero(counts).item()),
        active_candidates_after=int(torch.count_nonzero(update.counts).item()),
    )
