from __future__ import annotations

import torch
from torch import Tensor


def group_masses(logits: Tensor, temperature: float = 1.0) -> Tensor:
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty one-dimensional tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits / temperature, dim=0)


def example_multipliers(
    logits: Tensor,
    group_ids: Tensor,
    base_group_masses: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    if group_ids.ndim != 1 or group_ids.dtype != torch.long:
        raise ValueError("group_ids must be a one-dimensional torch.long tensor")
    if base_group_masses.shape != logits.shape:
        raise ValueError("base_group_masses must have the same shape as logits")
    if logits.device != group_ids.device or logits.device != base_group_masses.device:
        raise ValueError("logits, group_ids, and base_group_masses must share a device")
    if torch.any(base_group_masses <= 0):
        raise ValueError("base_group_masses must be positive")
    if not torch.isclose(
        base_group_masses.sum(),
        base_group_masses.new_tensor(1.0),
    ):
        raise ValueError("base_group_masses must sum to one")
    if group_ids.numel() and (
        torch.any(group_ids < 0) or torch.any(group_ids >= logits.numel())
    ):
        raise ValueError("group_ids contains an invalid group index")

    target_group_masses = group_masses(logits, temperature)
    return target_group_masses[group_ids] / base_group_masses[group_ids]


def weighted_example_loss(
    per_example_loss: Tensor,
    logits: Tensor,
    group_ids: Tensor,
    base_group_masses: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    if per_example_loss.ndim != 1 or per_example_loss.shape != group_ids.shape:
        raise ValueError("per_example_loss and group_ids must have the same 1D shape")
    if per_example_loss.device != logits.device:
        raise ValueError("per_example_loss and logits must share a device")

    multipliers = example_multipliers(
        logits, group_ids, base_group_masses, temperature
    )
    return (per_example_loss * multipliers).mean()
