from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call

from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from model import ViTConfig, VisionTransformerClassifier, cross_entropy_loss


class BenchmarkError(RuntimeError):
    """Raised when the benchmark encounters a non-finite or unsupported state."""


@dataclass(frozen=True)
class SyntheticBatch:
    images: Tensor
    labels: Tensor
    group_ids: Tensor
    base_group_masses: Tensor
    logits: Tensor


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 2
    image_size: int = 224
    patch_size: int = 16
    iterations: int = 1
    device: str = "auto"
    seed: int = 0
    final_logit_scale: float = 10.0
    json_output: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.final_logit_scale <= 0:
            raise ValueError("final_logit_scale must be positive")


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Focused ViT-S backbone GPU readiness benchmark."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final-logit-scale", type=float, default=10.0)
    parser.add_argument("--json-output", type=str, default=None)
    arguments = parser.parse_args()
    return BenchmarkConfig(
        batch_size=arguments.batch_size,
        image_size=arguments.image_size,
        patch_size=arguments.patch_size,
        iterations=arguments.iterations,
        device=arguments.device,
        seed=arguments.seed,
        final_logit_scale=arguments.final_logit_scale,
        json_output=arguments.json_output,
    )


def resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA was requested but is unavailable")
    return resolved


def configure_benchmark_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_vit_small_config(
    *, image_size: int, patch_size: int, final_logit_scale: float
) -> ViTConfig:
    return ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        final_logit_scale=final_logit_scale,
    )


def count_parameters(model: VisionTransformerClassifier) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def ensure_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise BenchmarkError(f"non-finite tensor detected for {name}")


def make_synthetic_batch(
    *,
    batch_size: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> SyntheticBatch:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    labels = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    # Keep a real two-group weighting direction even for batch_size=1. A
    # single-logit softmax is constant and would make the metagradient
    # identically zero, masking second-order readiness failures.
    group_ids = torch.arange(batch_size, dtype=torch.long).remainder(2).to(device)
    base_group_masses = torch.full(
        (2,),
        0.5,
        dtype=images.dtype,
    ).to(device)
    logits = torch.zeros(2, dtype=images.dtype, device=device, requires_grad=True)
    return SyntheticBatch(
        images=images,
        labels=labels,
        group_ids=group_ids,
        base_group_masses=base_group_masses,
        logits=logits,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_ms(device: torch.device, fn: Any) -> tuple[float, Any]:
    _synchronize(device)
    start = time.perf_counter()
    result = fn()
    _synchronize(device)
    return (time.perf_counter() - start) * 1_000.0, result


def summarize_series(values: list[float]) -> dict[str, float]:
    series = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_ms": float(series.mean().item()),
        "min_ms": float(series.min().item()),
        "max_ms": float(series.max().item()),
    }


def _peak_memory_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def run_benchmark_iteration(
    *,
    model: VisionTransformerClassifier,
    optimizer_config: SmoothAdamWConfig,
    inner_batch: SyntheticBatch,
    objective_batch: SyntheticBatch,
) -> dict[str, float | bool]:
    device = inner_batch.images.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)

    def forward_pass() -> tuple[Tensor, Tensor]:
        logits = model(inner_batch.images)
        loss = cross_entropy_loss(logits, inner_batch.labels)
        return logits, loss

    forward_ms, (forward_logits, forward_loss) = _measure_ms(device, forward_pass)
    ensure_finite("forward_logits", forward_logits)
    ensure_finite("forward_loss", forward_loss)

    backward_ms, _ = _measure_ms(device, forward_loss.backward)
    gradient_square_sum = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise BenchmarkError(f"missing gradient for parameter {name}")
        ensure_finite(f"grad:{name}", parameter.grad)
        gradient_square_sum += float(parameter.grad.square().sum().item())

    state = initialize_train_state(model)

    def inner_step() -> tuple[Any, Tensor]:
        return weighted_inner_step(
            model,
            state,
            inner_batch.images,
            inner_batch.labels,
            inner_batch.group_ids,
            inner_batch.logits,
            inner_batch.base_group_masses,
            optimizer_config,
            create_graph=True,
        )

    inner_step_ms, (next_state, inner_loss) = _measure_ms(device, inner_step)
    ensure_finite("inner_loss", inner_loss)

    def metagrad_pass() -> tuple[Tensor, Tensor]:
        objective_logits = functional_call(
            model,
            (next_state.parameters, next_state.buffers),
            (objective_batch.images,),
        )
        objective_loss = cross_entropy_loss(objective_logits, objective_batch.labels)
        metagradient = torch.autograd.grad(
            objective_loss,
            inner_batch.logits,
            create_graph=True,
        )[0]
        return objective_loss, metagradient

    metagrad_ms, (objective_loss, metagradient) = _measure_ms(device, metagrad_pass)
    ensure_finite("objective_loss", objective_loss)
    ensure_finite("metagradient", metagradient)

    def second_order_pass() -> tuple[Tensor, tuple[Tensor, ...]]:
        second_order_scalar = 0.5 * metagradient.square().sum()
        gradients = torch.autograd.grad(
            second_order_scalar,
            tuple(state.parameters.values()),
        )
        return second_order_scalar, gradients

    second_order_backward_ms, (second_order_scalar, second_order_grads) = _measure_ms(
        device, second_order_pass
    )
    ensure_finite("second_order_scalar", second_order_scalar)
    for index, gradient in enumerate(second_order_grads):
        ensure_finite(f"second_order_grad[{index}]", gradient)

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "inner_step_ms": inner_step_ms,
        "metagrad_ms": metagrad_ms,
        "second_order_backward_ms": second_order_backward_ms,
        "peak_memory_mib": _peak_memory_mib(device),
        "forward_loss": float(forward_loss.item()),
        "inner_loss": float(inner_loss.item()),
        "objective_loss": float(objective_loss.item()),
        "gradient_l2": gradient_square_sum ** 0.5,
        "metagrad_l2": float(metagradient.square().sum().sqrt().item()),
        "second_order_scalar": float(second_order_scalar.item()),
        "all_finite": True,
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    model_config: ViTConfig | None = None,
) -> dict[str, Any]:
    configure_benchmark_determinism(config.seed)
    device = resolve_device(config.device)
    vit_config = model_config or make_vit_small_config(
        image_size=config.image_size,
        patch_size=config.patch_size,
        final_logit_scale=config.final_logit_scale,
    )
    model = VisionTransformerClassifier(vit_config).to(device=device, dtype=torch.float32)
    optimizer_config = SmoothAdamWConfig()
    parameter_count = count_parameters(model)

    iterations: list[dict[str, float | bool]] = []
    for iteration in range(config.iterations):
        inner_batch = make_synthetic_batch(
            batch_size=config.batch_size,
            image_size=vit_config.image_size,
            num_classes=vit_config.num_classes,
            device=device,
            seed=config.seed + iteration * 2,
        )
        objective_batch = make_synthetic_batch(
            batch_size=config.batch_size,
            image_size=vit_config.image_size,
            num_classes=vit_config.num_classes,
            device=device,
            seed=config.seed + iteration * 2 + 1,
        )
        iterations.append(
            run_benchmark_iteration(
                model=model,
                optimizer_config=optimizer_config,
                inner_batch=inner_batch,
                objective_batch=objective_batch,
            )
        )

    timing_keys = (
        "forward_ms",
        "backward_ms",
        "inner_step_ms",
        "metagrad_ms",
        "second_order_backward_ms",
    )
    timing = {
        key: summarize_series([float(iteration[key]) for iteration in iterations])
        for key in timing_keys
    }
    peak_memory_mib = max(float(iteration["peak_memory_mib"]) for iteration in iterations)
    return {
        "benchmark": "vit_s_backbone_gpu_readiness",
        "device": str(device),
        "parameter_count": parameter_count,
        "config": {
            "batch_size": config.batch_size,
            "image_size": vit_config.image_size,
            "patch_size": vit_config.patch_size,
            "iterations": config.iterations,
            "seed": config.seed,
            "final_logit_scale": vit_config.final_logit_scale,
            "num_classes": vit_config.num_classes,
        },
        "timing": timing,
        "peak_cuda_memory_mib": peak_memory_mib,
        "iterations": iterations,
        "all_finite": all(bool(iteration["all_finite"]) for iteration in iterations),
    }


def benchmark_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)


def write_json_output(path: str, payload: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload + "\n", encoding="utf-8")


def main() -> None:
    config = parse_args()
    try:
        result = run_benchmark(config)
    except torch.cuda.OutOfMemoryError as error:
        raise SystemExit(f"CUDA OOM while running ViT-S benchmark: {error}") from error
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise SystemExit(f"OOM while running ViT-S benchmark: {error}") from error
        raise
    payload = benchmark_to_json(result)
    print(payload)
    if config.json_output is not None:
        write_json_output(config.json_output, payload)


if __name__ == "__main__":
    main()
