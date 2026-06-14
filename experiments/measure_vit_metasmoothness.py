from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, TextIO

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call

import metasmooth as ms
from functional_train import SmoothAdamWConfig, initialize_train_state, weighted_inner_step
from metagrad import InnerBatch, ObjectiveBatch
from model import ViTConfig, VisionTransformerClassifier, cross_entropy_loss


GroupKind = Literal["per_class", "per_example"]
SizePreset = Literal["small", "full"]


@dataclass(frozen=True)
class ViTAblation:
    name: str
    pool: Literal["mean", "cls"] = "mean"
    pre_norm: bool = True
    final_logit_scale: float = 10.0


@dataclass(frozen=True)
class SyntheticProblem:
    train_images: Tensor
    train_labels: Tensor
    objective_images: Tensor
    objective_labels: Tensor
    per_class_group_ids: Tensor
    per_example_group_ids: Tensor

    @property
    def num_train_examples(self) -> int:
        return int(self.train_labels.numel())

    @property
    def num_classes(self) -> int:
        return int(self.train_labels.max().item()) + 1

    def group_ids(self, kind: GroupKind) -> Tensor:
        if kind == "per_class":
            return self.per_class_group_ids
        if kind == "per_example":
            return self.per_example_group_ids
        raise ValueError(f"unsupported group kind: {kind}")

    def base_group_masses(self, kind: GroupKind) -> Tensor:
        group_count = self.num_classes if kind == "per_class" else self.num_train_examples
        return torch.full(
            (group_count,),
            1.0 / group_count,
            dtype=self.train_images.dtype,
            device=self.train_images.device,
        )


@dataclass(frozen=True)
class ProbeConfig:
    size: SizePreset = "small"
    image_size: int = 32
    patch_size: int = 8
    encoder_dim: int | None = None
    encoder_depth: int | None = None
    encoder_heads: int | None = None
    mlp_ratio: float | None = None
    num_classes: int = 4
    examples_per_class: int = 4
    objective_per_class: int = 2
    trajectory_length: int = 6
    batch_size: int = 4
    seed: int = 0
    direction_seed: int = 1_000
    h: float = 0.05
    num_directions: int = 2
    normalize_directions: bool = False
    temperature: float = 1.0
    optimizer: SmoothAdamWConfig = SmoothAdamWConfig(
        learning_rate=1.5e-2,
        betas=(0.9, 0.99),
        eps=1e-4,
        weight_decay=0.0,
    )
    device: str = "cpu"
    json_output: str | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.examples_per_class <= 0 or self.objective_per_class <= 0:
            raise ValueError("examples_per_class and objective_per_class must be positive")
        if self.trajectory_length <= 0 or self.batch_size <= 0:
            raise ValueError("trajectory_length and batch_size must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.num_directions <= 0:
            raise ValueError("num_directions must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


def default_ablations() -> list[ViTAblation]:
    return [
        ViTAblation(name="mean_pre_scale10", pool="mean", pre_norm=True, final_logit_scale=10.0),
        ViTAblation(name="mean_pre_scale1", pool="mean", pre_norm=True, final_logit_scale=1.0),
        ViTAblation(name="cls_pre_scale10", pool="cls", pre_norm=True, final_logit_scale=10.0),
        ViTAblation(name="mean_post_scale10", pool="mean", pre_norm=False, final_logit_scale=10.0),
    ]


def select_ablations(names: str | None) -> list[ViTAblation]:
    available = {ablation.name: ablation for ablation in default_ablations()}
    if names is None:
        return list(available.values())
    selected_names = tuple(name.strip() for name in names.split(",") if name.strip())
    if not selected_names:
        raise ValueError("at least one ablation name is required")
    unknown = [name for name in selected_names if name not in available]
    if unknown:
        raise ValueError(
            f"unknown ablations {unknown}; available: {sorted(available)}"
        )
    return [available[name] for name in selected_names]


def resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def make_vit_config(config: ProbeConfig, ablation: ViTAblation) -> ViTConfig:
    preset = {
        "small": dict(encoder_dim=64, encoder_depth=4, encoder_heads=4, mlp_ratio=2.0),
        "full": dict(encoder_dim=384, encoder_depth=12, encoder_heads=6, mlp_ratio=4.0),
    }[config.size]
    encoder_dim = config.encoder_dim or preset["encoder_dim"]
    encoder_depth = config.encoder_depth or preset["encoder_depth"]
    encoder_heads = config.encoder_heads or preset["encoder_heads"]
    mlp_ratio = config.mlp_ratio or preset["mlp_ratio"]
    return ViTConfig(
        image_size=config.image_size,
        patch_size=config.patch_size,
        encoder_dim=encoder_dim,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        mlp_ratio=mlp_ratio,
        num_classes=config.num_classes,
        pre_norm=ablation.pre_norm,
        pool=ablation.pool,
        smooth_activation=True,
        final_logit_scale=ablation.final_logit_scale,
    )


def _class_template(image_size: int, class_id: int) -> Tensor:
    coords = torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    freq = float(class_id + 1)
    template = torch.stack(
        (
            torch.sin(math.pi * freq * xx),
            torch.cos(math.pi * freq * yy),
            torch.sin(math.pi * (freq + 0.5) * (xx + yy)),
        ),
        dim=0,
    )
    return template


def make_synthetic_problem(config: ProbeConfig, *, device: torch.device) -> SyntheticProblem:
    train_images: list[Tensor] = []
    objective_images: list[Tensor] = []
    train_labels: list[Tensor] = []
    objective_labels: list[Tensor] = []

    for class_id in range(config.num_classes):
        template = _class_template(config.image_size, class_id)
        for example_id in range(config.examples_per_class):
            seed = config.seed + class_id * 10_000 + example_id
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = 0.05 * torch.randn(
                3,
                config.image_size,
                config.image_size,
                generator=gen,
                dtype=torch.float32,
            )
            train_images.append(template + noise)
            train_labels.append(torch.tensor(class_id, dtype=torch.long))
        for example_id in range(config.objective_per_class):
            seed = config.seed + 1_000_000 + class_id * 10_000 + example_id
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = 0.05 * torch.randn(
                3,
                config.image_size,
                config.image_size,
                generator=gen,
                dtype=torch.float32,
            )
            objective_images.append(template + noise)
            objective_labels.append(torch.tensor(class_id, dtype=torch.long))

    train_tensor = torch.stack(train_images).to(device=device, dtype=torch.float32)
    train_label_tensor = torch.stack(train_labels).to(device=device)
    objective_tensor = torch.stack(objective_images).to(device=device, dtype=torch.float32)
    objective_label_tensor = torch.stack(objective_labels).to(device=device)
    return SyntheticProblem(
        train_images=train_tensor,
        train_labels=train_label_tensor,
        objective_images=objective_tensor,
        objective_labels=objective_label_tensor,
        per_class_group_ids=train_label_tensor.clone(),
        per_example_group_ids=torch.arange(train_tensor.shape[0], device=device, dtype=torch.long),
    )


def build_fixed_trajectory(
    problem: SyntheticProblem,
    *,
    group_kind: GroupKind,
    trajectory_length: int,
    batch_size: int,
    seed: int,
) -> tuple[InnerBatch, ...]:
    group_ids = problem.group_ids(group_kind)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ordering = torch.randperm(problem.num_train_examples, generator=gen)
    batches: list[InnerBatch] = []
    for step in range(trajectory_length):
        start = (step * batch_size) % problem.num_train_examples
        step_indices = ordering[start : start + batch_size]
        if step_indices.numel() < batch_size:
            shortfall = batch_size - step_indices.numel()
            step_indices = torch.cat((step_indices, ordering[:shortfall]), dim=0)
        step_indices = step_indices.to(problem.train_images.device)
        batches.append(
            InnerBatch(
                images=problem.train_images[step_indices],
                labels=problem.train_labels[step_indices],
                group_ids=group_ids[step_indices],
            )
        )
    return tuple(batches)


def flatten_parameters(parameters: dict[str, Tensor]) -> Tensor:
    return torch.cat([value.detach().reshape(-1).cpu().float() for value in parameters.values()])


def compute_sign_agreement(theta0: Tensor, thetah: Tensor, theta2h: Tensor, h: float) -> float:
    delta0 = (thetah - theta0) / h
    deltah = (theta2h - thetah) / h
    distance = (theta2h - theta0).abs()
    total = distance.sum()
    if float(total) == 0.0:
        return float("nan")
    weights = distance / total
    return float((torch.sign(delta0) * torch.sign(deltah) * weights).sum().item())


def run_vit_training(
    z: Tensor,
    *,
    config: ProbeConfig,
    vit_config: ViTConfig,
    problem: SyntheticProblem,
    trajectory: tuple[InnerBatch, ...],
    group_kind: GroupKind,
    device: torch.device,
) -> ms.RunResult:
    ms.configure_determinism(config.seed, tf32=False, strict=False)
    model = VisionTransformerClassifier(vit_config).to(device=device, dtype=torch.float32)
    state = initialize_train_state(model)
    objective_batch = ObjectiveBatch(problem.objective_images, problem.objective_labels)
    base_group_masses = problem.base_group_masses(group_kind)
    last_train_loss = 0.0

    for batch in trajectory:
        state, train_loss = weighted_inner_step(
            model,
            state,
            batch.images,
            batch.labels,
            batch.group_ids,
            z.to(device=device, dtype=torch.float32),
            base_group_masses,
            config.optimizer,
            temperature=config.temperature,
            create_graph=False,
        )
        last_train_loss = float(train_loss.detach().cpu().item())

    with torch.no_grad():
        logits = functional_call(
            model,
            (state.parameters, state.buffers),
            (objective_batch.images,),
        )
        val_loss = float(cross_entropy_loss(logits, objective_batch.labels).item())
        val_acc = float((logits.argmax(dim=1) == objective_batch.labels).float().mean().item())

    return ms.RunResult(
        theta=flatten_parameters(state.parameters),
        val_loss=val_loss,
        train_loss=last_train_loss,
        val_acc=val_acc,
    )


def summarize_direction_results(
    directions: list[ms.DirectionResult],
    *,
    group_kind: GroupKind,
    ablation: ViTAblation,
    vit_config: ViTConfig,
    config: ProbeConfig,
) -> dict[str, object]:
    s_curvature = [direction.s_curvature for direction in directions]
    s_hat = [direction.s_hat for direction in directions]
    return {
        "ablation": ablation.name,
        "group_kind": group_kind,
        "model_config": asdict(vit_config),
        "h": config.h,
        "num_directions": len(directions),
        "s_curvature": s_curvature,
        "s_curvature_mean": float(np.mean(s_curvature)),
        "s_curvature_std": float(np.std(s_curvature)),
        "s_hat": s_hat,
        "s_hat_mean": float(np.nanmean(s_hat)),
        "s_hat_std": float(np.nanstd(s_hat)),
        "f0": directions[0].f0,
        "val_acc": directions[0].acc0,
        "directions": [
            {
                "index": index,
                "s_curvature": direction.s_curvature,
                "s_hat": direction.s_hat,
                "f0": direction.f0,
                "fh": direction.fh,
                "f2h": direction.f2h,
                "first_diff": direction.first_diff,
                "second_diff": direction.second_diff,
                "acc0": direction.acc0,
            }
            for index, direction in enumerate(directions)
        ],
    }


def emit_json_event(event: dict[str, object], *, output: TextIO | None) -> None:
    line = json.dumps(event, sort_keys=True)
    print(line)
    if output is not None:
        output.write(line + "\n")
        output.flush()


def run_probe_suite(
    config: ProbeConfig,
    *,
    ablations: list[ViTAblation] | None = None,
    group_kinds: tuple[GroupKind, ...] = ("per_class", "per_example"),
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    device = resolve_device(config.device)
    problem = make_synthetic_problem(config, device=device)
    results: list[dict[str, object]] = []
    output_handle = open(config.json_output, "a", encoding="utf-8") if config.json_output else None
    try:
        selected_ablations = ablations or default_ablations()
        emit_json_event(
            {
                "event": "start",
                "device": str(device),
                "config": asdict(config),
                "ablations": [asdict(ablation) for ablation in selected_ablations],
                "group_kinds": list(group_kinds),
            },
            output=output_handle,
        )
        for ablation in selected_ablations:
            vit_config = make_vit_config(config, ablation)
            for group_kind in group_kinds:
                trajectory = build_fixed_trajectory(
                    problem,
                    group_kind=group_kind,
                    trajectory_length=config.trajectory_length,
                    batch_size=config.batch_size,
                    # Granularity is the only intended variable. Keep the
                    # underlying example ordering identical across groupings.
                    seed=config.seed,
                )

                def run_fn(z: Tensor) -> ms.RunResult:
                    return run_vit_training(
                        z,
                        config=config,
                        vit_config=vit_config,
                        problem=problem,
                        trajectory=trajectory,
                        group_kind=group_kind,
                        device=device,
                    )

                z0 = torch.zeros(
                    problem.num_classes if group_kind == "per_class" else problem.num_train_examples,
                    device=device,
                    dtype=torch.float32,
                )
                directions: list[ms.DirectionResult] = []
                for direction_index in range(config.num_directions):
                    direction = ms.sample_direction(
                        z0.numel(),
                        config.direction_seed + direction_index,
                        normalize=config.normalize_directions,
                        device=device,
                    )
                    result = ms.measure_direction(run_fn, z0, direction, config.h)
                    directions.append(result)
                    emit_json_event(
                        {
                            "event": "direction",
                            "ablation": ablation.name,
                            "group_kind": group_kind,
                            "direction_index": direction_index,
                            "s_curvature": result.s_curvature,
                            "s_hat": result.s_hat,
                            "f0": result.f0,
                            "fh": result.fh,
                            "f2h": result.f2h,
                            "first_diff": result.first_diff,
                            "second_diff": result.second_diff,
                            "acc0": result.acc0,
                        },
                        output=output_handle,
                    )
                    if progress is not None:
                        progress(
                            f"[{ablation.name}/{group_kind}] dir {direction_index + 1}/{config.num_directions} "
                            f"S={result.s_curvature:.4f} S_hat={result.s_hat:+.4f}"
                        )
                summary = summarize_direction_results(
                    directions,
                    group_kind=group_kind,
                    ablation=ablation,
                    vit_config=vit_config,
                    config=config,
                )
                results.append(summary)
                emit_json_event({"event": "summary", **summary}, output=output_handle)
        emit_json_event({"event": "complete", "num_results": len(results)}, output=output_handle)
        return results
    finally:
        if output_handle is not None:
            output_handle.close()


def parse_args() -> tuple[ProbeConfig, list[ViTAblation], tuple[GroupKind, ...]]:
    parser = argparse.ArgumentParser(
        description=(
            "Short-horizon ViT metasmoothness probe on deterministic synthetic image classification. "
            "Uses fp32 functional SmoothAdamW so the Definition-1 second difference stays faithful."
        )
    )
    parser.add_argument("--size", choices=("small", "full"), default="small")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--encoder-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--encoder-heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--examples-per-class", type=int, default=4)
    parser.add_argument("--objective-per-class", type=int, default=2)
    parser.add_argument("--trajectory-length", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-seed", type=int, default=1_000)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--num-directions", type=int, default=2)
    parser.add_argument("--normalize-directions", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument(
        "--ablations",
        default=None,
        help="comma-separated subset of the default ablation names",
    )
    parser.add_argument(
        "--group-kinds",
        default="per_class,per_example",
        help="comma-separated subset of per_class,per_example",
    )
    args = parser.parse_args()
    group_kinds = tuple(name.strip() for name in args.group_kinds.split(",") if name.strip())
    if not group_kinds or any(name not in ("per_class", "per_example") for name in group_kinds):
        parser.error("--group-kinds must contain per_class and/or per_example")
    config = ProbeConfig(
        size=args.size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        examples_per_class=args.examples_per_class,
        objective_per_class=args.objective_per_class,
        trajectory_length=args.trajectory_length,
        batch_size=args.batch_size,
        seed=args.seed,
        direction_seed=args.direction_seed,
        h=args.h,
        num_directions=args.num_directions,
        normalize_directions=args.normalize_directions,
        temperature=args.temperature,
        device=args.device,
        json_output=args.json_output,
    )
    try:
        ablations = select_ablations(args.ablations)
    except ValueError as error:
        parser.error(str(error))
    return config, ablations, group_kinds  # type: ignore[return-value]


def main() -> None:
    config, ablations, group_kinds = parse_args()
    run_probe_suite(config, ablations=ablations, group_kinds=group_kinds)


if __name__ == "__main__":
    main()
