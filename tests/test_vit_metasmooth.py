"""Unit tests for the ViT metasmoothness study (CPU-only, no real CIFAR).

These check the two preconditions the finite-difference diagnostics depend on:

  * **Determinism** -- the ViT learning algorithm `A(z)` (functional smooth-AdamW
    training) is bit-reproducible, so the only thing varying across the 3 runs of
    a metasmoothness probe is the metaparameter `z`.  Without this, the
    Definition-1 second difference is meaningless.
  * **Menu construction** -- every `ViTRoutine` in the menu builds a valid
    `ViTConfig`/model, and the CLS-token and post-norm code paths actually run.
"""
from __future__ import annotations

import torch
from torch.func import functional_call

import metasmooth as ms
from _vit_menu import (
    BASELINE_VIT,
    SMOOTH_VIT,
    SMOOTH_WIDE_VIT,
    ViTGeometry,
    ViTRoutine,
)
from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from model import VisionTransformerClassifier, cross_entropy_loss

# A tiny, CPU-friendly geometry.
_GEOM = ViTGeometry(image_size=16, patch=8, dim=16, depth=1, heads=2, mlp_ratio=2.0)
_NUM_CLASSES = 4


def _toy_data(n_train: int = 8, n_val: int = 4):
    gen = torch.Generator().manual_seed(0)
    return (
        torch.randn(n_train, 3, 16, 16, generator=gen),
        torch.randint(0, _NUM_CLASSES, (n_train,), generator=gen),
        torch.randn(n_val, 3, 16, 16, generator=gen),
        torch.randint(0, _NUM_CLASSES, (n_val,), generator=gen),
    )


def _train_and_eval(routine: ViTRoutine, z: torch.Tensor, data, steps: int = 4):
    """A minimal stand-in for the runner's `run_vit_algorithm`: deterministically
    train the menu'd ViT with per-cluster loss weights `z` and return
    (flattened params, held-out CE)."""
    train_x, train_y, val_x, val_y = data
    ms.configure_determinism(0, tf32=False)
    model = VisionTransformerClassifier(routine.build_config(_NUM_CLASSES, _GEOM))
    state = initialize_train_state(model)
    base_masses = torch.full((z.numel(),), 1.0 / z.numel())
    opt = SmoothAdamWConfig(learning_rate=1e-2, betas=(0.9, 0.99), eps=1e-4)
    for _ in range(steps):
        state, _ = weighted_inner_step(
            model, state, train_x, train_y, train_y.clone(), z, base_masses, opt,
            create_graph=False,
        )
    with torch.no_grad():
        logits = functional_call(model, (state.parameters, state.buffers), (val_x,))
        val_loss = float(cross_entropy_loss(logits, val_y).item())
    theta = torch.cat([v.detach().reshape(-1) for v in state.parameters.values()])
    return theta, val_loss


# --------------------------------------------------------------------------- #
# Menu construction
# --------------------------------------------------------------------------- #
def test_menu_routines_build_and_forward() -> None:
    images = torch.randn(2, 3, 16, 16)
    for routine in (BASELINE_VIT, SMOOTH_VIT, SMOOTH_WIDE_VIT,
                    ViTRoutine(name="cls", pool="cls"),
                    ViTRoutine(name="post", pre_norm=False),
                    ViTRoutine(name="relu", smooth_act=False)):
        config = routine.build_config(_NUM_CLASSES, _GEOM)
        model = VisionTransformerClassifier(config)
        logits = model(images)
        assert logits.shape == (2, _NUM_CLASSES)
        assert torch.isfinite(logits).all()


def test_width_lever_widens_encoder_dim() -> None:
    base = BASELINE_VIT.build_config(_NUM_CLASSES, _GEOM)
    wide = ViTRoutine(name="w2", width_mult=2.0).build_config(_NUM_CLASSES, _GEOM)
    assert wide.encoder_dim > base.encoder_dim
    # encoder_dim must stay divisible by heads and 4 (ViTConfig invariants).
    assert wide.encoder_dim % _GEOM.heads == 0 and wide.encoder_dim % 4 == 0


def test_final_scale_divides_logits() -> None:
    images = torch.randn(2, 3, 16, 16)
    ms.configure_determinism(0)
    unscaled = VisionTransformerClassifier(
        ViTRoutine(name="s1", final_scale=1.0).build_config(_NUM_CLASSES, _GEOM)
    )
    ms.configure_determinism(0)
    scaled = VisionTransformerClassifier(
        ViTRoutine(name="s10", final_scale=10.0).build_config(_NUM_CLASSES, _GEOM)
    )
    torch.testing.assert_close(scaled(images), unscaled(images) / 10.0)


# --------------------------------------------------------------------------- #
# Determinism (the metasmoothness precondition)
# --------------------------------------------------------------------------- #
def test_training_is_deterministic_for_equal_z() -> None:
    data = _toy_data()
    z = torch.zeros(_NUM_CLASSES)
    theta_a, loss_a = _train_and_eval(SMOOTH_VIT, z, data)
    theta_b, loss_b = _train_and_eval(SMOOTH_VIT, z, data)
    assert float((theta_a - theta_b).abs().max()) == 0.0
    assert loss_a == loss_b


def test_perturbing_z_changes_the_trained_model() -> None:
    data = _toy_data()
    z0 = torch.zeros(_NUM_CLASSES)
    z1 = z0.clone()
    z1[0] = 0.5  # up-weight one cluster
    theta0, _ = _train_and_eval(SMOOTH_VIT, z0, data)
    theta1, _ = _train_and_eval(SMOOTH_VIT, z1, data)
    assert float((theta0 - theta1).abs().max()) > 0.0


def test_measure_direction_runs_on_toy_vit() -> None:
    """End-to-end: the metasmooth estimators consume the ViT run closure."""
    data = _toy_data()

    def run_fn(z: torch.Tensor) -> ms.RunResult:
        theta, val_loss = _train_and_eval(SMOOTH_VIT, z, data)
        return ms.RunResult(theta=theta, val_loss=val_loss, train_loss=0.0, val_acc=0.0)

    z0 = torch.zeros(_NUM_CLASSES)
    v = ms.sample_direction(_NUM_CLASSES, seed=1000, normalize=False, device=torch.device("cpu"))
    dr = ms.measure_direction(run_fn, z0, v, h=0.05)
    assert dr.s_curvature >= 0.0
    assert -1.0 <= dr.s_hat <= 1.0
