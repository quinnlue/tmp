import torch
from torch.func import functional_call

from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from model import ViTConfig, VisionTransformerClassifier, cross_entropy_loss


def tiny_model() -> VisionTransformerClassifier:
    return VisionTransformerClassifier(
        ViTConfig(
            image_size=16,
            patch_size=8,
            encoder_dim=16,
            encoder_depth=1,
            encoder_heads=2,
            mlp_ratio=2.0,
            num_classes=2,
        )
    ).double()


def test_weighted_inner_step_is_functional_and_depends_on_logits() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    original_state = initialize_train_state(model)
    images = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    labels = torch.tensor([0, 1])
    group_ids = torch.tensor([0, 1])
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)

    next_state, loss = weighted_inner_step(
        model,
        original_state,
        images,
        labels,
        group_ids,
        logits,
        base_group_masses,
        SmoothAdamWConfig(learning_rate=1e-3, weight_decay=0.05),
    )
    next_predictions = functional_call(
        model, (next_state.parameters, next_state.buffers), (images,)
    )
    next_loss = cross_entropy_loss(next_predictions, labels)
    meta_gradient = torch.autograd.grad(next_loss, logits)[0]

    assert original_state.step == 0
    assert next_state.step == 1
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(
        not torch.equal(original_state.parameters[name], next_state.parameters[name])
        for name in original_state.parameters
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(meta_gradient).all()
    assert not torch.equal(meta_gradient, torch.zeros_like(meta_gradient))
