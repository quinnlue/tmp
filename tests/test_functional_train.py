import torch
from torch.func import functional_call

from functional_train import (
    SmoothAdamWConfig,
    initialize_train_state,
    weighted_inner_step,
)
from mae import MaeConfig, MaskedAutoencoderViT, make_patch_mask, masked_reconstruction_loss


def tiny_model() -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        MaeConfig(
            image_size=16,
            patch_size=8,
            encoder_dim=16,
            encoder_depth=1,
            encoder_heads=2,
            decoder_dim=16,
            decoder_depth=1,
            decoder_heads=2,
            mlp_ratio=2.0,
            mask_ratio=0.5,
        )
    ).double()


def test_weighted_inner_step_is_functional_and_depends_on_logits() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    original_state = initialize_train_state(model)
    images = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    mask = make_patch_mask(
        [10, 20], step=0, seed=3, num_patches=4, mask_ratio=0.5
    )
    group_ids = torch.tensor([0, 1])
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)

    next_state, loss = weighted_inner_step(
        model,
        original_state,
        images,
        mask,
        group_ids,
        logits,
        base_group_masses,
        SmoothAdamWConfig(learning_rate=1e-3, weight_decay=0.05),
    )
    next_predictions = functional_call(
        model, (next_state.parameters, next_state.buffers), (images, mask)
    )
    next_loss = masked_reconstruction_loss(images, next_predictions, mask)
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
