import torch
from torch.func import functional_call

from mae import (
    MaeConfig,
    MaskedAutoencoderViT,
    make_patch_mask,
    masked_reconstruction_loss,
    patchify,
    per_example_masked_reconstruction_loss,
    unpatchify,
)


def test_patchify_unpatchify_round_trip() -> None:
    images = torch.randn(2, 3, 32, 32)

    patches = patchify(images, patch_size=8)
    restored = unpatchify(patches, patch_size=8, image_size=32)

    assert patches.shape == (2, 16, 192)
    torch.testing.assert_close(restored, images)


def test_make_patch_mask_is_pure_and_input_dependent() -> None:
    image_ids = torch.tensor([10, 20, 30])

    first = make_patch_mask(image_ids, step=4, seed=7, num_patches=16, mask_ratio=0.75)
    repeated = make_patch_mask(
        image_ids, step=4, seed=7, num_patches=16, mask_ratio=0.75
    )
    next_step = make_patch_mask(
        image_ids, step=5, seed=7, num_patches=16, mask_ratio=0.75
    )
    other_ids = make_patch_mask(
        image_ids + 1, step=4, seed=7, num_patches=16, mask_ratio=0.75
    )

    assert first.shape == (3, 16)
    assert first.dtype == torch.bool
    assert torch.equal(first.sum(dim=1), torch.tensor([12, 12, 12]))
    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_step)
    assert not torch.equal(first, other_ids)


def test_masked_reconstruction_loss_uses_only_hidden_patches() -> None:
    images = torch.zeros(1, 3, 8, 8)
    predictions = torch.zeros(1, 4, 48)
    mask = torch.tensor([[True, False, False, False]])

    predictions[:, 1:] = 100.0
    assert masked_reconstruction_loss(images, predictions, mask).item() == 0.0

    predictions[:, 0] = 2.0
    assert masked_reconstruction_loss(images, predictions, mask).item() == 4.0


def test_per_example_loss_preserves_batch_examples() -> None:
    images = torch.zeros(2, 3, 8, 8)
    predictions = torch.zeros(2, 4, 48)
    mask = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    predictions[0, 0] = 2.0
    predictions[1, 1] = 4.0

    losses = per_example_masked_reconstruction_loss(images, predictions, mask)

    torch.testing.assert_close(losses, torch.tensor([4.0, 16.0]))
    torch.testing.assert_close(
        masked_reconstruction_loss(images, predictions, mask), losses.mean()
    )


def test_forward_shapes_and_finite_parameter_gradients() -> None:
    config = MaeConfig()
    model = MaskedAutoencoderViT(config)
    images = torch.randn(2, 3, 32, 32)
    mask = make_patch_mask(
        [101, 202],
        step=0,
        seed=1,
        num_patches=config.num_patches,
        mask_ratio=config.mask_ratio,
    )

    predictions = model(images, mask)
    loss = masked_reconstruction_loss(images, predictions, mask)
    loss.backward()

    assert predictions.shape == (2, config.num_patches, config.patch_dim)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_forward_is_compatible_with_functional_call() -> None:
    config = MaeConfig(
        image_size=16,
        patch_size=8,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=2,
        decoder_dim=16,
        decoder_depth=1,
        decoder_heads=2,
    )
    model = MaskedAutoencoderViT(config)
    images = torch.randn(1, 3, 16, 16)
    mask = make_patch_mask(
        [5],
        step=0,
        seed=0,
        num_patches=config.num_patches,
        mask_ratio=config.mask_ratio,
    )
    state = dict(model.named_parameters())

    expected = model(images, mask)
    actual = functional_call(model, state, (images, mask))

    torch.testing.assert_close(actual, expected)


def test_second_order_gradient_through_math_sdpa_and_loss() -> None:
    config = MaeConfig(
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
    model = MaskedAutoencoderViT(config).double()
    images = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    mask = make_patch_mask(
        [42],
        step=3,
        seed=9,
        num_patches=config.num_patches,
        mask_ratio=config.mask_ratio,
    )
    parameters = tuple(model.parameters())

    loss = masked_reconstruction_loss(images, model(images, mask), mask)
    first_order = torch.autograd.grad(loss, parameters, create_graph=True)
    gradient_energy = sum(gradient.square().sum() for gradient in first_order)
    second_order = torch.autograd.grad(gradient_energy, parameters)

    assert all(torch.isfinite(gradient).all() for gradient in first_order)
    assert all(torch.isfinite(gradient).all() for gradient in second_order)
