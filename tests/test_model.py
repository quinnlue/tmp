import math

import torch
from torch.func import functional_call

from model import (
    ViTConfig,
    VisionTransformerClassifier,
    cross_entropy_loss,
    patchify,
    per_example_cross_entropy_loss,
)


def test_patchify_round_trip_shape() -> None:
    images = torch.randn(2, 3, 32, 32)

    patches = patchify(images, patch_size=8)

    assert patches.shape == (2, 16, 192)


def test_cross_entropy_loss_matches_reference() -> None:
    logits = torch.zeros(3, 4)
    labels = torch.tensor([0, 1, 2])

    per_example = per_example_cross_entropy_loss(logits, labels)
    torch.testing.assert_close(
        per_example, torch.full((3,), math.log(4.0))
    )
    torch.testing.assert_close(
        cross_entropy_loss(logits, labels), per_example.mean()
    )


def test_per_example_loss_preserves_batch_examples() -> None:
    logits = torch.tensor(
        [[10.0, 0.0], [0.0, 10.0]]
    )
    labels = torch.tensor([0, 0])

    losses = per_example_cross_entropy_loss(logits, labels)

    assert losses.shape == (2,)
    # The confident-correct example has near-zero loss; the confident-wrong one
    # is large.
    assert losses[0] < 1e-3
    assert losses[1] > 5.0
    torch.testing.assert_close(
        cross_entropy_loss(logits, labels), losses.mean()
    )


def test_forward_shapes_and_finite_parameter_gradients() -> None:
    config = ViTConfig()
    model = VisionTransformerClassifier(config)
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([3, 7])

    logits = model(images)
    loss = cross_entropy_loss(logits, labels)
    loss.backward()

    assert logits.shape == (2, config.num_classes)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_forward_is_compatible_with_functional_call() -> None:
    config = ViTConfig(
        image_size=16,
        patch_size=8,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=2,
        num_classes=5,
    )
    model = VisionTransformerClassifier(config)
    images = torch.randn(1, 3, 16, 16)
    state = dict(model.named_parameters())

    expected = model(images)
    actual = functional_call(model, state, (images,))

    torch.testing.assert_close(actual, expected)


def test_second_order_gradient_through_math_sdpa_and_loss() -> None:
    config = ViTConfig(
        image_size=16,
        patch_size=8,
        encoder_dim=16,
        encoder_depth=1,
        encoder_heads=2,
        mlp_ratio=2.0,
        num_classes=3,
    )
    model = VisionTransformerClassifier(config).double()
    images = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    labels = torch.tensor([0, 2])
    parameters = tuple(model.parameters())

    loss = cross_entropy_loss(model(images), labels)
    first_order = torch.autograd.grad(loss, parameters, create_graph=True)
    gradient_energy = sum(gradient.square().sum() for gradient in first_order)
    second_order = torch.autograd.grad(gradient_energy, parameters)

    assert all(torch.isfinite(gradient).all() for gradient in first_order)
    assert all(torch.isfinite(gradient).all() for gradient in second_order)
