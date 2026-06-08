import torch

from weighting import example_multipliers, weighted_example_loss


def test_uniform_logits_reproduce_base_distribution_loss() -> None:
    losses = torch.tensor([1.0, 3.0, 5.0, 7.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    logits = torch.zeros(2)
    base_group_masses = torch.tensor([0.5, 0.5])

    multipliers = example_multipliers(logits, group_ids, base_group_masses)
    weighted_loss = weighted_example_loss(
        losses, logits, group_ids, base_group_masses
    )

    torch.testing.assert_close(multipliers, torch.ones_like(losses))
    torch.testing.assert_close(weighted_loss, losses.mean())


def test_weighted_loss_is_differentiable_with_respect_to_logits() -> None:
    losses = torch.tensor([1.0, 1.0, 5.0, 5.0], dtype=torch.float64)
    group_ids = torch.tensor([0, 0, 1, 1])
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    base_group_masses = torch.tensor([0.5, 0.5], dtype=torch.float64)

    loss = weighted_example_loss(losses, logits, group_ids, base_group_masses)
    gradient = torch.autograd.grad(loss, logits)[0]

    assert gradient[0] < 0
    assert gradient[1] > 0
