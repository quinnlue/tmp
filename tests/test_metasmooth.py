"""Unit tests for the metasmoothness estimators in ``metasmooth.py``.

These exercise the pure-function metric machinery (Definitions 1 and 2 of
arXiv:2503.13751) against analytic training functions, so they are fast and need
no GPU training.
"""
from __future__ import annotations

import math

import torch

import metasmooth as ms


def _fake_subset(n_train: int, n_val: int, num_classes: int) -> ms.CifarSubset:
    gen = torch.Generator().manual_seed(0)
    return ms.CifarSubset(
        train_images=torch.randn(n_train, 3, 32, 32, generator=gen),
        train_labels=torch.randint(0, num_classes, (n_train,), generator=gen),
        val_images=torch.randn(n_val, 3, 32, 32, generator=gen),
        val_labels=torch.randint(0, num_classes, (n_val,), generator=gen),
    )


# --------------------------------------------------------------------------- #
# Metaparameter construction
# --------------------------------------------------------------------------- #
def test_per_example_metaparam_is_identity() -> None:
    subset = _fake_subset(8, 4, num_classes=10)
    mp = ms.make_metaparam("per_example", subset)
    assert mp.dim == 8
    torch.testing.assert_close(mp.coord_of_example, torch.arange(8))


def test_per_cluster_metaparam_maps_examples_to_class() -> None:
    subset = _fake_subset(8, 4, num_classes=10)
    mp = ms.make_metaparam("per_cluster", subset)
    assert mp.dim == int(subset.train_labels.max()) + 1
    torch.testing.assert_close(mp.coord_of_example, subset.train_labels)


def test_multipliers_are_unit_at_zero() -> None:
    subset = _fake_subset(8, 4, num_classes=10)
    for kind in ("per_example", "per_cluster"):
        mp = ms.make_metaparam(kind, subset)
        z0 = torch.zeros(mp.dim)
        mult = ms._example_multipliers(z0, mp)
        torch.testing.assert_close(mult, torch.ones(subset.n_train))


def test_per_cluster_weight_upweights_whole_class() -> None:
    subset = _fake_subset(8, 4, num_classes=10)
    mp = ms.make_metaparam("per_cluster", subset)
    z = torch.zeros(mp.dim)
    target = int(subset.train_labels[0])
    z[target] = math.log(2.0)  # double the weight of that class
    mult = ms._example_multipliers(z, mp)
    expected = torch.where(subset.train_labels == target, 2.0, 1.0)
    torch.testing.assert_close(mult, expected)


# --------------------------------------------------------------------------- #
# Definition 1: curvature metasmoothness
# --------------------------------------------------------------------------- #
def _quadratic_run(curvature: float, slope: float, theta_dir: torch.Tensor):
    """A fake learning algorithm whose held-out loss is exactly
    ``A + slope*t + curvature*t**2`` and whose parameters are linear in ``t``,
    where ``t`` is the signed distance along the probe direction from ``z0``."""

    def run(z: torch.Tensor) -> ms.RunResult:
        # The probe moves z by multiples of h*v with v chosen so that
        # z.sum() == t, so the scalar coordinate t is just the sum of z.
        t = float(z.sum())
        val = 1.0 + slope * t + curvature * t * t
        theta = (t * theta_dir).clone()
        return ms.RunResult(theta=theta, val_loss=val, train_loss=0.0)

    return run


def test_curvature_recovers_quadratic_second_derivative() -> None:
    # f along the line is 1 + 0.5 t + 3 t^2  ->  S = |2*curvature| = 6, any h.
    v = torch.ones(5)  # so z.sum() == t for z = t*v
    theta_dir = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    run = _quadratic_run(curvature=3.0, slope=0.5, theta_dir=theta_dir)
    z0 = torch.zeros(5)
    for h in (0.01, 0.1, 0.5):
        dr = ms.measure_direction(run, z0, v / 5.0, h)  # v/5 so z.sum()==t
        assert abs(dr.s_curvature - 6.0) < 1e-3  # S = 2*curvature, any h


def test_linear_training_function_is_perfectly_smooth() -> None:
    v = torch.ones(4)
    run = _quadratic_run(curvature=0.0, slope=2.0, theta_dir=torch.ones(4))
    dr = ms.measure_direction(run, torch.zeros(4), v / 4.0, 0.1)
    assert dr.s_curvature < 1e-6


# --------------------------------------------------------------------------- #
# Definition 2: empirical metasmoothness (sign agreement)
# --------------------------------------------------------------------------- #
def test_empirical_metasmoothness_perfect_agreement_is_one() -> None:
    # theta linear in t -> consecutive finite-difference metagradients identical
    # -> sign agreement == +1 on every coordinate.
    v = torch.ones(6)
    theta_dir = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    run = _quadratic_run(curvature=1.0, slope=0.0, theta_dir=theta_dir)
    dr = ms.measure_direction(run, torch.zeros(6), v / 6.0, 0.1)
    assert abs(dr.s_hat - 1.0) < 1e-6


def test_empirical_metasmoothness_in_unit_interval() -> None:
    v = torch.ones(6)
    run = _quadratic_run(curvature=1.0, slope=0.3, theta_dir=torch.randn(6))
    dr = ms.measure_direction(run, torch.zeros(6), v / 6.0, 0.05)
    assert -1.0 - 1e-6 <= dr.s_hat <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# Direction sampling
# --------------------------------------------------------------------------- #
def test_sample_direction_is_deterministic() -> None:
    a = ms.sample_direction(100, seed=7, normalize=False, device=torch.device("cpu"))
    b = ms.sample_direction(100, seed=7, normalize=False, device=torch.device("cpu"))
    torch.testing.assert_close(a, b)


def test_sample_direction_normalized_is_unit_norm() -> None:
    v = ms.sample_direction(100, seed=7, normalize=True, device=torch.device("cpu"))
    torch.testing.assert_close(v.norm(), torch.tensor(1.0))
