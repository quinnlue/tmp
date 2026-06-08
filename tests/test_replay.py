"""L3 + GATE verification for paper-faithful recursive REPLAY."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

import recursive_replay as recursive_replay_module
from _fixtures import assert_states_close, make_oracle_fixture, retarget_trajectory
from functional_train import TrainState
from metagrad import train_unrolled, unrolled_objective
from recursive_replay import (
    ReplayCheckpointConfig,
    _ReplayStats,
    _flatten_diff_state,
    _lazy_reverse_states,
    recursive_replay_state as shared_recursive_replay_state,
)
from replay import recursive_replay_state, replay_metagradient, replay_objective


STEPS = 5
BRANCHING_FACTORS = [2, 3, STEPS + 2]


def _scalar_state(value: float = 0.0) -> TrainState:
    parameter = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    return TrainState(
        parameters={"p": parameter},
        buffers={},
        first_moments={"p": torch.zeros_like(parameter)},
        second_moments={"p": torch.zeros_like(parameter)},
        step=0,
    )


def _advance_scalar_state(state: TrainState, step_index: int) -> TrainState:
    del step_index
    parameter = state.parameters["p"] + 1.0
    return TrainState(
        parameters={"p": parameter.detach().requires_grad_(True)},
        buffers={},
        first_moments={"p": state.first_moments["p"] + 1.0},
        second_moments={"p": state.second_moments["p"] + 1.0},
        step=state.step + 1,
    )


def _granularity(name: str) -> tuple[Tensor, Tensor, Tensor]:
    if name == "cluster":
        group_ids = torch.tensor([0, 0, 1, 1])
        base = torch.tensor([0.5, 0.5], dtype=torch.float64)
        logits = torch.tensor([-0.25, 0.35], dtype=torch.float64)
    elif name == "sample":
        group_ids = torch.arange(4)
        base = torch.full((4,), 0.25, dtype=torch.float64)
        logits = torch.tensor([-0.35, -0.1, 0.2, 0.4], dtype=torch.float64)
    else:  # pragma: no cover
        raise ValueError(name)
    return group_ids, base, logits


@pytest.mark.cpu
@pytest.mark.parametrize("granularity", ["cluster", "sample"])
@pytest.mark.parametrize("branching_factor", BRANCHING_FACTORS)
def test_replay_metagradient_matches_unrolled(
    granularity: str, branching_factor: int
) -> None:
    fixture = make_oracle_fixture(steps=STEPS)
    group_ids, base, logit_values = _granularity(granularity)
    trajectory = retarget_trajectory(fixture, group_ids)

    logits_unrolled = logit_values.clone().requires_grad_(True)
    objective_unrolled = unrolled_objective(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        logits_unrolled,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    (gradient_unrolled,) = torch.autograd.grad(objective_unrolled, logits_unrolled)

    logits_replay = logit_values.clone().requires_grad_(True)
    objective_replay = replay_objective(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        logits_replay,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=branching_factor,
    )
    (gradient_replay,) = torch.autograd.grad(objective_replay, logits_replay)

    torch.testing.assert_close(
        objective_replay, objective_unrolled, rtol=1e-9, atol=1e-11
    )
    torch.testing.assert_close(
        gradient_replay, gradient_unrolled, rtol=1e-9, atol=1e-11
    )
    significant = gradient_unrolled.abs() > 1e-6
    assert torch.equal(
        gradient_replay[significant].sign(), gradient_unrolled[significant].sign()
    )


@pytest.mark.cpu
def test_replay_metagradient_api_matches_objective_grad() -> None:
    fixture = make_oracle_fixture(steps=STEPS)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)

    logits = logit_values.clone().requires_grad_(True)
    phi, metagradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )

    logits_reference = logit_values.clone().requires_grad_(True)
    objective_reference = unrolled_objective(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        logits_reference,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    (gradient_reference,) = torch.autograd.grad(
        objective_reference, logits_reference
    )

    assert not phi.requires_grad
    torch.testing.assert_close(
        phi, objective_reference.detach(), rtol=1e-9, atol=1e-11
    )
    torch.testing.assert_close(
        metagradient, gradient_reference, rtol=1e-9, atol=1e-11
    )


@pytest.mark.cpu
def test_disk_backed_replay_matches_memory_and_cleans_up(tmp_path) -> None:
    fixture = make_oracle_fixture(steps=STEPS)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)

    memory_logits = logit_values.clone().requires_grad_(True)
    memory_objective, memory_gradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        memory_logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    disk_logits = logit_values.clone().requires_grad_(True)
    disk_objective, disk_gradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        disk_logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=2),
    )

    torch.testing.assert_close(disk_objective, memory_objective, rtol=0.0, atol=0.0)
    torch.testing.assert_close(disk_gradient, memory_gradient, rtol=0.0, atol=0.0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_replay_metagradient_requires_grad_logits() -> None:
    fixture = make_oracle_fixture(steps=2)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    with pytest.raises(ValueError, match="require grad"):
        replay_metagradient(
            fixture.model,
            fixture.initial_state,
            trajectory,
            fixture.objective_batch,
            logit_values.clone(),
            base,
            fixture.optimizer_config,
            fixture.temperature,
            branching_factor=2,
        )


@pytest.mark.cpu
@pytest.mark.parametrize("branching_factor", BRANCHING_FACTORS)
def test_replay_state_matches_unrolled_bit_for_bit(branching_factor: int) -> None:
    fixture = make_oracle_fixture(steps=STEPS)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    logits = logit_values.clone().requires_grad_(True)

    unrolled_state = train_unrolled(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    replay_state = recursive_replay_state(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=branching_factor,
    )

    assert_states_close(replay_state, unrolled_state, rtol=0.0, atol=0.0)


@pytest.mark.cpu
def test_replay_state_is_deterministic_across_runs() -> None:
    fixture = make_oracle_fixture(steps=STEPS)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    logits = logit_values.clone().requires_grad_(True)

    first = recursive_replay_state(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    second = recursive_replay_state(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    assert_states_close(first, second, rtol=0.0, atol=0.0)


@pytest.mark.cpu
def test_lazy_tree_reproduces_every_unrolled_state_in_reverse_order() -> None:
    fixture = make_oracle_fixture(steps=STEPS)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    logits = logit_values.clone().requires_grad_(True)

    recorded = [fixture.initial_state]
    state = fixture.initial_state
    for batch in trajectory:
        state = train_unrolled(
            fixture.model,
            state,
            (batch,),
            logits,
            base,
            fixture.optimizer_config,
            fixture.temperature,
            False,
        )
        recorded.append(state)

    def advance(state: TrainState, step_index: int) -> TrainState:
        return train_unrolled(
            fixture.model,
            state,
            (trajectory[step_index],),
            logits.detach(),
            base,
            fixture.optimizer_config,
            fixture.temperature,
            False,
        )

    yielded = []
    for index, replayed in _lazy_reverse_states(
        fixture.initial_state,
        len(trajectory),
        3,
        advance,
    ):
        yielded.append(index)
        assert_states_close(replayed, recorded[index], rtol=0.0, atol=0.0)
    assert yielded == list(range(STEPS, -1, -1))


@pytest.mark.cpu
def test_disk_checkpoint_lifecycle_on_completion_and_early_close(tmp_path) -> None:
    checkpoint_directory = tmp_path / "nested" / "checkpoints"
    config = ReplayCheckpointConfig("disk", checkpoint_directory, interval_steps=2)
    states = _lazy_reverse_states(
        _scalar_state(),
        total_steps=5,
        branching_factor=3,
        advance=_advance_scalar_state,
        checkpoint_config=config,
    )
    next(states)
    assert list(checkpoint_directory.glob("replay-checkpoints-*/*.pt"))
    states.close()
    assert checkpoint_directory.is_dir()
    assert list(checkpoint_directory.iterdir()) == []

    assert len(
        list(
            _lazy_reverse_states(
                _scalar_state(),
                total_steps=5,
                branching_factor=3,
                advance=_advance_scalar_state,
                checkpoint_config=config,
            )
        )
    ) == 6
    assert list(checkpoint_directory.iterdir()) == []


@pytest.mark.cpu
def test_disk_checkpoint_cleanup_on_failure(tmp_path) -> None:
    def fail_advance(state: TrainState, step_index: int) -> TrainState:
        del state, step_index
        raise RuntimeError("advance failed")

    states = _lazy_reverse_states(
        _scalar_state(),
        total_steps=5,
        branching_factor=3,
        advance=fail_advance,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=2),
    )
    with pytest.raises(RuntimeError, match="advance failed"):
        next(states)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_disk_checkpoint_runs_are_isolated(tmp_path) -> None:
    config = ReplayCheckpointConfig("disk", tmp_path, interval_steps=2)
    first = _lazy_reverse_states(
        _scalar_state(),
        total_steps=5,
        branching_factor=3,
        advance=_advance_scalar_state,
        checkpoint_config=config,
    )
    second = _lazy_reverse_states(
        _scalar_state(),
        total_steps=5,
        branching_factor=3,
        advance=_advance_scalar_state,
        checkpoint_config=config,
    )
    next(first)
    next(second)
    assert len(list(tmp_path.glob("replay-checkpoints-*"))) == 2
    first.close()
    assert len(list(tmp_path.glob("replay-checkpoints-*"))) == 1
    second.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_disk_checkpoint_uses_system_temp_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    original_mkdtemp = recursive_replay_module.tempfile.mkdtemp
    requested_directories = []

    def redirected_mkdtemp(*, prefix: str, dir=None) -> str:
        requested_directories.append(dir)
        return original_mkdtemp(prefix=prefix, dir=tmp_path)

    monkeypatch.setattr(recursive_replay_module.tempfile, "mkdtemp", redirected_mkdtemp)
    states = _lazy_reverse_states(
        _scalar_state(),
        total_steps=2,
        branching_factor=2,
        advance=_advance_scalar_state,
        checkpoint_config=ReplayCheckpointConfig("disk", interval_steps=1),
    )
    next(states)
    states.close()

    assert requested_directories == [None]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_disk_checkpoint_serialization_errors_are_explicit_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fail_save(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(recursive_replay_module.torch, "save", fail_save)
    states = _lazy_reverse_states(
        _scalar_state(),
        total_steps=5,
        branching_factor=3,
        advance=_advance_scalar_state,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=2),
    )
    with pytest.raises(RuntimeError, match="failed to write REPLAY checkpoint"):
        next(states)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_disk_checkpoint_load_errors_are_explicit_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fail_load(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("read failed")

    monkeypatch.setattr(recursive_replay_module.torch, "load", fail_load)
    states = _lazy_reverse_states(
        _scalar_state(),
        total_steps=5,
        branching_factor=3,
        advance=_advance_scalar_state,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=2),
    )
    with pytest.raises(RuntimeError, match="failed to load REPLAY checkpoint"):
        next(states)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_interval_disk_checkpointing_yields_each_state_once_and_reduces_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    save_calls = 0
    load_calls = 0
    original_save = recursive_replay_module.torch.save
    original_load = recursive_replay_module.torch.load

    def counted_save(*args, **kwargs) -> None:
        nonlocal save_calls
        save_calls += 1
        original_save(*args, **kwargs)

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(recursive_replay_module.torch, "save", counted_save)
    monkeypatch.setattr(recursive_replay_module.torch, "load", counted_load)
    stats = _ReplayStats(yielded_state_indices=[])
    yielded = [
        index
        for index, _ in _lazy_reverse_states(
            _scalar_state(),
            total_steps=1_000,
            branching_factor=3,
            advance=_advance_scalar_state,
            checkpoint_config=ReplayCheckpointConfig(
                "disk", tmp_path, interval_steps=100
            ),
            stats=stats,
        )
    ]

    assert yielded == list(range(1_000, -1, -1))
    assert stats.yielded_state_indices == yielded
    assert stats.live_states == 1
    assert save_calls == 9
    assert load_calls == 9
    assert list(tmp_path.iterdir()) == []


@pytest.mark.cpu
def test_recursive_replay_adjoint_matches_jacobian_dot_product() -> None:
    fixture = make_oracle_fixture(steps=3)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    direction = torch.tensor([0.4, -0.7], dtype=torch.float64)

    def unrolled_flat(logits: Tensor) -> tuple[Tensor, ...]:
        return _flatten_diff_state(
            train_unrolled(
                fixture.model,
                fixture.initial_state,
                trajectory,
                logits,
                base,
                fixture.optimizer_config,
                fixture.temperature,
                True,
            )
        )

    _, jacobian_direction = torch.autograd.functional.jvp(
        unrolled_flat,
        logit_values,
        direction,
        create_graph=False,
    )
    state_direction = tuple(
        torch.linspace(
            -0.2,
            0.3,
            value.numel(),
            dtype=value.dtype,
        ).reshape_as(value)
        for value in jacobian_direction
    )
    left = sum(
        (value * vector).sum()
        for value, vector in zip(jacobian_direction, state_direction, strict=True)
    )

    logits = logit_values.clone().requires_grad_(True)
    replay_state = recursive_replay_state(
        fixture.model,
        fixture.initial_state,
        trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=2,
    )
    (state_vjp,) = torch.autograd.grad(
        _flatten_diff_state(replay_state),
        logits,
        grad_outputs=state_direction,
    )
    right = (state_vjp * direction).sum()
    torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-11)


@pytest.mark.cpu
def test_recursive_replay_traversal_respects_complexity_bounds() -> None:
    total_steps = 17
    branching_factor = 3
    parameter = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    initial_state = TrainState(
        parameters={"p": parameter},
        buffers={},
        first_moments={"p": torch.zeros_like(parameter)},
        second_moments={"p": torch.zeros_like(parameter)},
        step=0,
    )
    metaparameter = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    stats = _ReplayStats(yielded_state_indices=[])

    def step(
        state: TrainState,
        step_index: int,
        replay_metaparameter: Tensor,
        create_graph: bool,
    ) -> TrainState:
        del step_index, create_graph
        return TrainState(
            parameters={"p": state.parameters["p"] + replay_metaparameter},
            buffers={},
            first_moments={"p": state.first_moments["p"] + replay_metaparameter},
            second_moments={
                "p": state.second_moments["p"] + replay_metaparameter.square()
            },
            step=state.step + 1,
        )

    final_state = shared_recursive_replay_state(
        initial_state,
        metaparameter,
        total_steps,
        step,
        branching_factor=branching_factor,
        stats=stats,
    )
    torch.autograd.grad(final_state.parameters["p"], metaparameter)

    depth = math.ceil(math.log(total_steps + 1, branching_factor))
    assert stats.forward_steps == total_steps
    assert stats.yielded_state_indices == list(range(total_steps, -1, -1))
    assert stats.peak_live_states <= 1 + (branching_factor - 1) * depth
    assert stats.replay_steps <= total_steps * depth
    assert stats.live_states == 1


@pytest.mark.cpu
def test_replay_rejects_empty_trajectory_and_invalid_branching_factor() -> None:
    fixture = make_oracle_fixture(steps=2)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    with pytest.raises(ValueError, match="trajectory"):
        recursive_replay_state(
            fixture.model,
            fixture.initial_state,
            (),
            logit_values.clone().requires_grad_(True),
            base,
            fixture.optimizer_config,
            branching_factor=2,
        )
    with pytest.raises(ValueError, match="branching_factor"):
        recursive_replay_state(
            fixture.model,
            fixture.initial_state,
            trajectory,
            logit_values.clone().requires_grad_(True),
            base,
            fixture.optimizer_config,
            branching_factor=1,
        )
    with pytest.raises(ValueError, match="unknown REPLAY checkpoint backend"):
        ReplayCheckpointConfig("remote")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="directory is only valid"):
        ReplayCheckpointConfig("memory", "checkpoints")
    with pytest.raises(ValueError, match="interval_steps is only valid"):
        ReplayCheckpointConfig("memory", interval_steps=100)
    with pytest.raises(ValueError, match="interval_steps must be positive"):
        ReplayCheckpointConfig("disk", interval_steps=0)


@pytest.mark.cpu
def test_recursive_replay_rejects_higher_order_differentiation() -> None:
    fixture = make_oracle_fixture(steps=2)
    group_ids, base, logit_values = _granularity("cluster")
    trajectory = retarget_trajectory(fixture, group_ids)
    logits = logit_values.clone().requires_grad_(True)
    objective = replay_objective(
        fixture.model,
        fixture.initial_state,
        trajectory,
        fixture.objective_batch,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=2,
    )
    with pytest.raises(RuntimeError, match="higher-order"):
        torch.autograd.grad(objective, logits, create_graph=True)
