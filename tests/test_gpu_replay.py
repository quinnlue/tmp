from __future__ import annotations

import pytest
import torch
from torch import Tensor

from _fixtures import OracleFixture, assert_states_close, make_oracle_fixture
from determinism import assert_replay_determinism, configure_replay_determinism
from functional_train import TrainState
from metagrad import InnerBatch, ObjectiveBatch, train_unrolled, unrolled_objective
from recursive_replay import ReplayCheckpointConfig, _lazy_reverse_states
from replay import recursive_replay_state, replay_metagradient


pytestmark = pytest.mark.gpu


def _cuda_fixture(steps: int) -> OracleFixture:
    fixture = make_oracle_fixture(steps=steps)
    device = torch.device("cuda")
    dtype = torch.float32

    def move_map(values: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: value.to(device=device, dtype=dtype)
            for name, value in values.items()
        }

    state = TrainState(
        parameters=move_map(fixture.initial_state.parameters),
        buffers=move_map(fixture.initial_state.buffers),
        first_moments=move_map(fixture.initial_state.first_moments),
        second_moments=move_map(fixture.initial_state.second_moments),
        step=fixture.initial_state.step,
    )
    trajectory = tuple(
        InnerBatch(
            batch.images.to(device=device, dtype=dtype),
            batch.patch_mask.to(device),
            batch.group_ids.to(device),
        )
        for batch in fixture.trajectory
    )
    objective_batch = ObjectiveBatch(
        fixture.objective_batch.images.to(device=device, dtype=dtype),
        fixture.objective_batch.patch_mask.to(device),
    )
    return OracleFixture(
        model=fixture.model.to(device=device, dtype=dtype),
        initial_state=state,
        trajectory=trajectory,
        objective_batch=objective_batch,
        optimizer_config=fixture.optimizer_config,
        temperature=fixture.temperature,
    )


def _cluster_inputs() -> tuple[Tensor, Tensor]:
    return (
        torch.tensor([-0.25, 0.35], device="cuda", requires_grad=True),
        torch.tensor([0.5, 0.5], device="cuda"),
    )


@pytest.mark.parametrize("tf32", [False, True])
def test_cuda_recursive_replay_is_bit_deterministic(tf32: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    configure_replay_determinism(123, tf32=tf32)
    assert_replay_determinism(tf32=tf32)
    fixture = _cuda_fixture(steps=4)
    logits, base = _cluster_inputs()

    unrolled = train_unrolled(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    first_state = recursive_replay_state(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    second_state = recursive_replay_state(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    assert_states_close(first_state, unrolled, rtol=0.0, atol=0.0)
    assert_states_close(first_state, second_state, rtol=0.0, atol=0.0)

    reference_logits = logits.detach().clone().requires_grad_(True)
    reference_objective = unrolled_objective(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        fixture.objective_batch,
        reference_logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        True,
    )
    (reference_gradient,) = torch.autograd.grad(
        reference_objective, reference_logits
    )
    first_logits = logits.detach().clone().requires_grad_(True)
    first_objective, first_gradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        fixture.objective_batch,
        first_logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    second_logits = logits.detach().clone().requires_grad_(True)
    second_objective, second_gradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        fixture.objective_batch,
        second_logits,
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=3,
    )
    assert torch.equal(first_objective, reference_objective.detach())
    assert torch.equal(first_gradient, reference_gradient)
    assert torch.equal(first_objective, second_objective)
    assert torch.equal(first_gradient, second_gradient)


@pytest.mark.parametrize("tf32", [False, True])
def test_cuda_lazy_tree_reproduces_every_recorded_state(tf32: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    configure_replay_determinism(321, tf32=tf32)
    fixture = _cuda_fixture(steps=5)
    logits, base = _cluster_inputs()

    recorded = [fixture.initial_state]
    state = fixture.initial_state
    for batch in fixture.trajectory:
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
            (fixture.trajectory[step_index],),
            logits.detach(),
            base,
            fixture.optimizer_config,
            fixture.temperature,
            False,
        )

    for index, replayed in _lazy_reverse_states(
        fixture.initial_state,
        len(fixture.trajectory),
        3,
        advance,
    ):
        assert_states_close(replayed, recorded[index], rtol=0.0, atol=0.0)


def test_cuda_disk_backed_replay_preserves_device_and_matches_memory(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    configure_replay_determinism(123, tf32=False)
    fixture = _cuda_fixture(steps=3)
    logits, base = _cluster_inputs()

    memory_objective, memory_gradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        fixture.objective_batch,
        logits.detach().clone().requires_grad_(True),
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=2,
    )
    disk_objective, disk_gradient = replay_metagradient(
        fixture.model,
        fixture.initial_state,
        fixture.trajectory,
        fixture.objective_batch,
        logits.detach().clone().requires_grad_(True),
        base,
        fixture.optimizer_config,
        fixture.temperature,
        branching_factor=2,
        checkpoint_config=ReplayCheckpointConfig("disk", tmp_path, interval_steps=2),
    )

    assert disk_objective.device.type == "cuda"
    assert disk_gradient.device.type == "cuda"
    assert torch.equal(disk_objective, memory_objective)
    assert torch.equal(disk_gradient, memory_gradient)
    assert list(tmp_path.iterdir()) == []


def test_determinism_guard_detects_misconfigured_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    configure_replay_determinism(123, tf32=False)
    torch.backends.cudnn.benchmark = True
    with pytest.raises(RuntimeError, match="benchmarking"):
        assert_replay_determinism(tf32=False)
    torch.backends.cudnn.benchmark = False

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        assert_replay_determinism(tf32=False)
