from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from functional_train import TrainState


ReplayStep = Callable[[TrainState, int, Tensor, bool], TrainState]
ProgressCallback = Callable[[], None]


@dataclass(frozen=True)
class ReplayCheckpointConfig:
    """Configure storage for ephemeral lazy-tree REPLAY checkpoints."""

    backend: Literal["memory", "disk"] = "memory"
    directory: str | Path | None = None
    interval_steps: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("memory", "disk"):
            raise ValueError(f"unknown REPLAY checkpoint backend: {self.backend}")
        if self.backend == "memory" and self.directory is not None:
            raise ValueError("directory is only valid for the disk checkpoint backend")
        if self.backend == "memory" and self.interval_steps is not None:
            raise ValueError("interval_steps is only valid for the disk checkpoint backend")
        if self.interval_steps is not None and self.interval_steps <= 0:
            raise ValueError("interval_steps must be positive")


@dataclass
class _ReplayStats:
    forward_steps: int = 0
    replay_steps: int = 0
    live_states: int = 1
    peak_live_states: int = 1
    yielded_state_indices: list[int] | None = None


class _CheckpointStorage(Protocol):
    def store(self, state: TrainState) -> object: ...

    def load(self, checkpoint: object) -> TrainState: ...

    def release(self, checkpoint: object) -> None: ...

    def close(self) -> None: ...


@dataclass
class _MemoryCheckpoint:
    state: TrainState | None


class _MemoryCheckpointStorage:
    def store(self, state: TrainState) -> object:
        return _MemoryCheckpoint(state)

    def load(self, checkpoint: object) -> TrainState:
        if not isinstance(checkpoint, _MemoryCheckpoint) or checkpoint.state is None:
            raise RuntimeError("invalid or released in-memory REPLAY checkpoint")
        return checkpoint.state

    def release(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, _MemoryCheckpoint):
            raise RuntimeError("invalid in-memory REPLAY checkpoint")
        checkpoint.state = None

    def close(self) -> None:
        return


@dataclass(frozen=True)
class _DiskCheckpoint:
    path: Path


def _checkpoint_payload(state: TrainState) -> dict[str, object]:
    def detached(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {name: value.detach() for name, value in values.items()}

    return {
        "parameters": detached(state.parameters),
        "buffers": detached(state.buffers),
        "first_moments": detached(state.first_moments),
        "second_moments": detached(state.second_moments),
        "step": state.step,
    }


def _state_from_checkpoint_payload(payload: object) -> TrainState:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    state = TrainState(
        parameters=dict(payload["parameters"]),
        buffers=dict(payload["buffers"]),
        first_moments=dict(payload["first_moments"]),
        second_moments=dict(payload["second_moments"]),
        step=int(payload["step"]),
    )
    return _detach_state(state, require_all=False)


class _DiskCheckpointStorage:
    def __init__(self, directory: str | Path | None) -> None:
        try:
            base_directory = None
            if directory is not None:
                base_directory = Path(directory)
                base_directory.mkdir(parents=True, exist_ok=True)
            self.run_directory = Path(
                tempfile.mkdtemp(prefix="replay-checkpoints-", dir=base_directory)
            )
        except OSError as error:
            raise RuntimeError("failed to create REPLAY checkpoint directory") from error
        self._counter = 0
        self._paths: set[Path] = set()
        self._closed = False

    def store(self, state: TrainState) -> object:
        if self._closed:
            raise RuntimeError("REPLAY checkpoint storage is closed")
        path = self.run_directory / f"{self._counter:08d}.pt"
        self._counter += 1
        try:
            torch.save(_checkpoint_payload(state), path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"failed to write REPLAY checkpoint: {path}") from error
        self._paths.add(path)
        return _DiskCheckpoint(path)

    def load(self, checkpoint: object) -> TrainState:
        if not isinstance(checkpoint, _DiskCheckpoint):
            raise RuntimeError("invalid disk REPLAY checkpoint")
        try:
            payload = torch.load(checkpoint.path, weights_only=True)
            return _state_from_checkpoint_payload(payload)
        except Exception as error:
            raise RuntimeError(
                f"failed to load REPLAY checkpoint: {checkpoint.path}"
            ) from error

    def release(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, _DiskCheckpoint):
            raise RuntimeError("invalid disk REPLAY checkpoint")
        try:
            checkpoint.path.unlink(missing_ok=True)
        except OSError as error:
            raise RuntimeError(
                f"failed to delete REPLAY checkpoint: {checkpoint.path}"
            ) from error
        self._paths.discard(checkpoint.path)

    def close(self) -> None:
        if self._closed:
            return
        try:
            shutil.rmtree(self.run_directory)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeError(
                f"failed to remove REPLAY checkpoint directory: {self.run_directory}"
            ) from error
        self._paths.clear()
        self._closed = True


def _checkpoint_storage(config: ReplayCheckpointConfig) -> _CheckpointStorage:
    if config.backend == "memory":
        return _MemoryCheckpointStorage()
    return _DiskCheckpointStorage(config.directory)


def _validate_branching_factor(branching_factor: int) -> None:
    if branching_factor < 2:
        raise ValueError("branching_factor must be at least 2")


def _flatten_diff_state(state: TrainState) -> tuple[Tensor, ...]:
    names = tuple(state.parameters)
    return (
        *(state.parameters[name] for name in names),
        *(state.first_moments[name] for name in names),
        *(state.second_moments[name] for name in names),
    )


def _unflatten_diff_state(
    flat: Sequence[Tensor],
    names: tuple[str, ...],
    buffers: dict[str, Tensor],
    step: int,
) -> TrainState:
    count = len(names)
    if len(flat) != 3 * count:
        raise ValueError("flattened state has an unexpected length")
    return TrainState(
        parameters=dict(zip(names, flat[:count])),
        buffers=buffers,
        first_moments=dict(zip(names, flat[count : 2 * count])),
        second_moments=dict(zip(names, flat[2 * count : 3 * count])),
        step=step,
    )


def _detach_state(state: TrainState, *, require_all: bool) -> TrainState:
    parameter_values = [
        value.detach().requires_grad_(True) for value in state.parameters.values()
    ]
    if require_all:
        first_moments = [
            value.detach().requires_grad_(True)
            for value in state.first_moments.values()
        ]
        second_moments = [
            value.detach().requires_grad_(True)
            for value in state.second_moments.values()
        ]
    else:
        first_moments = [value.detach() for value in state.first_moments.values()]
        second_moments = [value.detach() for value in state.second_moments.values()]
    return _unflatten_diff_state(
        (*parameter_values, *first_moments, *second_moments),
        tuple(state.parameters),
        state.buffers,
        state.step,
    )


def _balanced_boundaries(start: int, stop: int, branching_factor: int) -> list[int]:
    state_count = stop - start
    child_count = min(branching_factor, state_count)
    quotient, remainder = divmod(state_count, child_count)
    boundaries = [start]
    for child in range(child_count):
        boundaries.append(
            boundaries[-1] + quotient + (1 if child < remainder else 0)
        )
    return boundaries


def _lazy_reverse_states(
    initial_state: TrainState,
    total_steps: int,
    branching_factor: int,
    advance: Callable[[TrainState, int], TrainState],
    *,
    stats: _ReplayStats | None = None,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> Iterator[tuple[int, TrainState]]:
    """Yield ``s_T, ..., s_0`` using the paper's lazy balanced k-ary tree."""
    _validate_branching_factor(branching_factor)
    if total_steps < 0:
        raise ValueError("total_steps cannot be negative")

    if checkpoint_config.backend == "disk" and checkpoint_config.interval_steps:
        return _interval_disk_reverse_states(
            initial_state,
            total_steps,
            branching_factor,
            advance,
            stats=stats,
            checkpoint_config=checkpoint_config,
        )

    def generate() -> Iterator[tuple[int, TrainState]]:
        storage = _checkpoint_storage(checkpoint_config)

        def traverse(
            root_state: TrainState, start_state: int, stop_state: int
        ) -> Iterator[tuple[int, TrainState]]:
            if stop_state - start_state == 1:
                if stats is not None:
                    if stats.yielded_state_indices is not None:
                        stats.yielded_state_indices.append(start_state)
                yield start_state, root_state
                return

            boundaries = _balanced_boundaries(
                start_state, stop_state, branching_factor
            )
            child_roots: list[object] = []
            state = root_state
            cursor = start_state
            for target in boundaries[1:-1]:
                while cursor < target:
                    state = advance(state, cursor)
                    cursor += 1
                    if stats is not None:
                        stats.replay_steps += 1
                child_roots.append(storage.store(state))
            del state

            added_states = len(child_roots)
            if stats is not None:
                stats.live_states += added_states
                stats.peak_live_states = max(stats.peak_live_states, stats.live_states)
            try:
                for child in range(len(child_roots), -1, -1):
                    checkpoint = child_roots[child - 1] if child > 0 else None
                    child_root = (
                        storage.load(checkpoint)
                        if checkpoint is not None
                        else root_state
                    )
                    try:
                        yield from traverse(
                            child_root,
                            boundaries[child],
                            boundaries[child + 1],
                        )
                    finally:
                        if checkpoint is not None:
                            storage.release(checkpoint)
            finally:
                if stats is not None:
                    stats.live_states -= added_states

        try:
            yield from traverse(initial_state, 0, total_steps + 1)
        finally:
            storage.close()

    return generate()


def _interval_disk_reverse_states(
    initial_state: TrainState,
    total_steps: int,
    branching_factor: int,
    advance: Callable[[TrainState, int], TrainState],
    *,
    stats: _ReplayStats | None,
    checkpoint_config: ReplayCheckpointConfig,
) -> Iterator[tuple[int, TrainState]]:
    interval_steps = checkpoint_config.interval_steps
    if interval_steps is None:
        raise RuntimeError("interval disk traversal requires interval_steps")

    def generate() -> Iterator[tuple[int, TrainState]]:
        storage = _DiskCheckpointStorage(checkpoint_config.directory)
        if total_steps == 0:
            try:
                yield 0, initial_state
            finally:
                storage.close()
            return
        boundaries = list(range(0, total_steps, interval_steps))
        if not boundaries or boundaries[-1] != total_steps:
            boundaries.append(total_steps)
        checkpoints: dict[int, object] = {}
        state = initial_state
        cursor = 0
        try:
            for boundary in boundaries[1:-1]:
                while cursor < boundary:
                    state = advance(state, cursor)
                    cursor += 1
                    if stats is not None:
                        stats.replay_steps += 1
                checkpoints[boundary] = storage.store(state)
            del state

            for block in range(len(boundaries) - 2, -1, -1):
                start = boundaries[block]
                stop = boundaries[block + 1]
                checkpoint = checkpoints.get(start)
                root_state = (
                    initial_state if checkpoint is None else storage.load(checkpoint)
                )

                def block_advance(
                    block_state: TrainState, local_step: int
                ) -> TrainState:
                    return advance(block_state, start + local_step)

                block_stats = _ReplayStats() if stats is not None else None
                block_states = _lazy_reverse_states(
                    root_state,
                    stop - start,
                    branching_factor,
                    block_advance,
                    stats=block_stats,
                )
                try:
                    for local_index, block_state in block_states:
                        global_index = start + local_index
                        if block < len(boundaries) - 2 and global_index == stop:
                            continue
                        if stats is not None and stats.yielded_state_indices is not None:
                            stats.yielded_state_indices.append(global_index)
                        yield global_index, block_state
                finally:
                    close = getattr(block_states, "close", None)
                    if close is not None:
                        close()
                    if stats is not None and block_stats is not None:
                        stats.replay_steps += block_stats.replay_steps
                        stats.peak_live_states = max(
                            stats.peak_live_states, block_stats.peak_live_states
                        )
                    if checkpoint is not None:
                        storage.release(checkpoint)
        finally:
            storage.close()

    return generate()


class _RecursiveReplayEngine:
    def __init__(
        self,
        initial_state: TrainState,
        total_steps: int,
        branching_factor: int,
        step: ReplayStep,
        *,
        stats: _ReplayStats | None = None,
        progress_callback: ProgressCallback | None = None,
        checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
    ) -> None:
        if total_steps <= 0:
            raise ValueError("trajectory must contain at least one inner batch")
        _validate_branching_factor(branching_factor)
        self.initial_state = initial_state
        self.total_steps = total_steps
        self.branching_factor = branching_factor
        self.step = step
        self.stats = stats
        self.progress_callback = progress_callback
        self.checkpoint_config = checkpoint_config

    def _advance_without_graph(
        self, state: TrainState, step_index: int, metaparameter: Tensor
    ) -> TrainState:
        with torch.enable_grad():
            next_state = self.step(state, step_index, metaparameter, False)
        return _detach_state(next_state, require_all=False)

    def forward(self, metaparameter: Tensor) -> TrainState:
        state = _detach_state(self.initial_state, require_all=False)
        replay_metaparameter = metaparameter.detach()
        for step_index in range(self.total_steps):
            state = self._advance_without_graph(
                state, step_index, replay_metaparameter
            )
            if self.stats is not None:
                self.stats.forward_steps += 1
            if self.progress_callback is not None:
                self.progress_callback()
        return state

    def metagradient(
        self,
        metaparameter: Tensor,
        final_adjoint: Sequence[Tensor | None],
    ) -> Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "higher-order differentiation through recursive REPLAY is unsupported"
            )

        initial_state = _detach_state(self.initial_state, require_all=False)
        replay_metaparameter = metaparameter.detach()

        def advance(state: TrainState, step_index: int) -> TrainState:
            return self._advance_without_graph(
                state, step_index, replay_metaparameter
            )

        reverse_states = _lazy_reverse_states(
            initial_state,
            self.total_steps,
            self.branching_factor,
            advance,
            stats=self.stats,
            checkpoint_config=self.checkpoint_config,
        )
        try:
            final_index, final_state = next(reverse_states)
            if final_index != self.total_steps:
                raise RuntimeError("recursive REPLAY did not yield the final state first")

            final_flat = _flatten_diff_state(final_state)
            adjoint = tuple(
                torch.zeros_like(value) if gradient is None else gradient
                for value, gradient in zip(final_flat, final_adjoint, strict=True)
            )
            metagradient = torch.zeros_like(metaparameter)

            with torch.enable_grad():
                for expected_step in range(self.total_steps - 1, -1, -1):
                    state_index, replayed_state = next(reverse_states)
                    if state_index != expected_step:
                        raise RuntimeError(
                            "recursive REPLAY yielded states out of reverse order"
                        )

                    state_leaf = _detach_state(replayed_state, require_all=True)
                    metaparameter_leaf = metaparameter.detach().requires_grad_(True)
                    next_state = self.step(
                        state_leaf,
                        expected_step,
                        metaparameter_leaf,
                        True,
                    )
                    gradients = torch.autograd.grad(
                        _flatten_diff_state(next_state),
                        (*_flatten_diff_state(state_leaf), metaparameter_leaf),
                        grad_outputs=adjoint,
                        allow_unused=True,
                        materialize_grads=True,
                    )
                    adjoint = gradients[:-1]
                    metagradient = metagradient + gradients[-1]
                    if self.progress_callback is not None:
                        self.progress_callback()

            return metagradient
        finally:
            close = getattr(reverse_states, "close", None)
            if close is not None:
                close()


class _RecursiveReplayState(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        metaparameter: Tensor,
        engine: _RecursiveReplayEngine,
    ) -> tuple[Tensor, ...]:
        final_state = engine.forward(metaparameter)
        ctx.engine = engine
        ctx.names = tuple(final_state.parameters)
        ctx.buffers = final_state.buffers
        ctx.step = final_state.step
        ctx.save_for_backward(metaparameter.detach(), *_flatten_diff_state(final_state))
        return _flatten_diff_state(final_state)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        *grad_outputs: Tensor | None,
    ) -> tuple[Tensor | None, None]:
        saved = ctx.saved_tensors
        metaparameter = saved[0]
        final_state = _unflatten_diff_state(
            saved[1:],
            ctx.names,
            ctx.buffers,
            ctx.step,
        )
        final_flat = _flatten_diff_state(final_state)
        if len(grad_outputs) != len(final_flat):
            raise RuntimeError("recursive REPLAY received an unexpected state adjoint")
        metagradient = ctx.engine.metagradient(metaparameter, grad_outputs)
        return metagradient, None


def recursive_replay_state(
    initial_state: TrainState,
    metaparameter: Tensor,
    total_steps: int,
    step: ReplayStep,
    *,
    branching_factor: int,
    stats: _ReplayStats | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_config: ReplayCheckpointConfig = ReplayCheckpointConfig(),
) -> TrainState:
    """Return the final state with a lazy k-ary REPLAY backward pass."""
    engine = _RecursiveReplayEngine(
        initial_state,
        total_steps,
        branching_factor,
        step,
        stats=stats,
        progress_callback=progress_callback,
        checkpoint_config=checkpoint_config,
    )
    flat = _RecursiveReplayState.apply(metaparameter, engine)
    return _unflatten_diff_state(
        flat,
        tuple(initial_state.parameters),
        initial_state.buffers,
        initial_state.step + total_steps,
    )
