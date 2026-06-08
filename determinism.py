from __future__ import annotations

import os
import random

import numpy as np
import torch


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _set_tf32(tf32: bool) -> None:
    mode = "tf32" if tf32 else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.fp32_precision = mode
    else:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32


def _tf32_matches(tf32: bool) -> bool:
    mode = "tf32" if tf32 else "ieee"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return (
            torch.backends.fp32_precision == mode
            and torch.backends.cuda.matmul.fp32_precision == mode
            and torch.backends.cudnn.fp32_precision == mode
        )
    return (
        torch.backends.cuda.matmul.allow_tf32 is tf32
        and torch.backends.cudnn.allow_tf32 is tf32
    )


def configure_replay_determinism(seed: int, *, tf32: bool) -> None:
    """Configure strict same-machine determinism before running REPLAY on CUDA."""
    if (
        torch.cuda.is_initialized()
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _set_tf32(tf32)


def assert_replay_determinism(*, tf32: bool, require_cuda: bool = True) -> None:
    """Raise when the process is not configured for deterministic REPLAY."""
    failures: list[str] = []
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        failures.append("CUDA is required but unavailable")
    if not torch.are_deterministic_algorithms_enabled():
        failures.append("deterministic algorithms are disabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        failures.append("deterministic algorithms are in warn-only mode")
    if torch.backends.cudnn.benchmark:
        failures.append("cuDNN benchmarking is enabled")
    if not torch.backends.cudnn.deterministic:
        failures.append("cuDNN deterministic mode is disabled")
    if cuda_available or require_cuda:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
            failures.append("CUBLAS_WORKSPACE_CONFIG is not :4096:8")
        if not _tf32_matches(tf32):
            failures.append(f"TF32 configuration does not match tf32={tf32}")
    if failures:
        raise RuntimeError(
            "invalid REPLAY determinism configuration: " + "; ".join(failures)
        )
