from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from determinism import assert_replay_determinism, configure_replay_determinism


@pytest.mark.cpu
def test_configure_replay_determinism_reseeds_all_host_rngs() -> None:
    configure_replay_determinism(17, tf32=False)
    first = (random.random(), np.random.rand(), torch.rand(()))

    configure_replay_determinism(17, tf32=False)
    second = (random.random(), np.random.rand(), torch.rand(()))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert_replay_determinism(tf32=False, require_cuda=False)
