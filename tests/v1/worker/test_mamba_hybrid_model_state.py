# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
@pytest.mark.parametrize("per_request_counts", [False, True])
def test_postprocess_state_skips_filtered_request_slots(
    num_sampled: int, expected_value: int, per_request_counts: bool
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    counts = (
        torch.full((3,), num_sampled, dtype=torch.int32, device="cuda")
        if per_request_counts
        else num_sampled
    )
    state.postprocess_state(idx_mapping, counts)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)
