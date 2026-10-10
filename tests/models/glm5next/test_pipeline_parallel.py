# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM stage boundaries and loading contracts, without allocating model weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.common import model as glm_model
from vllm.models.glm5next.common import mtp as glm_mtp
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_utils import PPHandler


class _Layer:
    """Small deferred residual layer that exposes the mHC boundary contract."""

    n = 2

    def __init__(self, index):
        self.layer_idx = index

    def hc_post(self, hidden, residual, post, comb):
        return residual + hidden[:, None, :]

    def __call__(self, positions, hidden, residual, post, comb):
        if post is not None:
            hidden = self.hc_post(hidden, residual, post, comb)
        elif self.layer_idx == 0:
            hidden = hidden[:, None, :].expand(-1, self.n, -1)
        residual = hidden
        update = torch.full_like(hidden[:, 0], self.layer_idx + 1)
        if self.layer_idx == 2:
            return self.hc_post(update, residual, None, None).mean(1), None, None, None
        return update, residual, torch.ones(1), torch.ones(1)


def _model(start=0, end=3, mhc=True):
    model = object.__new__(glm_model.Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        mhc=mhc, mhc_num_residual_streams=2, hidden_size=4, num_hidden_layers=3
    )
    model.start_layer = start
    model.end_layer = end
    model._active_layers = [_Layer(i) for i in range(start, end)]
    model.is_sequence_parallel = False
    model.norm = lambda hidden: hidden * 2
    model.aux_hidden_state_layers = (1, 3)
    model._aux_slot_base_cached = 1 if start else 0
    model._aux_upstream_total_cached = 1 if start and end == 3 else 0
    return model


@pytest.mark.parametrize("mhc", [True, False])
def test_intermediate_buffer_matches_stage_residual_shape(mhc):
    tensors = _model(mhc=mhc).make_empty_intermediate_tensors(
        5, torch.bfloat16, torch.device("cpu")
    )
    assert tensors["residual"].shape == ((5, 2, 4) if mhc else (5, 4))
    assert tensors["residual"].dtype == torch.bfloat16
    if not mhc:
        assert tensors["hidden_states"].shape == (5, 4)


@pytest.mark.parametrize("boundaries", [(0, 2, 3), (0, 1, 2, 3)])
def test_materialized_mhc_boundary_preserves_output_and_aux_states(
    monkeypatch, boundaries
):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(glm_model, "get_pp_group", lambda: group)
    monkeypatch.setattr(glm_model, "hc_contract", lambda hidden, n: hidden.mean(1))
    embeddings = torch.arange(8, dtype=torch.float32).view(2, 4)
    positions = torch.arange(2)
    expected, expected_aux = _model().forward(None, positions, None, embeddings)
    intermediate = None
    for start, end in zip(boundaries, boundaries[1:]):
        group.is_first_rank = start == 0
        group.is_last_rank = end == 3
        output = _model(start, end).forward(
            None, positions, intermediate, embeddings if start == 0 else None
        )
        if end < 3:
            assert output["residual"].shape == (2, 2, 4)
            assert "hidden_states" not in output.tensors
            relay = object.__new__(PPHandler)
            relay.aux_hidden_state_relay_keys = (
                ("aux_hidden_states_0",) if start else ()
            )
            intermediate = relay.relay_aux_hidden_states(intermediate, output)
        else:
            actual, actual_aux = output
    torch.testing.assert_close(actual, expected)
    assert len(actual_aux) == len(expected_aux) == 2
    for actual_state, expected_state in zip(actual_aux, expected_aux):
        torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.parametrize("layers", [(0,), (4,), (1, 1)])
def test_invalid_auxiliary_boundaries_fail_before_capture(layers):
    with pytest.raises(ValueError, match="Invalid GLM"):
        _model()._set_aux_hidden_state_layers(layers)


def test_weight_loader_skips_other_stages_before_fp8_conversion(monkeypatch):
    model = _model()
    model.config.is_moe = False
    model.config.mla_nope = False
    seen = []
    monkeypatch.setattr(
        glm_model, "is_pp_missing_parameter", lambda name, model: "layers.2." in name
    )
    monkeypatch.setattr(
        glm_model, "get_spec_layer_idx_from_weight_name", lambda *args: None
    )

    def record_conversion(name, *args):
        seen.append(name)
        return True

    monkeypatch.setattr(glm_model, "_try_load_fp8_indexer_wk", record_conversion)
    model.load_weights(
        [(f"layers.{i}.indexer.wk.weight", torch.ones(1)) for i in [0, 2]]
    )
    assert seen == ["layers.0.indexer.wk.weight"]


def test_last_stage_mtp_accepts_hidden_state_intermediate():
    model = object.__new__(glm_mtp.Glm5NextMTP)
    nn.Module.__init__(model)
    model.model = lambda ids, pos, hidden, embeds, step: hidden + step
    hidden = torch.ones(2, 4)
    actual = model.forward(
        None,
        torch.arange(2),
        intermediate_tensors=IntermediateTensors({"hidden_states": hidden}),
        spec_step_idx=2,
    )
    torch.testing.assert_close(actual, hidden + 2)
